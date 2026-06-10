"""
CSM-1B sub-model loaders for Olive PyTorchModel.

CsmForConditionalGeneration has three sub-models:

  backbone_model   — Llama-1B (16L, h=2048)
                     Input:  input_ids [B,T] int64, attention_mask [B,T] int64
                     Output: hidden_states [B,T,2048] float32

  depth_decoder    — Llama-100M (4L, h=1024)
                     Input:  backbone_hidden [B,1,2048] float32,
                             input_ids [B,1] int64,
                             position_ids [B,1] int64
                     Output: logits [B,1,2051] float32

  codec_encoder    — Mimi audio encoder
                     Input:  audio [B,1,T_wav] float32
                     Output: codes [B,32,T_frames] int64

  codec_decoder    — Mimi audio decoder
                     Input:  codes [B,32,T_frames] int64
                     Output: audio [B,1,T_wav] float32

Each wrapper forces use_cache=False and returns plain tensors so
torch.onnx.export (TorchScript tracing) can handle them without
encountering DynamicCache or other non-tensor types.
"""

import os
import warnings
import torch
import torch.nn as nn
from transformers import CsmForConditionalGeneration


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_csm(model_path: str) -> CsmForConditionalGeneration:
    if not os.path.isabs(model_path):
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), model_path
        )
    # CSM uses dual vocabularies (audio: 2051, text: 128256).
    # Transformers' config validator flags the Llama special tokens (128000+)
    # as out-of-range against vocab_size=2051 — these warnings are harmless.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*must be.*None.*or an integer within the vocabulary.*")
        model = CsmForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32
        )
    model.eval()
    return model


# ════════════════════════════════════════════════════════════════════════════
# 1. BACKBONE
# ════════════════════════════════════════════════════════════════════════════

class BackboneWrapper(nn.Module):
    """
    Exports CSM's backbone (Llama-1B) as a standalone ONNX model.
    Returns hidden_states only — no KV cache, no logits.
    """
    def __init__(self, csm: CsmForConditionalGeneration):
        super().__init__()
        # Sub-modules are direct attributes of CsmForConditionalGeneration
        # (there is no intermediate `.model`).
        self.backbone = csm.backbone_model

    def forward(
        self,
        input_ids: torch.Tensor,       # [B, T]  int64
        attention_mask: torch.Tensor,  # [B, T]  int64
    ) -> torch.Tensor:                 # [B, T, 2048]  float32
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.last_hidden_state


def get_backbone_model(model_path: str) -> BackboneWrapper:
    return BackboneWrapper(_load_csm(model_path))


def get_backbone_io_config(model) -> dict:
    # input_ids is 3D [B, T, num_codebooks=32]: the backbone embeds each
    # codebook token (with per-codebook offsets) and sums across codebooks.
    return {
        "input_names":  ["input_ids", "attention_mask"],
        "output_names": ["hidden_states"],
        "input_shapes":  [[1, 64, 32], [1, 64]],
        "input_types":   ["int64", "int64"],
        "dynamic_axes": {
            "input_ids":      {"0": "batch_size", "1": "sequence_length"},
            "attention_mask": {"0": "batch_size", "1": "sequence_length"},
            "hidden_states":  {"0": "batch_size", "1": "sequence_length"},
        },
    }


def get_backbone_dummy_inputs(model):
    return (
        torch.randint(0, 100, (1, 64, 32), dtype=torch.long),
        torch.ones(1, 64, dtype=torch.long),
    )


# ════════════════════════════════════════════════════════════════════════════
# 2. DEPTH DECODER
# ════════════════════════════════════════════════════════════════════════════

class DepthDecoderWrapper(nn.Module):
    """
    Exports CSM's depth decoder (Llama-100M) as a standalone ONNX model.
    Takes one backbone hidden state vector + current codebook token id and
    returns logits for the next codebook level.
    """
    def __init__(self, csm: CsmForConditionalGeneration):
        super().__init__()
        self.depth_decoder = csm.depth_decoder

    def forward(
        self,
        backbone_hidden: torch.Tensor,  # [B, 2048]         float32
        input_ids: torch.Tensor,        # [B, num_codebooks] int64
        attention_mask: torch.Tensor,   # [B, num_codebooks] int64
    ) -> torch.Tensor:                  # [B, num_codebooks-1, vocab]  float32
        # backbone_last_hidden_state is 2D [B, backbone_hidden_size]; it is
        # spliced into position 0 of the codebook sequence. An explicit
        # attention_mask is required, otherwise the masking utility tries to
        # detect a packed-sequence layout and crashes on the 1D position_ids
        # the decoder builds internally. position_ids must NOT be passed
        # (the depth decoder derives them itself and warns if given any).
        out = self.depth_decoder(
            input_ids=input_ids,
            backbone_last_hidden_state=backbone_hidden,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def get_depth_decoder_model(model_path: str) -> DepthDecoderWrapper:
    return DepthDecoderWrapper(_load_csm(model_path))


def get_depth_decoder_io_config(model) -> dict:
    # 32 codebooks per frame → input_ids/attention_mask are [B, 32].
    return {
        "input_names":  ["backbone_hidden", "input_ids", "attention_mask"],
        "output_names": ["logits"],
        "input_shapes":  [[1, 2048], [1, 32], [1, 32]],
        "input_types":   ["float32", "int64", "int64"],
        "dynamic_axes": {
            "backbone_hidden": {"0": "batch_size"},
            "input_ids":       {"0": "batch_size", "1": "num_codebooks"},
            "attention_mask":  {"0": "batch_size", "1": "num_codebooks"},
            "logits":          {"0": "batch_size", "1": "num_codebooks"},
        },
    }


def get_depth_decoder_dummy_inputs(model):
    return (
        torch.randn(1, 2048, dtype=torch.float32),
        torch.randint(0, 100, (1, 32), dtype=torch.long),
        torch.ones(1, 32, dtype=torch.long),
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. CODEC ENCODER  (audio waveform → RVQ codes)
# ════════════════════════════════════════════════════════════════════════════

class CodecEncoderWrapper(nn.Module):
    """
    Exports Mimi's encoder: raw audio PCM → discrete RVQ code indices.
    Input sampling rate: 24 000 Hz, output frame rate: 12.5 fps.
    """
    def __init__(self, csm: CsmForConditionalGeneration):
        super().__init__()
        self.codec = csm.codec_model

    def forward(
        self,
        audio: torch.Tensor,   # [B, 1, T_wav]  float32
    ) -> torch.Tensor:         # [B, 32, T_frames]  int64
        # Mirror MimiModel._encode_frame directly. The high-level encode()
        # wrapper computes frame counts / padding caches with data-dependent
        # ops (surfacing as aten::diff) that neither ONNX exporter can trace;
        # the per-frame conv+transformer+quantizer path below is clean.
        c = self.codec
        embeddings = c.encoder(audio)
        hidden = embeddings.transpose(1, 2)      # [B, T_frames, dim]
        # Explicit attention_mask avoids the create_causal_mask path that calls
        # find_packed_sequence_indices → torch.diff (unsupported by the ONNX
        # TorchScript exporter). Same root cause as the depth-decoder fix.
        attn = torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.long)
        enc = c.encoder_transformer(
            hidden, attention_mask=attn, use_cache=False, return_dict=True
        )
        embeddings = enc[0].transpose(1, 2)
        embeddings = c.downsample(embeddings)
        codes = c.quantizer.encode(embeddings)   # [num_q, B, T_frames]
        codes = codes.transpose(0, 1)            # [B, num_q, T_frames]
        return codes


def get_codec_encoder_model(model_path: str) -> CodecEncoderWrapper:
    return CodecEncoderWrapper(_load_csm(model_path))


def get_codec_encoder_io_config(model) -> dict:
    # The time axis is kept STATIC (24000 samples = 1 s @ 24 kHz). Mimi's
    # SEANet conv stack computes frame counts from the sample length; with a
    # dynamic time axis the dynamo exporter cannot guard the data-dependent
    # conv-output-length formula. Only the batch axis is dynamic. Feed the
    # codec fixed-size 1-second chunks at inference.
    return {
        "input_names":  ["audio"],
        "output_names": ["audio_codes"],
        "input_shapes":  [[1, 1, 24000]],   # 1 second of audio at 24kHz
        "input_types":   ["float32"],
        "dynamic_axes": {
            "audio":       {"0": "batch_size"},
            "audio_codes": {"0": "batch_size"},
        },
    }


def get_codec_encoder_dummy_inputs(model):
    return (torch.randn(1, 1, 24000, dtype=torch.float32),)


# ════════════════════════════════════════════════════════════════════════════
# 4. CODEC DECODER  (RVQ codes → audio waveform)
# ════════════════════════════════════════════════════════════════════════════

class CodecDecoderWrapper(nn.Module):
    """
    Exports Mimi's decoder: discrete RVQ code indices → reconstructed PCM audio.
    """
    def __init__(self, csm: CsmForConditionalGeneration):
        super().__init__()
        self.codec = csm.codec_model

    def forward(
        self,
        audio_codes: torch.Tensor,  # [B, 32, T_frames]  int64
    ) -> torch.Tensor:              # [B, 1, T_wav]       float32
        # Mirror MimiModel._decode_frame directly (see encoder note above).
        c = self.codec
        embeddings = c.quantizer.decode(audio_codes)
        embeddings = c.upsample(embeddings)
        hidden = embeddings.transpose(1, 2)      # [B, T_frames, dim]
        attn = torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.long)
        dec = c.decoder_transformer(
            hidden, attention_mask=attn, use_cache=False, return_dict=True
        )
        embeddings = dec[0].transpose(1, 2)
        audio = c.decoder(embeddings)
        return audio


def get_codec_decoder_model(model_path: str) -> CodecDecoderWrapper:
    return CodecDecoderWrapper(_load_csm(model_path))


def get_codec_decoder_io_config(model) -> dict:
    return {
        "input_names":  ["audio_codes"],
        "output_names": ["audio"],
        "input_shapes":  [[1, 32, 13]],   # ~1 second at 12.5fps
        "input_types":   ["int64"],
        "dynamic_axes": {
            "audio_codes": {"0": "batch_size"},
            "audio":       {"0": "batch_size"},
        },
    }


def get_codec_decoder_dummy_inputs(model):
    return (torch.randint(0, 2048, (1, 32, 13), dtype=torch.long),)


# ════════════════════════════════════════════════════════════════════════════
# 5. FULL MODEL (kept for compatibility / one-shot export)
# ════════════════════════════════════════════════════════════════════════════

class CsmTraceableWrapper(nn.Module):
    def __init__(self, model: CsmForConditionalGeneration):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


def get_csm_model(model_path: str) -> CsmTraceableWrapper:
    return CsmTraceableWrapper(_load_csm(model_path))


def get_csm_io_config(model) -> dict:
    return {
        "input_names":  ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "input_shapes":  [[1, 64], [1, 64]],
        "input_types":   ["int64", "int64"],
        "dynamic_axes": {
            "input_ids":      {"0": "batch_size", "1": "sequence_length"},
            "attention_mask": {"0": "batch_size", "1": "sequence_length"},
            "logits":         {"0": "batch_size", "1": "sequence_length"},
        },
    }


def get_csm_dummy_inputs(model):
    return (
        torch.randint(0, 100, (1, 64), dtype=torch.long),
        torch.ones(1, 64, dtype=torch.long),
    )
