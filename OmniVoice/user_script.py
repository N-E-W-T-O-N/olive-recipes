"""
Olive user_script for Prince-1/OmniVoice ONNX export.

Backbone sub-models (Olive JSON configs in cpu_and_mobile/, cuda/, cpu_fp16/):
  - get_audio_embeddings_model   → audio_embeddings_encoder.onnx
  - get_audio_heads_model        → audio_heads_decoder.onnx
  - (llm_decoder via ModelBuilder in llm_decoder.json)

Higgs Audio V2 Tokenizer (Olive JSON configs in higgs/):
  - get_higgs_acoustic_model     → acoustic_encoder.onnx
  - get_higgs_semantic_model     → semantic_encoder.onnx
  - get_higgs_quantizer_model    → quantizer_encoder.onnx
  - get_higgs_decoder_model      → higgs_decoder.onnx

HiggsAudioV2TokenizerModel is natively in transformers >= 5.4.0 —
no external packages needed.
https://huggingface.co/docs/transformers/v5.4.0/en/model_doc/higgs_audio_v2_tokenizer

The DAC acoustic encoder and decoder contain Python control-flow branches that
torch.onnx.export cannot resolve without pre-tracing. get_higgs_acoustic_model
and get_higgs_decoder_model return torch.jit.trace() results so Olive sees a
branch-free graph.

OmniVoice backbone constants (from config.json):
  HIDDEN_SIZE   = 1024
  NUM_CODEBOOKS = 8
  AUDIO_VOCAB   = 1025
Higgs Audio constants (from audio_tokenizer/config.json + preprocessor_config.json):
  SR_24K = 24 000 Hz   (acoustic sample rate)
  SR_16K = 16 000 Hz   (HuBERT semantic sample rate)
  DOWNSAMPLE_FACTOR = 960  (product of DAC downsampling_ratios [8,5,4,2,3]; 24000/960=25 fps)
"""

import os
import sys
import torch
import torch.nn as nn

# Add this script's directory to sys.path for codes module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from codes.model_wrappers import (
    AudioEmbeddingsEncoderWrapper,
    AudioHeadsDecoderWrapper,
    HIDDEN_SIZE,
    NUM_CODEBOOKS,
    AUDIO_VOCAB,
)

model_name = "Prince-1/OmniVoice"


# =============================================================================
# Model loading
# =============================================================================

def _load_omnivoice(model_path: str):
    """Load the full OmniVoice model with trust_remote_code.

    train=True is required by OmniVoice.from_pretrained to initialise
    all sub-modules correctly for export (some modules behave differently
    in eval-only mode before export).
    attn_implementation='eager' avoids FlashAttention which is not ONNX-traceable.
    """
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    return model


# =============================================================================
# Audio Embeddings Encoder
# =============================================================================

def get_audio_embeddings_model(model_path=None):
    """Return AudioEmbeddingsEncoderWrapper for ONNX export.

    Wraps OmniVoice._prepare_embed_inputs() — text + audio embedding fusion.
    The Qwen3 text embeddings and OmniVoice audio embeddings are extracted
    and combined into a single ONNX-exportable module.
    """
    model = _load_omnivoice(model_path or model_name)
    wrapper = AudioEmbeddingsEncoderWrapper(
        text_embed    = model.get_input_embeddings(),
        audio_embed   = model.audio_embeddings,
        layer_offsets = model.codebook_layer_offsets,
    )
    wrapper.eval()
    return wrapper


def get_audio_embeddings_io_config(model_path=None):
    return {
        "input_names":  ["input_ids", "audio_mask"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": {
            "input_ids":     {0: "batch", 2: "seq"},
            "audio_mask":    {0: "batch", 1: "seq"},
            "inputs_embeds": {0: "batch", 1: "seq"},
        },
    }


def get_audio_embeddings_dummy_inputs(model=None):
    """Dummy inputs for audio_embeddings_encoder ONNX export.

    Uses B=1, S=64 (representative sequence with mixed text + audio tokens).
    Codebook IDs are in [0, AUDIO_VOCAB) for audio positions, [0, text_vocab)
    for text positions — we use a single range for simplicity.
    """
    B, S = 1, 64
    input_ids  = torch.randint(0, AUDIO_VOCAB, (B, NUM_CODEBOOKS, S), dtype=torch.int64)
    audio_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_mask[:, S // 4 : S * 3 // 4] = True   # middle 50% are audio tokens
    return {"input_ids": input_ids, "audio_mask": audio_mask}


# =============================================================================
# Audio Heads Decoder
# =============================================================================

def get_audio_heads_model(model_path=None):
    """Return AudioHeadsDecoderWrapper for ONNX export.

    Wraps model.audio_heads (nn.Linear 1024 → 8*1025) + reshape.
    This is SEPARATE from the Qwen3 LLM — it projects hidden_states
    to per-codebook audio-token logits.
    """
    model = _load_omnivoice(model_path or model_name)
    wrapper = AudioHeadsDecoderWrapper(heads=model.audio_heads)
    wrapper.eval()
    return wrapper


def get_audio_heads_io_config(model_path=None):
    return {
        "input_names":  ["hidden_states"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "hidden_states": {0: "batch", 1: "seq"},
            "logits":        {0: "batch", 2: "seq"},
        },
    }


def get_audio_heads_dummy_inputs(model=None):
    """Dummy inputs for audio_heads_decoder ONNX export.

    B=1, S=64 representative sequence length for iterative decoding.
    """
    B, S = 1, 64
    return {"hidden_states": torch.randn(B, S, HIDDEN_SIZE, dtype=torch.float32)}


# =============================================================================
# LLM utility: save Qwen3 standalone for ModelBuilder
# =============================================================================

def save_qwen3_standalone(model_path: str, output_dir: str) -> str:
    """Save the Qwen3 LLM as a standalone HuggingFace directory.

    onnxruntime-genai's ModelBuilder (create_model) needs to recognise the
    model as Qwen3ForCausalLM.  OmniVoice's internal LLM config may not have
    the right architectures[] field, so we patch it after save_pretrained().

    output_dir is the PARENT of qwen3_standalone/ — e.g. pass "cpu_fp16" and
    the weights are saved to "cpu_fp16/qwen3_standalone/".
    optimize.py always passes the resolved absolute parent directory.

    Returns the path to the saved directory.
    """
    import json
    from pathlib import Path

    model = _load_omnivoice(model_path)
    qwen3_dir = Path(output_dir) / "qwen3_standalone"
    qwen3_dir.mkdir(parents=True, exist_ok=True)

    # Save weights + config
    model.llm.save_pretrained(str(qwen3_dir))
    print(f"Saved Qwen3 weights to {qwen3_dir}")

    # Patch config.json: architectures must be ["Qwen3ForCausalLM"]
    cfg_path = qwen3_dir / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["architectures"] = ["Qwen3ForCausalLM"]
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f'Patched config.json → "architectures": ["Qwen3ForCausalLM"]')

    del model
    return str(qwen3_dir)


# =============================================================================
# Higgs Audio V2 Tokenizer — loading + sub-model loaders
# =============================================================================

from codes.model_wrappers import (
    HiggsAcousticEncoderWrapper,
    HiggsSemanticEncoderWrapper,
    HiggsQuantizerEncoderWrapper,
    HiggsDecoderWrapper,
    SR_24K, SR_16K, DOWNSAMPLE_FACTOR,
    HIGGS_D_ACOUSTIC, HIGGS_D_SEMANTIC, HIGGS_N_CB, HIGGS_CB_SIZE,
)

_AUDIO_TOK_SUBDIR = "audio_tokenizer"   # relative to OmniVoice model root


def _load_higgs_tokenizer(model_path: str):
    """Load HiggsAudioV2TokenizerModel from the audio_tokenizer/ subfolder.

    HiggsAudioV2TokenizerModel is natively supported in transformers >= 5.4.0
    (https://huggingface.co/docs/transformers/v5.4.0/en/model_doc/higgs_audio_v2_tokenizer).
    No external packages (boson_multimodal etc.) are required.

    Transformers attribute mapping used by our wrappers:
      tok.acoustic_encoder  — DAC acoustic encoder (nn.Module)
      tok.semantic_model    — HuBERT model (nn.Module)
      tok.encoder_semantic  — semantic projection conv (nn.Module)
      tok.fc                — linear projection before RVQ (nn.Module)
      tok.quantizer         — HiggsAudioV2TokenizerResidualVectorQuantization
      tok.fc2               — linear projection after RVQ decode (nn.Module)
      tok.acoustic_decoder  — DAC decoder (nn.Module)
    """
    from huggingface_hub import snapshot_download
    from pathlib import Path
    from transformers import AutoModel

    # Resolve the model root directory
    local_root = Path(model_path)
    if not local_root.is_dir():
        local_root = Path(snapshot_download(model_path))

    audio_tok_dir = local_root / _AUDIO_TOK_SUBDIR
    if not audio_tok_dir.is_dir():
        raise FileNotFoundError(
            f"audio_tokenizer/ not found under {local_root}. "
            "The full OmniVoice model must be downloaded (not just config.json)."
        )

    tok = AutoModel.from_pretrained(
        str(audio_tok_dir),
        torch_dtype=torch.float32,
    )
    tok.eval()
    print(f"  [Higgs] Loaded HiggsAudioV2TokenizerModel from {audio_tok_dir}")
    return tok


# ---------------------------------------------------------------------------
# 1. acoustic_encoder  (B, 1, T_24k) → (B, D_a, T_frames)
# ---------------------------------------------------------------------------

def _prepare_tok(tok):
    """Strip weight_norm and detach all grad tensors on the full tokenizer.

    Must be called on the full tok object BEFORE extracting any bound methods
    (e.g. tok.encode) because:
      - tok.encode is a bound method (function), not an nn.Module — you cannot
        call _strip_weight_norm() on it directly.
      - weight_norm hooks live on tok's Conv layers; stripping tok cleans them
        all before any sub-module is handed to a wrapper.
    """
    from codes.model_wrappers import _strip_weight_norm
    tok.eval()
    _strip_weight_norm(tok)
    tok.requires_grad_(False)
    # Detach any remaining plain tensor attributes (weight_norm leaves computed
    # weights as plain tensors with requires_grad=True after removal)
    for sub in tok.modules():
        for attr_name in list(vars(sub)):
            v = getattr(sub, attr_name, None)
            if (isinstance(v, torch.Tensor)
                    and not isinstance(v, torch.nn.Parameter)
                    and v.requires_grad):
                setattr(sub, attr_name, v.detach())
    return tok


def get_higgs_acoustic_model(model_path=None):
    """Return HiggsAcousticEncoderWrapper (nn.Module) for ONNX export via Olive.

    Uses tok.acoustic_encoder (the DAC nn.Module) — NOT tok.encode (the full
    encode method that processes both 24kHz acoustic AND 16kHz semantic audio).
    tok.acoustic_encoder is a proper nn.Module so weight_norm can be stripped on
    it directly (unlike tok.encode which is a bound method).

    Transformers HiggsAudioV2TokenizerModel attribute: self.acoustic_encoder
    (created via AutoModel from config.acoustic_model_config).

    TracerWarnings about Python branches (if channels != 1, if padding > 0) are
    benign — our dummy always has channels=1, branches resolve to constants.
    """
    tok = _load_higgs_tokenizer(model_path or model_name)
    tok = _prepare_tok(tok)
    wrapper = HiggsAcousticEncoderWrapper(tok.acoustic_encoder)
    wrapper.eval()
    return wrapper


def get_higgs_acoustic_io_config(model_path=None):
    return {
        "input_names":  ["waveform_24k"],
        "output_names": ["acoustic_features"],
        "dynamic_axes": {
            "waveform_24k":       {0: "batch", 2: "samples"},
            "acoustic_features":  {0: "batch", 2: "frames"},
        },
    }


def get_higgs_acoustic_dummy_inputs(model=None):
    """1 second mono 24 kHz audio."""
    return {"waveform_24k": torch.randn(1, 1, SR_24K, dtype=torch.float32)}


# ---------------------------------------------------------------------------
# 2. semantic_encoder  (B, T_16k) → (B, D_s, T_frames)
# ---------------------------------------------------------------------------

def get_higgs_semantic_model(model_path=None):
    """HuBERT semantic encoder — replicates _extract_semantic_features() exactly.

    Passes downsample_factor and pad from tok.config so the wrapper matches
    the real encode() pipeline precisely:
      - pad=160      : F.pad(input_values, (160, 160)) before HuBERT
      - all hidden states averaged (not last_hidden_state)
      - downsample_factor=2 : every-other-frame slice after averaging
    """
    tok = _load_higgs_tokenizer(model_path or model_name)
    tok = _prepare_tok(tok)
    wrapper = HiggsSemanticEncoderWrapper(
        tok.semantic_model,
        tok.encoder_semantic,
        downsample_factor=getattr(tok.config, "semantic_downsample_factor", 2),
        pad=160,   # hardcoded in _extract_semantic_features: F.pad(..., (160, 160))
    )
    wrapper.eval()
    return wrapper


def get_higgs_semantic_io_config(model_path=None):
    return {
        "input_names":  ["waveform_16k"],
        "output_names": ["semantic_features"],
        "dynamic_axes": {
            "waveform_16k":      {0: "batch", 1: "samples"},
            "semantic_features": {0: "batch", 2: "frames"},
        },
    }


def get_higgs_semantic_dummy_inputs(model=None):
    """1 second mono 16 kHz audio (no channel dimension — HuBERT expects (B, T))."""
    return {"waveform_16k": torch.randn(1, SR_16K, dtype=torch.float32)}


# ---------------------------------------------------------------------------
# 3. quantizer_encoder  acoustic + semantic → codes
# ---------------------------------------------------------------------------

def get_higgs_quantizer_model(model_path=None):
    """RVQ encoder — maps fused acoustic+semantic features to discrete codec codes.

    Transformers HiggsAudioV2TokenizerModel attribute mapping (confirmed from source):
      boson_multimodal    transformers
      tok.fc_prior   →   tok.fc       (Linear: hidden → hidden, projects merged features)
      tok.quantizer  →   tok.quantizer (HiggsAudioV2TokenizerResidualVectorQuantization)

    The full encode pipeline is: concat(acoustic, semantic) → tok.fc → tok.quantizer.encode
    merge_mode is always 'concat' in the transformers implementation.
    """
    tok = _load_higgs_tokenizer(model_path or model_name)
    tok = _prepare_tok(tok)
    wrapper = HiggsQuantizerEncoderWrapper(tok.fc, tok.quantizer, merge_mode="concat")
    wrapper.eval()
    return wrapper


def get_higgs_quantizer_io_config(model_path=None):
    return {
        "input_names":  ["acoustic_features", "semantic_features"],
        "output_names": ["codes"],
        "dynamic_axes": {
            "acoustic_features": {0: "batch", 2: "frames"},
            "semantic_features": {0: "batch", 2: "frames"},
            "codes":             {1: "batch", 2: "frames"},
        },
    }


def get_higgs_quantizer_dummy_inputs(model=None):
    """Dummy feature tensors for 1 second of audio.

    DAC acoustic encoder: 24000 / 960 = 25 frames/sec  (DOWNSAMPLE_FACTOR=960)
    HIGGS_D_ACOUSTIC=256, HIGGS_D_SEMANTIC=768 confirmed from model inspection.
    tok.fc.in_features = 256 + 768 = 1024 (concat along channel dim).
    Both acoustic and semantic features are passed with the same T (acoustic rate)
    since the quantizer pipeline aligns them at inference time.
    """
    T = SR_24K // DOWNSAMPLE_FACTOR   # 25 frames/sec
    return {
        "acoustic_features": torch.randn(1, HIGGS_D_ACOUSTIC, T, dtype=torch.float32),
        "semantic_features": torch.randn(1, HIGGS_D_SEMANTIC, T, dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# 4. higgs_decoder  codes → waveform_24k
# ---------------------------------------------------------------------------

def get_higgs_decoder_model(model_path=None):
    """Return HiggsDecoderWrapper (nn.Module) for ONNX export via Olive.

    Transformers HiggsAudioV2TokenizerModel attribute mapping (confirmed from source):
      boson_multimodal    transformers
      tok.quantizer  →   tok.quantizer      (same — RVQ quantizer)
      tok.fc_post2   →   tok.fc2            (Linear: projects quantizer output to DAC input)
      tok.decoder    →   tok.acoustic_decoder (DAC decoder model)

    The decode pipeline: tok.quantizer.decode(codes) → tok.fc2(...) → tok.acoustic_decoder(...)

    TracerWarnings about Python branches are benign (branches resolve to constants
    from dummy inputs).
    """
    tok = _load_higgs_tokenizer(model_path or model_name)
    tok = _prepare_tok(tok)
    wrapper = HiggsDecoderWrapper(tok.quantizer, tok.fc2, tok.acoustic_decoder)
    wrapper.eval()
    return wrapper


def get_higgs_decoder_io_config(model_path=None):
    return {
        "input_names":  ["codes"],
        "output_names": ["waveform_24k"],
        "dynamic_axes": {
            "codes":        {1: "batch", 2: "frames"},
            "waveform_24k": {0: "batch", 2: "samples"},
        },
    }


def get_higgs_decoder_dummy_inputs(model=None):
    """Dummy codec codes for 1 second of audio (25 frames at 24kHz / hop=960)."""
    T = SR_24K // DOWNSAMPLE_FACTOR   # 25 frames/sec
    return {
        "codes": torch.randint(0, HIGGS_CB_SIZE, (HIGGS_N_CB, 1, T), dtype=torch.int64)
    }
