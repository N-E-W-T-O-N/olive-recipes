"""
user_script for bosonai/higgs-audio-v3-tts-4b ONNX export.

The checkpoint stores weights in a custom layout:
  body.layers.{i}.*                              → Qwen3 transformer layers
  body.norm.weight                               → final RMSNorm
  tied.embedding.text_embedding.weight           → token embeddings (tied lm_head)
  tied.embedding.modality_embeddings.0.model.*   → embedded Higgs audio tokenizer
                                                    (DAC acoustic_decoder, semantic_model, ...)

The LLM sub-part (Qwen3-4B-Base) can be extracted DIRECTLY from the safetensors
— no `boson-multimodal` / custom class required — by remapping the `body.*` and
`tied.embedding.text_embedding.weight` keys into a standard Qwen3ForCausalLM
directory that onnxruntime-genai ModelBuilder consumes.

The audio tokenizer sub-models (acoustic/semantic encoder, quantizer, decoder)
live under `modality_embeddings.0.model.*` and mirror the Higgs Audio tokenizer
already exported in the OmniVoice project (see OmniVoice/codes/model_wrappers.py
and OmniVoice/higgs/*.json). Those wrappers are reused once the full multimodal
class is loadable; see STATUS.md.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "bosonai/higgs-audio-v3-tts-4b"

# Qwen3-4B-Base dimensions (from config.json text_config)
LLM_NUM_LAYERS = 36

# Audio modality dims (from config.json audio_encoder_config)
AUDIO_NUM_CODEBOOKS = 8
AUDIO_VOCAB = 1026
AUDIO_HIDDEN = 2560
AUDIO_EMBED_KEY = "tied.embedding.modality_embeddings.0.embedding.weight"


# =============================================================================
# LLM sub-part: extract a standalone Qwen3ForCausalLM for ModelBuilder
# =============================================================================

def _resolve_model_dir(model_path: str) -> Path:
    """Return a local directory containing the checkpoint, downloading if needed."""
    p = Path(model_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path))


def _remap_key_to_qwen3(key: str):
    """Map a checkpoint key to its Qwen3ForCausalLM name, or None to skip.

    body.layers.{i}.<rest>  → model.layers.{i}.<rest>
    body.norm.weight        → model.norm.weight
    tied.embedding.text_embedding.weight → model.embed_tokens.weight
    everything else (audio tokenizer / modality embeddings) → skipped
    """
    if key.startswith("body.layers."):
        return "model." + key[len("body."):]
    if key == "body.norm.weight":
        return "model.norm.weight"
    if key == "tied.embedding.text_embedding.weight":
        return "model.embed_tokens.weight"
    return None


def extract_qwen3_standalone(model_path: str, output_dir: str) -> str:
    """Write a standalone Qwen3ForCausalLM HF directory for ModelBuilder.

    Reads the (possibly sharded) safetensors of higgs-audio-v3, keeps only the
    Qwen3 LLM tensors, remaps their names, and saves them plus a clean
    config.json (the checkpoint's text_config) and tokenizer files.

    Returns the path to the standalone directory.
    """
    from safetensors.torch import load_file, save_file

    src = _resolve_model_dir(model_path)
    out = Path(output_dir) / "qwen3_standalone"
    out.mkdir(parents=True, exist_ok=True)

    # 1. config.json — the checkpoint's text_config IS a Qwen3ForCausalLM config
    full_cfg = json.loads((src / "config.json").read_text())
    text_cfg = full_cfg["text_config"]
    text_cfg["architectures"] = ["Qwen3ForCausalLM"]
    text_cfg["model_type"] = "qwen3"
    (out / "config.json").write_text(json.dumps(text_cfg, indent=2))

    # 2. tokenizer + generation config (copy if present)
    import shutil
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "vocab.json", "merges.txt", "special_tokens_map.json",
               "chat_template.jinja"):
        s = src / fn
        if s.exists():
            shutil.copy2(s, out / fn)

    # 3. weights: locate all shards via the index (or single file)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        weight_map = json.loads(idx_path.read_text())["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    qwen3_state = {}
    for shard in shard_files:
        tensors = load_file(str(src / shard))
        for k, v in tensors.items():
            nk = _remap_key_to_qwen3(k)
            if nk is not None:
                qwen3_state[nk] = v
        del tensors

    n_layers = len({k.split(".")[2] for k in qwen3_state if k.startswith("model.layers.")})
    assert "model.embed_tokens.weight" in qwen3_state, "embed_tokens not found in checkpoint"
    assert n_layers == LLM_NUM_LAYERS, f"expected {LLM_NUM_LAYERS} layers, got {n_layers}"

    save_file(qwen3_state, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    print(f"  [LLM] standalone Qwen3 saved to {out}  ({len(qwen3_state)} tensors, {n_layers} layers)")
    return str(out)


# =============================================================================
# Text embedding sub-part: input_ids → inputs_embeds  (makes the pipeline
# self-contained — llm_decoder is built with exclude_embeds, so the embedding
# Gather must ship as its own ONNX instead of relying on qwen3_standalone at
# runtime). Weight = the same tied table used as the (excluded) lm_head.
# =============================================================================

TEXT_EMBED_KEY = "tied.embedding.text_embedding.weight"


def _load_text_embed_weight(model_path: str) -> torch.Tensor:
    from safetensors import safe_open
    src = _resolve_model_dir(model_path)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        shard = json.loads(idx_path.read_text())["weight_map"][TEXT_EMBED_KEY]
    else:
        shard = "model.safetensors"
    with safe_open(str(src / shard), framework="pt") as f:
        return f.get_tensor(TEXT_EMBED_KEY).float()


class TextEmbed(nn.Module):
    """input_ids [B, L] (int64) → inputs_embeds [B, L, D]  (plain Gather)."""
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding(weight.shape[0], weight.shape[1])
        self.emb.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)


def get_text_embed_model(model_path=None):
    return TextEmbed(_load_text_embed_weight(model_path or MODEL_NAME)).eval()


def get_text_embed_io_config(model=None):
    return {
        "input_names": ["input_ids"],
        "output_names": ["inputs_embeds"],
        "input_shapes": [[1, 16]],
        "input_types": ["int64"],
        "dynamic_axes": {"input_ids": {0: "batch", 1: "seq"},
                         "inputs_embeds": {0: "batch", 1: "seq"}},
    }


def get_text_embed_dummy_inputs(model=None):
    return {"input_ids": torch.randint(0, 1000, (1, 16), dtype=torch.int64)}


# =============================================================================
# Audio embed / head sub-parts (fused multi-codebook, tied)
# =============================================================================
#
# Reference: sglang_omni/models/higgs_tts/modeling.py
#   HiggsFusedMultiTextEmbedding: weight [N*V, D]; codes[...,N] + arange(N)*V,
#       F.embedding(...).sum(dim=-2)  → [...,D]
#   HiggsFusedMultiTextHead:       logits = F.linear(hidden, weight) → [L, N, V]
#   The head is TIED to the embedding weight. Delay pattern is applied in the
#   generation loop (sampler), not inside these modules → ONNX-clean.

def _load_audio_embed_weight(model_path: str) -> torch.Tensor:
    """Read the fused audio embedding weight [N*V, D] from the checkpoint."""
    from safetensors import safe_open
    src = _resolve_model_dir(model_path)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        shard = json.loads(idx_path.read_text())["weight_map"][AUDIO_EMBED_KEY]
    else:
        shard = "model.safetensors"
    with safe_open(str(src / shard), framework="pt") as f:
        return f.get_tensor(AUDIO_EMBED_KEY).float()


class AudioFusedEmbed(nn.Module):
    """codes [B, L, N] (int64) → fused hidden [B, L, D]."""
    def __init__(self, weight: torch.Tensor, num_codebooks=AUDIO_NUM_CODEBOOKS,
                 vocab_size=AUDIO_VOCAB):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.num_codebooks, device=codes.device,
                               dtype=codes.dtype) * self.vocab_size
        fused_ids = codes + offsets
        return F.embedding(fused_ids, self.weight).sum(dim=-2)


class AudioFusedHead(nn.Module):
    """hidden [B, L, D] → per-codebook logits [B, L, N, V] (tied weight)."""
    def __init__(self, weight: torch.Tensor, num_codebooks=AUDIO_NUM_CODEBOOKS,
                 vocab_size=AUDIO_VOCAB):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states, self.weight)                # [B, L, N*V]
        B, L = hidden_states.shape[0], hidden_states.shape[1]
        return logits.reshape(B, L, self.num_codebooks, self.vocab_size)


def get_audio_embed_model(model_path=None):
    return AudioFusedEmbed(_load_audio_embed_weight(model_path or MODEL_NAME)).eval()


def get_audio_embed_io_config(model=None):
    return {
        "input_names": ["codes"],
        "output_names": ["audio_embeds"],
        "input_shapes": [[1, 16, AUDIO_NUM_CODEBOOKS]],
        "input_types": ["int64"],
        "dynamic_axes": {"codes": {0: "batch", 1: "seq"},
                         "audio_embeds": {0: "batch", 1: "seq"}},
    }


def get_audio_embed_dummy_inputs(model=None):
    return {"codes": torch.randint(0, AUDIO_VOCAB, (1, 16, AUDIO_NUM_CODEBOOKS),
                                   dtype=torch.int64)}


def get_audio_heads_model(model_path=None):
    return AudioFusedHead(_load_audio_embed_weight(model_path or MODEL_NAME)).eval()


def get_audio_heads_io_config(model=None):
    return {
        "input_names": ["hidden_states"],
        "output_names": ["audio_logits"],
        "input_shapes": [[1, 16, AUDIO_HIDDEN]],
        "input_types": ["float32"],
        "dynamic_axes": {"hidden_states": {0: "batch", 1: "seq"},
                         "audio_logits": {0: "batch", 1: "seq"}},
    }


def get_audio_heads_dummy_inputs(model=None):
    return {"hidden_states": torch.randn(1, 16, AUDIO_HIDDEN, dtype=torch.float32)}


# =============================================================================
# Audio tokenizer (waveform codec) — decode: codes → waveform
# =============================================================================
#
# The codec weights are bundled in the TTS checkpoint under
# `tied.embedding.modality_embeddings.0.model.*` (the Higgs v2 tokenizer:
# DAC acoustic_decoder + semantic_model + RVQ quantizer). transformers 5.10.2
# provides HiggsAudioV2TokenizerModel; we build it from the v2-tokenizer config
# and load the bundled weights (prefix stripped). Reference:
# sglang_omni/models/higgs_tts/audio_codec.py.

CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."
CODEC_CONFIG_REPO = "bosonai/higgs-audio-v2-tokenizer"
CODEC_NUM_CODEBOOKS = 8
CODEC_SR = 24000
CODEC_FPS = 25   # 24000 / 960


def _load_higgs_codec(model_path: str):
    """Build HiggsAudioV2TokenizerModel and load the bundled codec weights."""
    import json as _json
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import HiggsAudioV2TokenizerModel, HiggsAudioV2TokenizerConfig

    cfg_dict = _json.loads(Path(hf_hub_download(CODEC_CONFIG_REPO, "config.json")).read_text())
    for k in ("architectures", "torch_dtype", "dtype", "transformers_version"):
        cfg_dict.pop(k, None)
    model = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**cfg_dict)).float().eval()

    src = _resolve_model_dir(model_path)
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
        shards = sorted({v for k, v in wmap.items() if k.startswith(CODEC_PREFIX)})
    else:
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(str(src / shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(CODEC_PREFIX):
                    state[k[len(CODEC_PREFIX):]] = f.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  [codec] loaded {len(state)} tensors (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


class CodecDecoderWrapper(nn.Module):
    """audio_codes [B, N, T] int64 → waveform [B, 1, L] float32."""
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        out = self.codec.decode(audio_codes, return_dict=False)
        wav = out[0] if isinstance(out, (tuple, list)) else out
        return wav


class CodecEncoderWrapper(nn.Module):
    """input_values [B, 1, T] float32 (24 kHz) → audio_codes [B, 8, frames] int64.

    Reference→codes for VOICE CLONING (the <|ref_audio|> prompt segment). Mirrors
    `HiggsAudioV2TokenizerModel.encode(...).audio_codes` (DAC-style encoder + RVQ).
    """
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.codec.encode(input_values)
        codes = out.audio_codes if hasattr(out, "audio_codes") else (
            out[0] if isinstance(out, (tuple, list)) else out)
        return codes


def get_audio_encoder_model(model_path=None):
    codec = _load_higgs_codec(model_path or MODEL_NAME)
    for p in codec.parameters():
        p.requires_grad_(False)
    return CodecEncoderWrapper(codec).eval()


def get_audio_encoder_io_config(model=None):
    return {
        "input_names": ["input_values"],
        "output_names": ["audio_codes"],
        "input_shapes": [[1, 1, CODEC_SR]],          # 1 s; time axis dynamic below
        "input_types": ["float32"],
        "dynamic_axes": {"input_values": {0: "batch", 2: "samples"},
                         "audio_codes": {0: "batch", 2: "frames"}},
    }


def get_audio_encoder_dummy_inputs(model=None):
    return {"input_values": torch.randn(1, 1, CODEC_SR, dtype=torch.float32)}


def get_audio_tokenizer_model(model_path=None):
    codec = _load_higgs_codec(model_path or MODEL_NAME)
    for p in codec.parameters():
        p.requires_grad_(False)
    return CodecDecoderWrapper(codec).eval()


def get_audio_tokenizer_io_config(model=None):
    return {
        "input_names": ["audio_codes"],
        "output_names": ["waveform"],
        "input_shapes": [[1, CODEC_NUM_CODEBOOKS, CODEC_FPS]],
        "input_types": ["int64"],
        "dynamic_axes": {"audio_codes": {0: "batch", 2: "frames"},
                         "waveform": {0: "batch", 2: "samples"}},
    }


def get_audio_tokenizer_dummy_inputs(model=None):
    return {"audio_codes": torch.randint(0, 1024, (1, CODEC_NUM_CODEBOOKS, CODEC_FPS),
                                         dtype=torch.int64)}
