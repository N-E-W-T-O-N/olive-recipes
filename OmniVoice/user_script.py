"""
Olive user_script for Prince-1/OmniVoice ONNX export.

Exposes three sub-model loader functions consumed by the Olive JSON configs:
  - get_audio_embeddings_model   → audio_embeddings_encoder.onnx
  - get_audio_heads_model        → audio_heads_decoder.onnx
  - (llm_decoder is handled by ModelBuilder in llm_decoder.json)

The Higgs Audio V2 Tokenizer (acoustic/semantic encoder, quantizer, decoder)
requires the external `boson-multimodal` package and is NOT exported here.
Run the companion script:
  python convert_omnivoice_to_onnx.py --only higgs
from the downloaded OmniVoice model directory to export those parts.

OmniVoice architecture constants (from config.json):
  HIDDEN_SIZE   = 1024
  NUM_CODEBOOKS = 8
  AUDIO_VOCAB   = 1025  (1024 real codes + 1 mask at index 1024)
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
