import os
import sys

# Fix Windows cp1252 encoding crash when PyTorch prints emoji in error messages
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

from transformers import Qwen2_5_VLConfig

# Add current directory to sys.path to import codes module
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Import custom model from codes directory
from codes.modeling_qwen2_5_vl import Qwen2_5_VLModel

model_name = "microsoft/Fara-7B"
config = Qwen2_5_VLConfig.from_pretrained(model_name)


# =============================================================================
# Key remapping: Fara-7B checkpoint uses a flat naming convention
# =============================================================================
#
# Fara-7B was saved with its text decoder at the top level:
#   model.layers.*              (in checkpoint)
#   model.embed_tokens.weight   (in checkpoint)
#   model.norm.weight           (in checkpoint)
#   lm_head.weight              (in checkpoint, not part of Qwen2_5_VLModel)
#
# Qwen2_5_VLModel expects the text decoder nested under language_model:
#   language_model.layers.*
#   language_model.embed_tokens.weight
#   language_model.norm.weight
#
# Visual weights (visual.*) load correctly without remapping.

def _remap_fara7b_state_dict(state_dict: dict) -> dict:
    """Remap Fara-7B flat checkpoint keys to Qwen2_5_VLModel nested structure.

    Transforms:
        model.*       -> language_model.*
        lm_head.*     -> (dropped — not part of Qwen2_5_VLModel)
        visual.*      -> visual.*  (unchanged, already correct)
        everything else unchanged
    """
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            remapped["language_model." + k[len("model."):]] = v
        elif k.startswith("lm_head."):
            pass  # lm_head is part of Qwen2_5_VLForConditionalGeneration, not Qwen2_5_VLModel
        else:
            remapped[k] = v  # visual.* and any other top-level keys stay as-is
    return remapped


def _load_state_dict_from_checkpoint(local_dir: "os.PathLike") -> dict:
    """Load all weight shards from a local model directory into a single dict."""
    import glob
    from pathlib import Path

    local_dir = Path(local_dir)

    # Prefer safetensors (faster, safer)
    sf_files = sorted(local_dir.glob("*.safetensors"))
    if sf_files:
        from safetensors.torch import load_file
        state_dict = {}
        for f in sf_files:
            state_dict.update(load_file(str(f), device="cpu"))
        return state_dict

    # Fall back to pytorch_model*.bin shards
    bin_files = sorted(local_dir.glob("pytorch_model*.bin"))
    if bin_files:
        state_dict = {}
        for f in bin_files:
            state_dict.update(torch.load(str(f), map_location="cpu", weights_only=True))
        return state_dict

    raise FileNotFoundError(
        f"No .safetensors or pytorch_model*.bin files found in {local_dir}"
    )


def _load_fara7b_model(model_path: str, attn_implementation: str = "sdpa",
                       torch_dtype=torch.float32) -> Qwen2_5_VLModel:
    """Build Qwen2_5_VLModel and load Fara-7B weights with key remapping.

    Handles both a local directory and a HuggingFace Hub model ID.
    After loading, only non-persistent buffers (inv_freq) will be missing —
    those are handled separately by _reinit_inv_freq.
    """
    from pathlib import Path

    # Resolve HF Hub ID to a local snapshot directory if needed
    local_path = Path(model_path)
    if not local_path.is_dir():
        from huggingface_hub import snapshot_download
        local_path = Path(snapshot_download(model_path))

    # Build the model structure on the meta device (no memory allocated for weights)
    # then materialize empty tensors on CPU before loading the real weights.
    cfg = Qwen2_5_VLConfig.from_pretrained(str(local_path))
    cfg._attn_implementation = attn_implementation
    with torch.device("meta"):
        model = Qwen2_5_VLModel(cfg)
    model = model.to_empty(device="cpu").to(torch_dtype)

    # Load, remap, and apply checkpoint weights
    raw_sd = _load_state_dict_from_checkpoint(local_path)
    remapped_sd = _remap_fara7b_state_dict(raw_sd)
    missing, unexpected = model.load_state_dict(remapped_sd, strict=False)

    # Warn only on truly unexpected missing keys (inv_freq buffers are expected to be absent)
    real_missing = [k for k in missing if "inv_freq" not in k]
    if real_missing:
        print(f"[WARN] Fara-7B load: {len(real_missing)} missing keys "
              f"(first 5): {real_missing[:5]}")
    if unexpected:
        print(f"[WARN] Fara-7B load: {len(unexpected)} unexpected keys "
              f"(first 5): {unexpected[:5]}")

    return model


# =============================================================================
# Embedding model
# =============================================================================

def get_embedding_model(model_path=None):
    model = _load_fara7b_model(
        model_path or model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model


def get_embedding_io_config(model_path=None):
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
        "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
    }
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": dynamic_axes,
    }


def get_embedding_dummy_inputs(model=None):
    # assume 2 batches, each with 1 image input (3577 logical patches)
    # out_hidden_size: 3584 for Fara-7B
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        2,
        3606,
        3577,
        3584,  # Fara-7B hidden_size
    )
    num_logical_patches = batch_size * patches_per_image

    # Fara-7B special token IDs (same as Qwen2.5-VL base)
    vision_start_token_id = config.vision_start_token_id  # 151652
    vision_end_token_id = config.vision_end_token_id      # 151653
    image_token_id = config.image_token_id                # 151655

    inputs = {
        "input_ids": torch.randint(
            low=0,
            high=image_token_id,
            size=(batch_size, sequence_length),
            dtype=torch.int64,
        ),
        "image_features": torch.randn(
            num_logical_patches,
            out_hidden_size,
            dtype=torch.float32,
        ),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image  # 3 + 3577 = 3580

    # Fill in with image token indices
    inputs["input_ids"][0][2] = vision_start_token_id          # <|vision_start|>
    inputs["input_ids"][0][img_start_index:img_end_index] = image_token_id  # <|image_pad|>
    inputs["input_ids"][0][img_end_index] = vision_end_token_id             # <|vision_end|>

    inputs["input_ids"][1][2] = vision_start_token_id
    inputs["input_ids"][1][img_start_index:img_end_index] = image_token_id
    inputs["input_ids"][1][img_end_index] = vision_end_token_id

    return {
        "input_ids": inputs["input_ids"],
        "image_features": inputs["image_features"],
    }


# =============================================================================
# Vision model
# =============================================================================

def _reinit_inv_freq(model):
    """Recompute inv_freq buffers that are missing from the HF checkpoint.

    The upstream Qwen code registers inv_freq with persistent=False, so
    the buffer is never saved in the checkpoint.  Our local modeling code
    uses persistent=True so that torch.export captures the buffer, but
    the manual load leaves it uninitialized.  Re-derive the correct values
    from the same formula used in __init__.
    """
    rope = model.visual.rotary_pos_emb
    dim = rope.inv_freq.shape[0] * 2          # original dim passed to __init__
    theta = 10000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    rope.inv_freq.data.copy_(inv_freq)


def get_vision_model(model_path=None):
    model = _load_fara7b_model(
        model_path or model_name,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )
    _reinit_inv_freq(model)
    model.forward, model.get_image_features = model.get_image_features, model.forward
    return model


def get_vision_io_config(model_path=None):
    """Vision model IO config with dynamic shapes.

    Both pixel_values and image_grid_thw have symbolic dim-0 so the model
    accepts any number of patches (any image resolution) and any number of
    images in a single call.  The RenameInputDims graph surgery in the Olive
    config labels dim-0 of image_grid_thw as 'num_images' in the final ONNX.

    Requires torch >= 2.10 for reliable dynamo export with dynamic_shapes.
    """
    return {
        "input_names": ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
        },
    }


def get_vision_dummy_inputs(model=None):
    """Dummy inputs for vision model export.

    Two images with the same 14x14 grid (196 patches each, 392 total)
    to exercise the dynamic num_images dimension during torch.export tracing.
    Fara-7B: patch_size=14, temporal_patch_size=2 -> 1176 channels/patch.
    """
    pixel_values = torch.randn((2 * 196, 1176), dtype=torch.float32)
    pixel_values = pixel_values * (0.95 - (-1)) + (-1)
    grid_thw = torch.tensor([[1, 14, 14], [1, 14, 14]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
