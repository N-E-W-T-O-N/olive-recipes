import os
import sys
import glob
import torch

# Fix Windows cp1252 encoding crash when PyTorch prints emoji
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add this script's directory to sys.path to import codes module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from transformers import AutoConfig
from codes.modeling_qwen3_5_vl import Qwen3_5VLModel

# ---------------------------------------------------------------------------
# Model identity
# NOTE: unsloth/Qwen3.5-0.8B is a gated model on HuggingFace.
#       Run `huggingface-cli login` and accept the license before use.
# ---------------------------------------------------------------------------
model_name = "unsloth/Qwen3.5-0.8B"
config = AutoConfig.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Checkpoint loader with key remapping
# ---------------------------------------------------------------------------

def _load_base_model(model_path: str) -> Qwen3_5VLModel:
    """Load Qwen3.5-VL checkpoint into Qwen3_5VLModel with key remapping.

    HF saves Qwen3_5ForConditionalGeneration with keys:
        model.visual.*          -> our visual.*
        model.language_model.*  -> our language_model.embed_tokens.*  (only key needed)
        lm_head.weight          -> dropped (not part of Qwen3_5VLModel)

    We strip the leading 'model.' prefix and load with strict=False so that
    the Gated DeltaNet text-decoder weights (which we don't define) are silently
    ignored — only visual.* and language_model.embed_tokens.* are populated.
    """
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    # snapshot_download ensures ALL model files (weights + config) are present
    # in the local cache before we try to glob for .safetensors shards.
    # This fixes a race condition where hf_hub_download(config.json) would
    # only download the config, leaving the weight shards absent on the first
    # run (ModelBuilder in text.json triggers the full download later, but
    # embedding.json runs first and would fail without snapshot_download here).
    model_dir = snapshot_download(model_path)

    # Collect all safetensors shards (handles both single-file and sharded formats)
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}. "
            "Ensure the model is accessible and not corrupted."
        )

    # Load + remap: strip 'model.' prefix, drop lm_head
    state_dict = {}
    for sf in st_files:
        for k, v in load_file(sf).items():
            if k.startswith("model."):
                state_dict[k[len("model."):]] = v
            elif k.startswith("lm_head."):
                pass  # not part of Qwen3_5VLModel
            else:
                state_dict[k] = v

    # Build model from config and load weights
    cfg = AutoConfig.from_pretrained(model_path)
    model = Qwen3_5VLModel(cfg)
    result = model.load_state_dict(state_dict, strict=False)

    # Only visual.* and language_model.embed_tokens.* should be expected;
    # report genuine misses (not DeltaNet decoder layers which are intentionally absent)
    genuine_missing = [
        k for k in result.missing_keys
        if k.startswith("visual.") or "embed_tokens" in k
    ]
    if genuine_missing:
        print(f"[WARN] {len(genuine_missing)} missing visual/embedding keys: {genuine_missing[:5]}")

    model = model.to(torch.bfloat16)
    model.eval()
    del state_dict
    return model


# ---------------------------------------------------------------------------
# Embedding sub-model
# ---------------------------------------------------------------------------

def get_embedding_model(model_path=None):
    """Load and prepare the embedding sub-model for ONNX export.

    Swaps forward <-> get_fused_input_embeddings so Olive exports the embedding
    fusion logic (embed_tokens + image feature scatter) as the primary graph.
    """
    model = _load_base_model(model_path or model_name)
    model = model.to(torch.float32)
    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model


def get_embedding_io_config(model_path=None):
    dynamic_axes = {
        "input_ids":      {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
        "inputs_embeds":  {0: "batch_size", 1: "sequence_length"},
    }
    return {
        "input_names":  ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": dynamic_axes,
    }


def get_embedding_dummy_inputs(model=None):
    """Dummy inputs for the embedding sub-model ONNX export.

    Qwen3.5-VL-7B: out_hidden_size = config.vision_config.out_hidden_size (typically 3584)
    Grid (1, 16, 16) -> 256 raw patches -> after spatial_merge_size=2: 8*8=64 logical patches.
    Two batches, each with one image: 128 total logical patches.
    sequence_length covers: text prefix + vision_start + image_pads + vision_end + text suffix.
    """
    out_hidden_size = config.vision_config.out_hidden_size  # loaded from HF config
    spatial_merge_size = config.vision_config.spatial_merge_size  # 2

    grid_h, grid_w = 16, 16
    patches_per_image = (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)  # 64
    batch_size = 2
    num_logical_patches = batch_size * patches_per_image  # 128

    # Build a realistic sequence: [text..., vision_start, image_pads..., vision_end, text...]
    text_prefix_len = 5
    text_suffix_len = 5
    sequence_length = text_prefix_len + 1 + patches_per_image + 1 + text_suffix_len  # 76

    vision_start_token_id = config.vision_start_token_id  # 151652
    vision_end_token_id   = config.vision_end_token_id    # 151653
    image_token_id        = config.image_token_id         # 151655

    input_ids = torch.randint(low=0, high=image_token_id,
                              size=(batch_size, sequence_length), dtype=torch.int64)
    image_features = torch.randn(num_logical_patches, out_hidden_size, dtype=torch.float32)

    img_start = text_prefix_len
    img_end   = img_start + patches_per_image  # exclusive

    for b in range(batch_size):
        input_ids[b][img_start - 1] = vision_start_token_id
        input_ids[b][img_start:img_end] = image_token_id
        input_ids[b][img_end] = vision_end_token_id

    return {"input_ids": input_ids, "image_features": image_features}


# ---------------------------------------------------------------------------
# Vision sub-model
# ---------------------------------------------------------------------------

def get_vision_model(model_path=None):
    """Load and prepare the vision encoder sub-model for ONNX export.

    Swaps forward <-> get_image_features so Olive exports the vision encoder
    (patch_embed → absolute pos embed → rotary → ViT blocks → merger) as the
    primary graph.
    """
    model = _load_base_model(model_path or model_name)
    model = model.to(torch.float32)
    model.forward, model.get_image_features = (
        model.get_image_features,
        model.forward,
    )
    return model


def get_vision_io_config(model_path=None):
    """Vision encoder IO config with dynamic shapes.

    pixel_values dim-0 (num_patches) and image_grid_thw dim-0 (num_images)
    are both symbolic so the ONNX model handles any image count / resolution.
    Requires torch >= 2.10 for reliable dynamo export with dynamic_shapes.
    """
    return {
        "input_names":  ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values":    {0: "num_patches"},
            "image_grid_thw":  {0: "num_images"},
        },
    }


def get_vision_dummy_inputs(model=None):
    """Dummy inputs for the vision encoder ONNX export.

    Two images with grid (1, 16, 16):
        raw patches per image = 16*16 = 256, total = 512.
    Channels per patch = in_channels * patch_size * patch_size * temporal_patch_size
                       = 3 * 16 * 16 * 2 = 1536.

    NOTE: The fast_pos_embed_interpolate method requires h >= 2 and w >= 2 (checked
    with torch._check).  Grid (1, 16, 16) satisfies this constraint.
    """
    patch_size = config.vision_config.patch_size           # 16
    temporal_patch_size = config.vision_config.temporal_patch_size  # 2
    in_channels = config.vision_config.in_channels         # 3
    channels_per_patch = in_channels * patch_size * patch_size * temporal_patch_size  # 1536

    grid_h, grid_w = 16, 16
    num_patches_per_image = grid_h * grid_w  # 256
    num_images = 2

    pixel_values = torch.randn((num_images * num_patches_per_image, channels_per_patch), dtype=torch.float32)
    pixel_values = pixel_values * (0.95 - (-1.0)) + (-1.0)  # rescale to approx [-1, 1]
    grid_thw = torch.tensor([[1, grid_h, grid_w], [1, grid_h, grid_w]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
