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
# surya-ocr-2: 650M OCR/document-intelligence model (Qwen3.5 architecture)
#   vocab_size    : 65425   (custom OCR vocabulary, NOT Qwen's 152064)
#   image_token_id: 11      (NOT Qwen's 151655)
#   vision_start  : 9       (NOT Qwen's 151652)
#   vision_end    : 10      (NOT Qwen's 151653)
#   video_token_id: 12      (NOT Qwen's 151656)
#   vision hidden : 768     (depth=12, 12 heads)
#   out_hidden    : 1024    (matches text hidden_size)
#   deepstack     : []      (empty — no deepstack mergers)
# All values loaded from config at runtime via AutoConfig.
# ---------------------------------------------------------------------------
model_name = "datalab-to/surya-ocr-2"
config = AutoConfig.from_pretrained(model_name)


# ---------------------------------------------------------------------------
# Checkpoint loader with key remapping
# ---------------------------------------------------------------------------

def _load_base_model(model_path: str) -> Qwen3_5VLModel:
    """Load surya-ocr-2 checkpoint into Qwen3_5VLModel with key remapping.

    HF saves Qwen3_5ForConditionalGeneration with keys:
        model.visual.*          -> our visual.*
        model.language_model.*  -> our language_model.embed_tokens.*  (only key needed)
        lm_head.weight          -> dropped

    snapshot_download ensures all weight shards are present before globbing.
    """
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    # Download entire model snapshot (cached on subsequent runs)
    model_dir = snapshot_download(model_path)

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

    cfg = AutoConfig.from_pretrained(model_path)
    model = Qwen3_5VLModel(cfg)
    result = model.load_state_dict(state_dict, strict=False)

    # Only visual.* and language_model.embed_tokens.* are expected;
    # DeltaNet decoder layers are intentionally absent (strict=False)
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

    surya-ocr-2:
      vision hidden_size   = 768,  depth = 12
      out_hidden_size      = 1024  (matches text hidden_size)
      spatial_merge_size   = 2
      image_token_id       = 11   (custom OCR vocab, NOT standard Qwen 151655)

    Grid (1, 16, 16) -> 256 raw patches -> after merge_size=2: 8*8=64 logical patches.
    Two batches: 128 total logical patches.
    """
    out_hidden_size    = config.vision_config.out_hidden_size   # 1024
    spatial_merge_size = config.vision_config.spatial_merge_size  # 2

    grid_h, grid_w = 16, 16
    patches_per_image  = (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)  # 64
    batch_size         = 2
    num_logical_patches = batch_size * patches_per_image  # 128

    text_prefix_len = 5
    text_suffix_len = 5
    sequence_length = text_prefix_len + 1 + patches_per_image + 1 + text_suffix_len  # 76

    vision_start_token_id = config.vision_start_token_id   # 9
    vision_end_token_id   = config.vision_end_token_id     # 10
    image_token_id        = config.image_token_id          # 11

    input_ids = torch.randint(low=0, high=image_token_id,
                              size=(batch_size, sequence_length), dtype=torch.int64)
    image_features = torch.randn(num_logical_patches, out_hidden_size, dtype=torch.float32)

    img_start = text_prefix_len
    img_end   = img_start + patches_per_image

    for b in range(batch_size):
        input_ids[b][img_start - 1] = vision_start_token_id
        input_ids[b][img_start:img_end] = image_token_id
        input_ids[b][img_end] = vision_end_token_id

    return {"input_ids": input_ids, "image_features": image_features}


# ---------------------------------------------------------------------------
# Vision sub-model
# ---------------------------------------------------------------------------

def get_vision_model(model_path=None):
    model = _load_base_model(model_path or model_name)
    model = model.to(torch.float32)
    model.forward, model.get_image_features = (
        model.get_image_features,
        model.forward,
    )
    return model


def get_vision_io_config(model_path=None):
    return {
        "input_names":  ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values":   {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
        },
    }


def get_vision_dummy_inputs(model=None):
    """Dummy inputs for the vision encoder ONNX export.

    surya-ocr-2 vision encoder:
      patch_size=16, temporal_patch_size=2, in_channels=3
      channels_per_patch = 3 * 16 * 16 * 2 = 1536

    Grid (1, 16, 16): 256 patches × 2 images = 512 total.
    The fast_pos_embed_interpolate requires h >= 2 and w >= 2.
    """
    patch_size          = config.vision_config.patch_size           # 16
    temporal_patch_size = config.vision_config.temporal_patch_size  # 2
    in_channels         = config.vision_config.in_channels          # 3
    channels_per_patch  = in_channels * patch_size * patch_size * temporal_patch_size  # 1536

    grid_h, grid_w = 16, 16
    num_images     = 2
    num_patches    = num_images * grid_h * grid_w  # 512

    pixel_values = torch.randn((num_patches, channels_per_patch), dtype=torch.float32)
    pixel_values = pixel_values * (0.95 - (-1.0)) + (-1.0)
    grid_thw = torch.tensor([[1, grid_h, grid_w], [1, grid_h, grid_w]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
