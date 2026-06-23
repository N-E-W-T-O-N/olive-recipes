import os
import sys
import torch

from transformers import Qwen3VLConfig

# Add this script's directory to sys.path to import codes module
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Import custom model from codes directory
from codes.modeling_qwen3_vl import Qwen3VLModel

# Resolve the model source: a local snapshot dir (set by optimize.py via
# CHANDRA_MODEL_DIR) takes precedence over the HF repo id. from_pretrained accepts
# either a local directory or a repo id transparently.
model_name = os.environ.get("CHANDRA_MODEL_DIR") or "datalab-to/chandra"
config = Qwen3VLConfig.from_pretrained(model_name)


def _load_base_model(model_path):
    """Load weights directly from safetensors, stripping 'model.' prefix,
    into our custom Qwen3VLModel without loading the full HF model.

    `model_path` may be a local directory (a snapshot downloaded by optimize.py)
    or an HF repo id. Local dirs are used as-is; repo ids are fetched via the hub."""
    from safetensors.torch import load_file
    import glob

    src = model_path or model_name
    if os.path.isdir(src):
        model_dir = src
    else:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(src, 'config.json')
        model_dir = os.path.dirname(config_path)
    st_files = sorted(glob.glob(os.path.join(model_dir, '*.safetensors')))

    # Load and strip 'model.' prefix, keeping native bfloat16 precision
    state_dict = {}
    for sf in st_files:
        tensors = load_file(sf)
        for k, v in tensors.items():
            if k.startswith('model.'):
                state_dict[k[6:]] = v

    # Create custom model and load weights in bfloat16 (native dtype)
    custom_model = Qwen3VLModel(config)
    result = custom_model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        # Categorise missing keys so we can distinguish harmless vs critical
        # Non-persistent buffers (inv_freq) are computed correctly in __init__
        # and are never saved in HF checkpoints — always expected to be missing.
        inv_freq_keys   = [k for k in result.missing_keys if "inv_freq" in k]
        missing_embed   = [k for k in result.missing_keys if "embed_tokens" in k]
        missing_decoder = [k for k in result.missing_keys
                           if k.startswith("language_model.layers.")]
        missing_other   = [k for k in result.missing_keys
                           if k not in inv_freq_keys + missing_embed + missing_decoder]

        print(f"[LOAD] {len(result.missing_keys)} missing keys: "
              f"inv_freq(expected)={len(inv_freq_keys)}, embed={len(missing_embed)}, "
              f"decoder={len(missing_decoder)}, other={len(missing_other)}")

        if inv_freq_keys:
            # inv_freq is non-persistent in HF checkpoints; our model registers
            # it as persistent=True so __init__ computes the correct value.
            print(f"  [OK] inv_freq buffers absent from checkpoint — "
                  f"__init__ values are correct: {inv_freq_keys}")
        if missing_embed:
            print(f"  [CRITICAL] Missing embed_tokens — embedding model will have wrong weights!")
            print(f"    Keys: {missing_embed}")
        if missing_decoder:
            # Text decoder layers are NOT used in vision/embedding ONNX export.
            # ModelBuilder handles the full text decoder via text.json directly from HF.
            print(f"  [OK] {len(missing_decoder)} text decoder layer keys missing "
                  f"(not needed for vision/embedding ONNX export)")
        if missing_other:
            print(f"  [WARN] Unexpected missing (first 5): {missing_other[:5]}")

        if result.unexpected_keys:
            print(f"[LOAD] {len(result.unexpected_keys)} unexpected keys "
                  f"(first 3): {result.unexpected_keys[:3]}")

    custom_model = custom_model.to(torch.bfloat16)
    custom_model.eval()

    del state_dict
    return custom_model


### Embedding
# Dynamo export

def get_embedding_model(model_path=None):
    model = _load_base_model(model_path)
    # Export in fp32 for Olive fp16 pass compatibility (same approach as vision)
    model = model.to(torch.float32)

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
    # Chandra (Qwen3-VL-8B-based): out_hidden_size=4096, patch_size=16
    # assume 2 batches, each with 1 image input
    # With patch_size=16, spatial_merge_size=2:
    #   For a 540x360 image: grid = (1, 22, 34) -> merged = (1, 11, 17) -> 187 logical patches per image
    #   raw patches = 22*34 = 748 per frame
    batch_size, sequence_length, patches_per_image, out_hidden_size = (
        2,
        216,     # approximate sequence length with image tokens
        187,     # logical patches per image after merge (11*17)
        4096,    # Chandra out_hidden_size (same as Qwen3-VL-8B)
    )
    num_logical_patches = batch_size * patches_per_image

    # Chandra special token IDs (same as Qwen3-VL-8B base)
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
            dtype=torch.float32,  # fp32 to match embedding model export dtype
        ),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image

    # Fill in with image token index
    inputs["input_ids"][0][2] = vision_start_token_id  # <|vision_start|>
    inputs["input_ids"][0][
        img_start_index:img_end_index
    ] = image_token_id  # <|image_pad|>
    inputs["input_ids"][0][img_end_index] = vision_end_token_id  # <|vision_end|>

    inputs["input_ids"][1][2] = vision_start_token_id  # <|vision_start|>
    inputs["input_ids"][1][
        img_start_index:img_end_index
    ] = image_token_id  # <|image_pad|>
    inputs["input_ids"][1][img_end_index] = vision_end_token_id  # <|vision_end|>

    return {
        "input_ids": inputs["input_ids"],  # input_ids: torch.LongTensor
        "image_features": inputs["image_features"],  # image_features: Optional[torch.FloatTensor] = None,
    }


### Vision
def get_vision_model(model_path=None):
    # Export in fp32 for maximum compatibility; Olive fp16 pass converts weights.
    model = _load_base_model(model_path)
    model = model.to(torch.float32)
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

    Two images with the same 22x34 grid (748 patches each, 1496 total)
    to exercise the dynamic num_images dimension during torch.export tracing.
    Chandra: patch_size=16, temporal_patch_size=2 -> 1536 channels/patch.
    (in_channels=3 * patch_size=16 * patch_size=16 * temporal_patch_size=2 = 1536)
    """
    pixel_values = torch.randn((2 * 748, 1536), dtype=torch.float32)
    pixel_values = pixel_values * (0.95 - (-1)) + (-1)
    grid_thw = torch.tensor([[1, 22, 34], [1, 22, 34]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
