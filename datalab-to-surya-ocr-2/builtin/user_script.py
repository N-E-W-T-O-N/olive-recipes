import os
import sys
import torch

from transformers import Qwen3_5Config

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from codes.modeling_qwen3_5 import Qwen3_5Model

# Prefer a local snapshot in ./model (what the recipe downloads to); fall back to the HF id.
_HF_ID = "datalab-to/surya-ocr-2"
_LOCAL_MODEL = os.path.join(_script_dir, "model")
model_name = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else _HF_ID
config = Qwen3_5Config.from_pretrained(model_name)


def _resolve_model_dir(model_path):
    """Return a local directory holding config.json + *.safetensors.

    Accepts either a local dir (used directly) or an HF repo id (downloaded). Olive passes
    the JSON `model_path`; None falls back to the module-level `model_name`.
    """
    from huggingface_hub import snapshot_download

    cand = model_path or model_name
    if cand and os.path.isdir(cand):
        return cand
    return snapshot_download(cand)


def _load_base_model(model_path):
    """Load weights from safetensors into custom Qwen3_5Model."""
    from safetensors.torch import load_file
    import glob

    model_dir = _resolve_model_dir(model_path)
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))

    state_dict = {}
    for sf in st_files:
        tensors = load_file(sf)
        for k, v in tensors.items():
            if k.startswith("model."):
                stripped = k[6:]
                state_dict[stripped] = v
                # Map language_model.embed_tokens -> embed_tokens for our flat model
                if stripped.startswith("language_model.embed_tokens."):
                    state_dict[stripped[len("language_model."):]] = v

    custom_model = Qwen3_5Model(config)
    result = custom_model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        print(f"Warning: {len(result.missing_keys)} missing keys")
    custom_model = custom_model.to(torch.bfloat16)
    custom_model.eval()

    del state_dict
    return custom_model


# ── Embedding ────────────────────────────────────────────────────────────

def get_embedding_model(model_path=None):
    model = _load_base_model(model_path)
    model = model.to(torch.float32)
    model.get_fused_input_embeddings, model.forward = (
        model.forward,
        model.get_fused_input_embeddings,
    )
    return model


def get_embedding_io_config(model_path=None):
    return {
        "input_names": ["input_ids", "image_features"],
        "output_names": ["inputs_embeds"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "image_features": {0: "num_logical_patches"},
            "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
        },
    }


def get_embedding_dummy_inputs(model=None):
    # surya-ocr-2: out_hidden_size=1024 (== text hidden_size), patch_size=16.
    # Read from config so this stays correct across checkpoints/scales.
    out_hidden_size = config.vision_config.out_hidden_size
    batch_size, sequence_length, patches_per_image = 2, 216, 187
    num_logical_patches = batch_size * patches_per_image

    vision_start_token_id = config.vision_start_token_id
    vision_end_token_id = config.vision_end_token_id
    image_token_id = config.image_token_id

    inputs = {
        "input_ids": torch.randint(0, image_token_id, (batch_size, sequence_length), dtype=torch.int64),
        "image_features": torch.randn(num_logical_patches, out_hidden_size, dtype=torch.float32),
    }

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image

    for b in range(batch_size):
        inputs["input_ids"][b][2] = vision_start_token_id
        inputs["input_ids"][b][img_start_index:img_end_index] = image_token_id
        inputs["input_ids"][b][img_end_index] = vision_end_token_id

    return inputs


# ── Vision ───────────────────────────────────────────────────────────────

def get_vision_model(model_path=None):
    model = _load_base_model(model_path)
    model = model.to(torch.float32)
    model.forward, model.get_image_features = model.get_image_features, model.forward
    return model


def get_vision_io_config(model_path=None):
    return {
        "input_names": ["pixel_values", "image_grid_thw"],
        "output_names": ["image_features"],
        "dynamic_shapes": {
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": None,
        },
    }


def get_vision_dummy_inputs(model=None):
    # patch_size=16, temporal_patch_size=2, in_channels=3
    # patch dim: 3 * 2 * 16 * 16 = 1536
    # For 544x352 image: grid=(1, 22, 34), 748 raw patches
    patches = 22 * 34
    pixel_values = torch.randn((patches, 1536), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 22, 34]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}
