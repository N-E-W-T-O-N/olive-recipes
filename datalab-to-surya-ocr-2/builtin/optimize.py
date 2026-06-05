"""End-to-end optimization pipeline for datalab-to/surya-ocr-2 ONNX models.

surya-ocr-2 is a 650M-parameter OCR / document-intelligence model built on the
Qwen3.5 architecture (Qwen3_5ForConditionalGeneration).  Key differences from
standard Qwen3.5-VL models:

  - Custom OCR vocabulary: vocab_size=65425 (not Qwen's 152064)
  - Custom token IDs:  image=11, vision_start=9, vision_end=10, video=12
  - Smaller vision encoder: hidden_size=768, depth=12, out_hidden_size=1024
  - No deepstack: deepstack_visual_indexes=[] (empty)
  - Text decoder: Hybrid Gated DeltaNet (same 3:1 DeltaNet/attention ratio)

Preprocessing (confirmed from preprocessor_config.json):
  - image_mean / image_std : [0.5, 0.5, 0.5]
  - rescale_factor         : 1/255
  - patch_size             : 16
  - spatial_merge_size     : 2
  - temporal_patch_size    : 2
  - min_pixels             : 65536  (size.shortest_edge)
  - max_pixels             : 16777216 (size.longest_edge)

Usage:
    python optimize.py --config-dir cpu_and_mobile --device cpu
    python optimize.py --config-dir cuda --device gpu
    python optimize.py --config-dir cpu_and_mobile --skip-export
"""
import argparse
import json
import logging
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODELS_DIR = "models"

# Preprocessing values — confirmed from surya-ocr-2 preprocessor_config.json
_PREPROCESSOR = {
    "patch_size": 16,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 0.00392156862745098,   # 1/255
    "min_pixels": 65536,       # size.shortest_edge = 256*256
    "max_pixels": 16777216,    # size.longest_edge  = 4096*4096
}

# Token IDs — confirmed from surya-ocr-2 config.json (custom OCR vocabulary)
# NOTE: These differ entirely from standard Qwen VL models (151655, 151652 etc.)
_TOKEN_IDS = {
    "image_token_id":        11,
    "video_token_id":        12,
    "vision_start_token_id":  9,
}


# =============================================================================
# 1. Olive Export + Optimization + Quantization
# =============================================================================

def export_models(config_dir: str):
    from olive import run

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")
    for config in ("embedding.json", "text.json", "vision.json"):
        print(f"  Running {config}...")
        run(str(config_path / config))
    print()


# =============================================================================
# 2. GenAI Runtime Config Generation
# =============================================================================

def update_genai_config(output_dir: str = MODELS_DIR, device: str = "cpu"):
    """Patch genai_config.json with embedding/vision sections and processor_config."""
    config_path = Path(output_dir) / "genai_config.json"
    with open(config_path) as f:
        config = json.load(f)

    if device == "gpu":
        provider_options = [
            {"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}
        ]
    else:
        provider_options = []

    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}

    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs":  {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }

    config["model"]["vision"] = {
        "filename": "vision.onnx",
        "config_filename": "processor_config.json",
        "spatial_merge_size": _PREPROCESSOR["merge_size"],
        "tokens_per_second": 2.0,
        "patch_size": _PREPROCESSOR["patch_size"],
        "window_size": 64,
        "inputs":  {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": session_options,
    }

    # Custom OCR token IDs — NOT the standard Qwen 151655/151652 values
    config["model"]["image_token_id"]        = _TOKEN_IDS["image_token_id"]
    config["model"]["video_token_id"]        = _TOKEN_IDS["video_token_id"]
    config["model"]["vision_start_token_id"] = _TOKEN_IDS["vision_start_token_id"]

    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    # Create processor_config.json
    # All values confirmed from surya-ocr-2 preprocessor_config.json.
    processor_config = {
        "processor": {
            "name": "surya_ocr2_image_processor",
            "transforms": [
                {"operation": {"name": "decode_image",   "type": "DecodeImage",  "attrs": {"color_space": "RGB"}}},
                {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
                {"operation": {"name": "resize", "type": "Resize", "attrs": {
                    "width": 540, "height": 360,
                    "smart_resize": 1,
                    "min_pixels": _PREPROCESSOR["min_pixels"],
                    "max_pixels": _PREPROCESSOR["max_pixels"],
                    "patch_size": _PREPROCESSOR["patch_size"],
                    "merge_size": _PREPROCESSOR["merge_size"],
                }}},
                {"operation": {"name": "rescale", "type": "Rescale", "attrs": {
                    "rescale_factor": _PREPROCESSOR["rescale_factor"],
                }}},
                {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
                    "mean": _PREPROCESSOR["image_mean"],
                    "std":  _PREPROCESSOR["image_std"],
                    "qwen3_vl": 1,
                }}},
                {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
                    "patch_size":          _PREPROCESSOR["patch_size"],
                    "temporal_patch_size": _PREPROCESSOR["temporal_patch_size"],
                    "merge_size":          _PREPROCESSOR["merge_size"],
                }}},
            ],
        }
    }

    processor_path = Path(output_dir) / "processor_config.json"
    with open(processor_path, "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {processor_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize datalab-to/surya-ocr-2 ONNX models")
    parser.add_argument("--device", choices=["gpu", "cpu"], default="cpu")
    parser.add_argument("--config-dir", default="cpu_and_mobile")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args()

    models_dir = args.models_dir or str(Path(args.config_dir) / MODELS_DIR)

    if not args.skip_export:
        export_models(args.config_dir)

    print("=== Generating GenAI runtime configs ===")
    update_genai_config(output_dir=models_dir, device=args.device)
    print()
    print("Done.")


if __name__ == "__main__":
    main()
