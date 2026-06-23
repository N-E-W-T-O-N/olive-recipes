"""End-to-end optimization pipeline for datalab-to/chandra (Qwen3-VL-8B) ONNX models.

Builds three sub-models per target — text decoder, text embedding, vision encoder —
parameterized by device × precision:

    targets: cpu_fp32  cpu_fp16  cpu_int4  cuda_fp32  cuda_fp16

- **text decoder**: built with onnxruntime-genai **ModelBuilder (create_model) directly**
  (NOT via an Olive pass) so cpu + fp32 is allowed — Olive's ModelBuilder pass restricts
  the cpu/precision combos. precision ∈ {int4, fp16, fp32}, EP cpu|cuda.
- **vision / embedding**: exported + graph-optimized via Olive (configs generated in-Python),
  with the final precision pass swapped per target:
    int4 → OnnxBlockWiseRtnQuantization   fp16 → OnnxFloatToFloat16   fp32 → (no quant pass)

Output: <device>_<precision>/models/  (text.onnx, embedding.onnx, vision.onnx,
genai_config.json, processor_config.json, tokenizer files).

Usage:
    python optimize.py --device cpu  --precision int4
    python optimize.py --device cpu  --precision fp32
    python optimize.py --device cuda --precision fp16
    python optimize.py --device cuda --precision fp32 --components vision   # subset
    python optimize.py --device cpu  --precision int4 --skip-export         # regen configs only
"""
import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODEL_ID = "datalab-to/chandra"
HERE = Path(__file__).parent

DEVICE = {  # device → (olive accelerator device, EP, ModelBuilder EP)
    "cpu": ("cpu", "CPUExecutionProvider", "cpu"),
    "cuda": ("gpu", "CUDAExecutionProvider", "cuda"),
}
ALL_COMPONENTS = ["text", "embedding", "vision"]


def target_dir(device: str, precision: str) -> Path:
    return HERE / f"{device}_{precision}" / "models"


# =============================================================================
# precision pass (shared by vision + embedding Olive configs)
# =============================================================================

def _precision_pass(precision: str, data_name: str):
    """Return the final {name: pass} dict for the given precision, or {} for fp32."""
    if precision == "int4":
        return {"int4": {
            "type": "OnnxBlockWiseRtnQuantization", "block_size": 128, "is_symmetric": True,
            "accuracy_level": 4, "save_as_external_data": True, "external_data_name": data_name}}
    if precision == "fp16":
        return {"fp16": {
            "type": "OnnxFloatToFloat16", "op_block_list": ["LayerNormalization", "Range"],
            "save_as_external_data": True, "external_data_name": data_name}}
    return {}  # fp32: keep the optimized float32 graph as-is


def _engine(device: str):
    dev, ep, _ = DEVICE[device]
    return {"target": {"type": "LocalSystem",
                       "accelerators": [{"device": dev, "execution_providers": [ep]}]}}


# =============================================================================
# Olive config builders (vision + embedding) — generated in-Python per target
# =============================================================================

def _vision_config(device: str, precision: str, out_dir: Path, model_src: Path | None = None) -> dict:
    cuda = device == "cuda"
    passes = {
        "c": {"type": "OnnxConversion", "use_dynamo_exporter": True},
        "gs": {"type": "GraphSurgeries", "surgeries": [
            {"surgeon": "PackedAttentionToLoopMHA"},
            {"surgeon": "ReciprocalMulToDiv"},
            {"surgeon": "RenameOutputDims", "output_idx": 0, "dim_idx": 0, "dim_name": "num_logical_patches"},
            {"surgeon": "RenameInputDims", "input_name": "image_grid_thw", "dim_idx": 0, "dim_name": "num_images"},
        ]},
        "ort": {"type": "OrtTransformersOptimization",
                "model_type": "vit" if cuda else "", "opt_level": 2 if cuda else 1,
                "only_onnxruntime": True},
    }
    if cuda:
        passes["dedup"] = {"type": "GraphSurgeries", "surgeries": [{"surgeon": "DeduplicateSubgraphInitializers"}]}
    passes["cast"] = {"type": "OnnxPeepholeOptimizer", "onnxscript_optimize": False,
                      "onnxoptimizer_optimize": False, "fuse_reshape_operations": False,
                      "fix_com_microsoft_opset": True, "cast_chain_elimination": True}
    passes["gs2"] = {"type": "GraphSurgeries", "surgeries": [{"surgeon": "GemmToMatMulAdd"}]}
    passes.update(_precision_pass(precision, "vision.onnx.data"))
    if precision == "fp16":  # cuda fp16 benefits from the memcpy cleanup
        passes["cleanup"] = {"type": "GraphSurgeries",
                             "surgeries": [{"surgeon": "DeduplicateNodes"}, {"surgeon": "RemoveMemcpy"}],
                             "save_as_external_data": True, "external_data_name": "vision.onnx.data"}
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(model_src) if model_src else MODEL_ID,
                        "model_loader": "get_vision_model", "model_script": "user_script.py",
                        "io_config": "get_vision_io_config", "dummy_inputs_func": "get_vision_dummy_inputs"},
        "passes": passes, "no_artifacts": True,
        "output_dir": str(out_dir / "vision.onnx"),
    }
    cfg.update(_engine(device))
    return cfg


def _embedding_config(device: str, precision: str, out_dir: Path, model_src: Path | None = None) -> dict:
    passes = {
        "convert": {"type": "OnnxConversion", "use_dynamo_exporter": False},
        "ort": {"type": "OrtTransformersOptimization", "model_type": "", "opt_level": 1,
                "only_onnxruntime": True},
        "cast": {"type": "OnnxPeepholeOptimizer", "onnxscript_optimize": False,
                 "onnxoptimizer_optimize": False, "fuse_reshape_operations": False,
                 "fix_com_microsoft_opset": True, "cast_chain_elimination": True},
        "gemm2mm": {"type": "GraphSurgeries", "surgeries": [{"surgeon": "GemmToMatMulAdd"}]},
    }
    passes.update(_precision_pass(precision, "embedding.onnx.data"))
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(model_src) if model_src else MODEL_ID,
                        "model_loader": "get_embedding_model", "model_script": "user_script.py",
                        "io_config": "get_embedding_io_config", "dummy_inputs_func": "get_embedding_dummy_inputs"},
        "passes": passes, "no_artifacts": True,
        "output_dir": str(out_dir / "embedding.onnx"),
    }
    cfg.update(_engine(device))
    return cfg


def _run_olive(cfg: dict, name: str):
    from olive import run
    with tempfile.NamedTemporaryFile("w", suffix=f"_{name}.json", delete=False, dir=str(HERE)) as f:
        json.dump(cfg, f, indent=2)
        cfg_path = f.name
    try:
        print(f"  Olive: {name} → {cfg['output_dir']}")
        run(cfg_path)
    finally:
        Path(cfg_path).unlink(missing_ok=True)


# =============================================================================
# text decoder via ModelBuilder (create_model) DIRECTLY (allows cpu fp32)
# =============================================================================

def build_text(device: str, precision: str, out_dir: Path, model_src: Path | None = None):
    from onnxruntime_genai.models.builder import create_model
    _, _, mb_ep = DEVICE[device]
    text_out = out_dir  # ModelBuilder writes text.onnx + genai_config.json + tokenizer here
    text_out.mkdir(parents=True, exist_ok=True)
    cache = HERE / ".mb_cache"  # ModelBuilder's own scratch dir (NOT the model download dir)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"  ModelBuilder: text decoder precision={precision} ep={mb_ep} → {text_out}/text.onnx")
    # create_model signature: (model_name, input_path, output_dir, precision,
    # execution_provider, cache_dir, **extra_options). When input_path is a local dir,
    # ModelBuilder loads from there; otherwise it downloads model_name from the hub.
    # `filename` is an extra_option passed as a kwarg (NOT extra_options={...}).
    input_path = str(model_src) if model_src else ""
    create_model(
        MODEL_ID, input_path, str(text_out), precision, mb_ep,
        cache_dir=str(cache), filename="text.onnx",
    )


# =============================================================================
# GenAI runtime config (embedding + vision sections + processor_config)
# =============================================================================

def update_genai_config(out_dir: Path, device: str):
    config_path = out_dir / "genai_config.json"
    with open(config_path) as f:
        config = json.load(f)

    if device == "cuda":
        provider_options = [{"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}]
    else:
        provider_options = []
    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}

    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": session_options,
    }
    config["model"]["vision"] = {
        "filename": "vision.onnx", "config_filename": "processor_config.json",
        "spatial_merge_size": 2, "tokens_per_second": 2.0, "patch_size": 16, "window_size": 64,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": session_options,
    }
    config["model"]["image_token_id"] = 151655
    config["model"]["video_token_id"] = 151656
    config["model"]["vision_start_token_id"] = 151652
    if config["search"].get("top_k") is None:
        config["search"]["top_k"] = 50
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    processor_config = {"processor": {"name": "qwen3_vl_image_processor", "transforms": [
        {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
        {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
        {"operation": {"name": "resize", "type": "Resize", "attrs": {
            "width": 540, "height": 360, "smart_resize": 1,
            "min_pixels": 65536, "max_pixels": 16777216, "patch_size": 16, "merge_size": 2}}},
        {"operation": {"name": "rescale", "type": "Rescale", "attrs": {"rescale_factor": 0.00392156862745098}}},
        {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "qwen3_vl": 1}}},
        {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}}},
    ]}}
    with open(out_dir / "processor_config.json", "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {out_dir / 'processor_config.json'}")


# =============================================================================
# Main
# =============================================================================

VALID = {("cpu", "int4"), ("cpu", "fp16"), ("cpu", "fp32"), ("cuda", "fp16"), ("cuda", "fp32")}


def prepare_download_dir(model_dir: Path) -> Path:
    """Download the full HF snapshot as a flat repo copy into model_dir and return it.

    Files land directly in model_dir (config.json, *.safetensors, tokenizer*, …) — no
    cache-hash layout. Both consumers are pointed straight at this dir afterwards:
    ModelBuilder via input_path, and Olive's user_script via CHANDRA_MODEL_DIR. One
    download, no HF cache involved."""
    import os
    from huggingface_hub import snapshot_download

    model_dir = model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(MODEL_ID, local_dir=str(model_dir))
    # user_script.py reads this to load the local snapshot instead of the HF repo id.
    os.environ["CHANDRA_MODEL_DIR"] = local
    print(f"  snapshot_download({MODEL_ID}) → {local}")
    return Path(local)


def main():
    p = argparse.ArgumentParser(description="Optimize datalab-to/chandra → ONNX (device × precision)")
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--precision", choices=["int4", "fp16", "fp32"], required=True)
    p.add_argument("--components", nargs="*", default=ALL_COMPONENTS,
                   help="subset of: text embedding vision (default: all)")
    p.add_argument("--skip-export", action="store_true", help="only regenerate genai/processor configs")
    p.add_argument("--model-dir", default="model",
                   help="folder to download the HF model snapshot into (flat repo copy). "
                        "Default: ./model next to this script. Pass empty to fetch from "
                        "the HF repo id directly (usual HF cache).")
    args = p.parse_args()

    if (args.device, args.precision) not in VALID:
        p.error(f"unsupported target {args.device}_{args.precision}; supported: "
                + ", ".join(f"{d}_{pr}" for d, pr in sorted(VALID)))

    out_dir = target_dir(args.device, args.precision)
    print(f"=== Target {args.device}_{args.precision} → {out_dir} ===")
    model_src = None
    if args.model_dir and not args.skip_export:
        print(f"=== Downloading model snapshot → {args.model_dir} ===")
        model_src = prepare_download_dir(Path(args.model_dir))

    if not args.skip_export:
        # text first: ModelBuilder writes genai_config.json + tokenizer that the others extend
        if "text" in args.components:
            build_text(args.device, args.precision, out_dir, model_src)
        if "embedding" in args.components:
            _run_olive(_embedding_config(args.device, args.precision, out_dir, model_src), "embedding")
        if "vision" in args.components:
            _run_olive(_vision_config(args.device, args.precision, out_dir, model_src), "vision")

    print("=== Generating GenAI + processor configs ===")
    update_genai_config(out_dir, args.device)
    print("\nDone →", out_dir)


if __name__ == "__main__":
    main()
