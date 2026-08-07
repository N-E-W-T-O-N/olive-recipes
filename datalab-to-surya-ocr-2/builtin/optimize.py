"""End-to-end ONNX optimization for datalab-to/surya-ocr-2 (Qwen3.5-VL, model_type qwen3_5).

Single Python-driven pipeline — every Olive config is generated in-Python (no static per-target
JSON), parameterized by device × precision:

    supported targets: cpu_int4  cpu_fp32  cuda_int4  cuda_fp16  webgpu_int4  webgpu_fp16
    (fp16 is NOT available on CPU — Olive's ModelBuilder rejects it: "FP16 is not supported on CPU".
    cuda_fp32 is NOT supported — genai's own builder resolves it to the "packed Attention" op,
    which this architecture's attention override doesn't support: KeyError('root_input') in
    make_packed_attention. Confirmed by tracing genai's builder source, not just guessed — see
    the VALID set below and HANDOFF.md. Every other cuda/webgpu combo resolves to
    GroupQueryAttention instead, which is safe.)

Three sub-models, re-composed by onnxruntime-genai at runtime via genai_config.json:
  - text      : the hybrid GatedDeltaNet decoder, built with the Olive **ModelBuilder pass**
                (type=ModelBuilder). Calling genai's create_model() directly works identically —
                the crash below is about (execution_provider, io_dtype), not which Python entry
                point calls it. See HANDOFF for the full trace.
  - vision    : Olive dynamo export + graph surgeries + precision pass.
  - embedding : Olive legacy export (+GemmToMatMulAdd) + precision pass; fuses text+image embeds.

Precision pass per component: int4 -> OnnxBlockWiseRtnQuantization | fp16 -> OnnxFloatToFloat16 |
fp32 -> (none). The text decoder takes its precision straight from the ModelBuilder pass.

Usage:
    python optimize.py --device cpu  --precision int4
    python optimize.py --device cuda --precision fp16
    python optimize.py --device cuda --precision int4 --components vision   # subset
    python optimize.py --device cpu  --precision int4 --skip-export         # regen configs only

Windows: run with PYTHONUTF8=1 — the vision dynamo export prints a '✅' that crashes on cp1252.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODEL_ID = "datalab-to/surya-ocr-2"
ENV_MODEL_DIR = "SURYA2_MODEL_DIR"
HERE = Path(__file__).parent

DEVICE = {  # device -> (olive accelerator device, execution provider)
    "cpu": ("cpu", "CPUExecutionProvider"),
    "cuda": ("gpu", "CUDAExecutionProvider"),
    "webgpu": ("gpu", "WebGpuExecutionProvider"),
}
ALL_COMPONENTS = ["text", "embedding", "vision"]

# fp16 on CPU is rejected by Olive's ModelBuilder ("FP16 is not supported on CPU").
#
# cuda_fp32 is EXCLUDED (not just "untested") — it is architecturally broken for this hybrid
# model, confirmed by tracing genai's own builder source (onnxruntime_genai/models/builders/
# base.py + qwen.py), not just by testing:
#   - set_io_dtype("fp32", "cuda", ...) -> ir.DataType.FLOAT (plain fp32 I/O)
#   - make_attention_init() picks the attention op via (execution_provider, io_dtype) lookup
#     tables: is_gqa_supported()'s allow-list has ("cuda", FLOAT16)/("cuda", BFLOAT16) but NOT
#     ("cuda", FLOAT) -> falls through to is_packed_attn_supported(), which DOES allow
#     ("cuda", FLOAT) -> op_type becomes "Attention" (the packed op).
#   - qwen.py's _make_full_attention() (Qwen3.5's doubled Q-projection/gating override) only
#     ever calls make_attention_op(..., q_path=, k_path=, v_path=...) — it never supplies
#     root_input. That's fine for GroupQueryAttention/MultiHeadAttention (which read
#     q_path/k_path/v_path), but make_packed_attention() unconditionally does
#     kwargs["root_input"] -> KeyError('root_input') for this exact (cuda, fp32) combination.
#   - cuda_int4 and cuda_fp16 both resolve io_dtype=FLOAT16 (see set_io_dtype), which IS in the
#     GQA allow-list -> safe. Only cuda_fp32 hits the broken packed-attention fallback.
# This is a genai bug (a gap in qwen.py's Qwen3.5 override, not anything specific to Olive vs
# create_model()) — not something this recipe can route around at the config level.
VALID = {
    ("cpu", "int4"), ("cpu", "fp32"),
    ("cuda", "int4"), ("cuda", "fp16"),
    ("webgpu", "int4"), ("webgpu", "fp16"),
}


def target_dir(device: str, precision: str) -> Path:
    return HERE / f"{device}_{precision}" / "models"


def _engine(device: str) -> dict:
    olive_dev, ep = DEVICE[device]
    return {"engine": {"target": {"type": "LocalSystem",
            "accelerators": [{"device": olive_dev, "execution_providers": [ep]}]}}}


def _precision_pass(precision: str, data_name: str) -> dict:
    """Final quant/cast pass for vision + embedding. fp32 -> no pass."""
    if precision == "int4":
        return {"int4": {"type": "OnnxBlockWiseRtnQuantization", "block_size": 128,
                         "is_symmetric": True, "accuracy_level": 4,
                         "save_as_external_data": True, "external_data_name": data_name}}
    if precision == "fp16":
        return {"fp16": {"type": "OnnxFloatToFloat16", "op_block_list": ["LayerNormalization", "Range"],
                         "save_as_external_data": True, "external_data_name": data_name}}
    return {}


# =============================================================================
# text decoder — Olive ModelBuilder PASS (allows cpu int4/fp32, gpu int4/fp16/fp32)
# =============================================================================

def _text_config(device: str, precision: str, out_dir: Path, model_src: Path) -> dict:
    mb = {"type": "ModelBuilder", "precision": precision,
          "extra_options": {"filename": "text.onnx", "prune_lm_head": True}}
    if precision == "int4":
        mb["int4_accuracy_level"] = 4
        mb["int4_algo_config"] = "k_quant_linear"
    cfg = {
        "input_model": {"type": "HfModel", "model_path": str(model_src)},
        # ModelBuilder writes text.onnx + genai_config.json + tokenizer into out_dir; prune_lm_head
        # drops the tied lm_head weights, TieWordEmbeddings re-wires the tie in the graph.
        "passes": {"m": mb, "t": {"type": "GraphSurgeries",
                                  "surgeries": [{"surgeon": "TieWordEmbeddings"}]}},
        "no_artifacts": True,
        "output_dir": str(out_dir / "text.onnx"),
    }
    cfg.update(_engine(device))
    return cfg


# =============================================================================
# vision + embedding — Olive export + surgeries + precision pass
# =============================================================================

def _vision_config(device: str, precision: str, out_dir: Path, model_src: Path) -> dict:
    gpu = device in ("cuda", "webgpu")
    edata = {"save_as_external_data": True, "external_data_name": "vision.onnx.data"}
    passes = {
        "c": {"type": "OnnxConversion", "use_dynamo_exporter": True, **edata},
        "gs": {"type": "GraphSurgeries", "surgeries":
               ([{"surgeon": "PackedAttentionToLoopMHA"}]
                + ([{"surgeon": "ReciprocalMulToDiv"}] if gpu else [])
                + [{"surgeon": "RenameOutputDims", "output_idx": 0, "dim_idx": 0,
                    "dim_name": "num_logical_patches"}]), **edata},
        "ort": {"type": "OrtTransformersOptimization",
                "model_type": "vit" if gpu else "", "opt_level": 2 if gpu else 1,
                "only_onnxruntime": True},
    }
    if gpu:
        passes["dedup"] = {"type": "GraphSurgeries",
                           "surgeries": [{"surgeon": "DeduplicateSubgraphInitializers"}], **edata}
    passes["cast"] = {"type": "OnnxPeepholeOptimizer", "onnxscript_optimize": False,
                      "onnxoptimizer_optimize": False, "fuse_reshape_operations": False,
                      "fix_com_microsoft_opset": True, "cast_chain_elimination": True, **edata}
    if not gpu:  # cpu path folds Gemm -> MatMul+Add (gpu keeps the dedup path instead)
        passes["gs2"] = {"type": "GraphSurgeries",
                         "surgeries": [{"surgeon": "GemmToMatMulAdd"}], **edata}
    passes.update(_precision_pass(precision, "vision.onnx.data"))
    if precision == "fp16":
        passes["cleanup"] = {"type": "GraphSurgeries",
                             "surgeries": [{"surgeon": "DeduplicateNodes"}], **edata}
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(model_src),
                        "model_loader": "get_vision_model", "model_script": "user_script.py",
                        "io_config": "get_vision_io_config", "dummy_inputs_func": "get_vision_dummy_inputs"},
        "passes": passes, "no_artifacts": True, "output_dir": str(out_dir / "vision.onnx"),
    }
    cfg.update(_engine(device))
    return cfg


def _embedding_config(device: str, precision: str, out_dir: Path, model_src: Path) -> dict:
    edata = {"save_as_external_data": True, "external_data_name": "embedding.onnx.data"}
    passes = {
        "convert": {"type": "OnnxConversion", "use_dynamo_exporter": False, **edata},
        "ort": {"type": "OrtTransformersOptimization", "model_type": "", "opt_level": 1,
                "only_onnxruntime": True, **edata},
        "cast": {"type": "OnnxPeepholeOptimizer", "onnxscript_optimize": False,
                 "onnxoptimizer_optimize": False, "fuse_reshape_operations": False,
                 "fix_com_microsoft_opset": True, "cast_chain_elimination": True, **edata},
        "gemm2mm": {"type": "GraphSurgeries", "surgeries": [{"surgeon": "GemmToMatMulAdd"}], **edata},
    }
    passes.update(_precision_pass(precision, "embedding.onnx.data"))
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(model_src),
                        "model_loader": "get_embedding_model", "model_script": "user_script.py",
                        "io_config": "get_embedding_io_config", "dummy_inputs_func": "get_embedding_dummy_inputs"},
        "passes": passes, "no_artifacts": True, "output_dir": str(out_dir / "embedding.onnx"),
    }
    cfg.update(_engine(device))
    return cfg


def _run_olive(cfg: dict, name: str):
    from olive import run
    with tempfile.NamedTemporaryFile("w", suffix=f"_{name}.json", delete=False, dir=str(HERE)) as f:
        json.dump(cfg, f, indent=2)
        cfg_path = f.name
    try:
        print(f"  Olive: {name} ({cfg['passes'] and list(cfg['passes'])}) -> {cfg['output_dir']}")
        run(cfg_path)
    finally:
        Path(cfg_path).unlink(missing_ok=True)


# =============================================================================
# GenAI runtime config + processor + tokenizer
# =============================================================================

def update_genai_config(out_dir: Path, device: str):
    config_path = out_dir / "genai_config.json"
    if not config_path.exists():
        print(f"  [skip] {config_path} missing — the text decoder didn't produce it "
              f"(build 'text' first / check for an OOM on the int4 serialize).")
        return
    with open(config_path) as f:
        config = json.load(f)

    if device == "cuda":
        provider_options = [{"cuda": {"enable_cuda_graph": "1", "enable_skip_layer_norm_strict_mode": "1"}}]
        # Vision/embedding have Loop nodes (per-ViT-block) that break CUDA graph capture -> off.
        vision_provider_options = [{"cuda": {"enable_cuda_graph": "0", "enable_skip_layer_norm_strict_mode": "1"}}]
    elif device == "webgpu":
        provider_options = vision_provider_options = [{"webgpu": {}}]
    else:
        provider_options = vision_provider_options = []
    session_options = {"log_id": "onnxruntime-genai", "provider_options": provider_options}
    vision_session_options = {"log_id": "onnxruntime-genai", "provider_options": vision_provider_options}

    config["model"]["decoder"]["session_options"] = session_options
    config["model"]["embedding"] = {
        "filename": "embedding.onnx",
        "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
        "session_options": vision_session_options,
    }
    config["model"]["vision"] = {
        "filename": "vision.onnx", "config_filename": "processor_config.json",
        "spatial_merge_size": 2, "tokens_per_second": 2.0, "patch_size": 16,
        "inputs": {"pixel_values": "pixel_values", "image_grid_thw": "image_grid_thw"},
        "outputs": {"image_features": "image_features"},
        "session_options": vision_session_options,
    }
    # surya-ocr-2 custom 65425-vocab WordLevel tokenizer: special tokens remapped to LOW ids
    # (tokenizer.json added_tokens). config.json text_config.eos_token_id (248044) is OUT OF VOCAB;
    # the real eos/pad come from generation_config.json (eos=2, pad=0). The model has no real BOS
    # (config's bos_token_id is null), but genai's C++ JSON parser REQUIRES a number here
    # ("Expected a number but saw a null") — mirror the 0.8B recipe's convention of bos=eos as an
    # unused placeholder (bos is never sampled; generation starts from the chat-template prompt).
    config["model"]["bos_token_id"] = 2
    config["model"]["eos_token_id"] = [2]
    config["model"]["pad_token_id"] = 0
    config["model"]["image_token_id"] = 11
    config["model"]["video_token_id"] = 12
    config["model"]["vision_start_token_id"] = 9
    config["search"]["top_k"] = 1
    if config["search"].get("top_p") is None:
        config["search"]["top_p"] = 1.0

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Updated {config_path}")

    processor_config = {"processor": {"name": "qwen2_5_image_processor", "transforms": [
        {"operation": {"name": "decode_image", "type": "DecodeImage", "attrs": {"color_space": "RGB"}}},
        {"operation": {"name": "convert_to_rgb", "type": "ConvertRGB"}},
        {"operation": {"name": "resize", "type": "Resize", "attrs": {
            "width": 960, "height": 672, "smart_resize": 1,
            "min_pixels": 65536, "max_pixels": 16777216, "patch_size": 16, "merge_size": 2}}},
        {"operation": {"name": "rescale", "type": "Rescale", "attrs": {"rescale_factor": 0.00392156862745098}}},
        {"operation": {"name": "normalize", "type": "Normalize", "attrs": {
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "qwen2_5_vl": 1}}},
        {"operation": {"name": "patch_image", "type": "PatchImage", "attrs": {
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}}},
    ]}}
    with open(out_dir / "processor_config.json", "w") as f:
        json.dump(processor_config, f, indent=2)
    print(f"  Created {out_dir / 'processor_config.json'}")


def fix_tokenizer(out_dir: Path):
    """surya-ocr-2 ships a CHARACTER-LEVEL WordLevel tokenizer (65425 vocab, single Split
    pre-tokenizer Regex='.', tokenizer_class 'TokenizersBackend'). onnxruntime-genai's C++
    tokenizer runtime does NOT understand model.type=WordLevel: it crashes in
    create_multimodal_processor() with "Cannot know how to parse line: " (confirmed empirically —
    it unconditionally expects a BPE-shaped model with a `merges` list).

    FIX (verified working end-to-end, 2026-07-29): rewrite model.type -> "BPE" with an EMPTY
    merges list, keeping the same vocab/unk_token. This is functionally identical to WordLevel
    here: the pre-tokenizer already isolates exactly one Unicode character per piece
    (Split(Regex="." behavior="Isolated")), so BPE with zero merge rules degenerates to a direct
    single-character vocab lookup — exactly what WordLevel does. Confirmed by running real
    document-image inference through the built ONNX model (coherent structured OCR/layout output).
    Idempotent: no-ops if the file is already BPE (e.g. re-run via --skip-export).
    """
    tk_path = out_dir / "tokenizer.json"
    if not tk_path.exists():
        return
    tk = json.loads(tk_path.read_text(encoding="utf-8"))
    model = tk.get("model", {})
    if model.get("type") == "WordLevel":
        model["type"] = "BPE"
        model["merges"] = []
        model.setdefault("dropout", None)
        model.setdefault("continuing_subword_prefix", "")
        model.setdefault("end_of_word_suffix", "")
        model.setdefault("fuse_unk", False)
        model.setdefault("byte_fallback", False)
        tk_path.write_text(json.dumps(tk, ensure_ascii=False), encoding="utf-8")
        print("  [tokenizer] converted WordLevel -> BPE(merges=[]) for genai C++ tokenizer compat "
              "(same per-character vocab lookup; pre_tokenizer already isolates single chars)")
    else:
        print(f"  [tokenizer] model.type={model.get('type')!r} — already genai-compatible, no-op")


# =============================================================================
# model source resolution (prefer local ./model, else HF download)
# =============================================================================

def resolve_model_src(model_dir: str, skip_download: bool) -> Path:
    """Return a local dir with config.json + *.safetensors; download the HF snapshot if needed.
    Sets ENV_MODEL_DIR so user_script.py loads the same snapshot."""
    from huggingface_hub import snapshot_download

    p = Path(model_dir).resolve() if model_dir else (HERE / "model")
    if (p / "config.json").exists():
        local = str(p)
    elif skip_download:
        raise SystemExit(f"--skip-download set but no snapshot at {p} (config.json missing)")
    else:
        p.mkdir(parents=True, exist_ok=True)
        local = snapshot_download(MODEL_ID, local_dir=str(p))
    os.environ[ENV_MODEL_DIR] = local
    print(f"  model source: {local}")
    return Path(local)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Optimize datalab-to/surya-ocr-2 -> ONNX (device x precision)")
    p.add_argument("--device", choices=list(DEVICE), required=True)
    p.add_argument("--precision", choices=["int4", "fp16", "fp32"], required=True)
    p.add_argument("--components", nargs="*", default=ALL_COMPONENTS,
                   help="subset of: text embedding vision (default: all)")
    p.add_argument("--skip-export", action="store_true", help="only regenerate genai/processor configs")
    p.add_argument("--model-dir", default="model", help="local HF snapshot dir (default ./model). "
                   "Empty string forces an HF download.")
    p.add_argument("--skip-download", action="store_true", help="require the existing ./model snapshot")
    p.add_argument("--no-isolate", action="store_true",
                   help="build all components in ONE process (may OOM the int4 text serialize on big "
                        "checkpoints). Default: one subprocess per component for fresh memory.")
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if (args.device, args.precision) not in VALID:
        hint = ""
        if (args.device, args.precision) == ("cpu", "fp16"):
            hint = "  (fp16 is GPU-only — Olive rejects FP16 on CPU)"
        elif (args.device, args.precision) == ("cuda", "fp32"):
            hint = ("  (cuda_fp32 is architecturally broken for this model — genai's builder "
                    "resolves it to the packed-Attention op, which qwen3_5's attention override "
                    "doesn't support: KeyError('root_input'). Use cuda_int4 or cuda_fp16 instead.)")
        p.error(f"unsupported target {args.device}_{args.precision}; supported: "
                + ", ".join(f"{d}_{pr}" for d, pr in sorted(VALID)) + hint)

    out_dir = target_dir(args.device, args.precision)
    print(f"=== Target {args.device}_{args.precision} -> {out_dir} ===")

    model_src = None
    if not args.skip_export:
        model_src = resolve_model_src(args.model_dir, args.skip_download or args._child)

    # Memory isolation: re-invoke once per component in a fresh subprocess so the int4 text
    # serialize doesn't share peak memory with the vision/embedding Olive passes.
    if not args.skip_export and not args._child and not args.no_isolate and len(args.components) > 1:
        order = [c for c in ALL_COMPONENTS if c in args.components]  # text first (writes genai cfg)
        print(f"=== Isolating {order} in subprocesses ===")
        for comp in order:
            cmd = [sys.executable, str(HERE / "optimize.py"), "--device", args.device,
                   "--precision", args.precision, "--components", comp, "--_child",
                   "--skip-download", "--model-dir", str(model_src)]
            print(f"\n=== [subprocess] {comp} ===")
            rc = subprocess.run(cmd, env={**os.environ}).returncode
            if rc != 0:
                print(f"  [warn] component '{comp}' failed (rc {rc}) — continuing")
        print("\n=== Generating GenAI + processor configs ===")
        update_genai_config(out_dir, args.device)
        fix_tokenizer(out_dir)
        print("\nDone ->", out_dir)
        return

    if not args.skip_export:
        if "text" in args.components:      # text first: writes genai_config.json + tokenizer
            _run_olive(_text_config(args.device, args.precision, out_dir, model_src), "text")
        if "embedding" in args.components:
            _run_olive(_embedding_config(args.device, args.precision, out_dir, model_src), "embedding")
        if "vision" in args.components:
            _run_olive(_vision_config(args.device, args.precision, out_dir, model_src), "vision")

    if not args._child:                    # children skip config gen; parent (or single run) does it
        print("=== Generating GenAI + processor configs ===")
        update_genai_config(out_dir, args.device)
        fix_tokenizer(out_dir)
        print("\nDone ->", out_dir)


if __name__ == "__main__":
    main()
