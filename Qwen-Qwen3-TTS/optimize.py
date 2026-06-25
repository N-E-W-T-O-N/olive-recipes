# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3",
#   "torch",
#   "torchvision",
#   "torchaudio",
#   "olive-ai",
#   "onnx",
#   "onnxruntime>=1.20",
#   "onnxruntime-genai",
#   "safetensors",
#   "numpy",
#   "tabulate",
#   "huggingface_hub",
#   "accelerate",
#   "librosa",
#   "numba>=0.60.0",
#   "llvmlite>=0.43.0",
#   "soundfile",
# ]
# ///
# numba/llvmlite pinned >=0.60/0.43: librosa otherwise resolves to numba 0.53.1 →
# llvmlite 0.36, which fails to build on Python 3.12 ("only versions <3.10 supported").
"""Qwen3-TTS → ONNX sub-parts.  Run with:  uv run optimize.py --model <name|path> ...

Self-contained uv script: pins transformers==4.57.3 (the version the vendored
qwen_tts modeling needs) in an isolated env, leaving the shared 5.10.2 venv alone.

Handles three model kinds (auto-detected from config.json `model_type`):
  qwen3_tts                 → TTS  : talker (+ code_predictor) [+ embedded tokenizer]
  qwen3_tts_tokenizer_12hz  → codec: tok_encoder + tok_decoder

Outputs flat sub-parts under  onnx/{device}_{precision}/  + manifest.json.

  --device {cpu,cuda}   --precision {int4,fp16,fp32}

The talker is a custom Qwen3+MROPE LLM (dual codec/text embedding, codec_head) and is
NOT onnxruntime-genai ModelBuilder-compatible (MROPE + dual embedding don't map to a
stock Qwen3) — it is exported via Olive like the rest. (ModelBuilder remap is attempted
only when the talker is detectably stock; otherwise Olive is used.)

Usage:
  uv run optimize.py --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign  --device cpu --precision int4
  uv run optimize.py --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice  --device cpu --precision fp16
  uv run optimize.py --model Qwen/Qwen3-TTS-Tokenizer-12Hz         --device cpu --precision fp32
  uv run optimize.py --model voicedesign --skip-download           # use local dir
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROVIDER = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
OLIVE_DEV = {"cpu": "cpu", "cuda": "gpu"}

# component → (loader, io_config, dummy_inputs, model_subdir or "")
TTS_COMPONENTS = {
    "text_embed":     ("get_text_embed_model", "get_text_embed_io_config", "get_text_embed_dummy_inputs", ""),
    "codec_embed":    ("get_codec_embed_model", "get_codec_embed_io_config", "get_codec_embed_dummy_inputs", ""),
    "talker":         ("get_talker_model", "get_talker_io_config", "get_talker_dummy_inputs", ""),
    "talker_cache":   ("get_talker_cache_model", "get_talker_cache_io_config", "get_talker_cache_dummy_inputs", ""),
    "code_predictor": ("get_code_predictor_model", "get_code_predictor_io_config", "get_code_predictor_dummy_inputs", ""),
    "residual_embed": ("get_residual_embed_model", "get_residual_embed_io_config", "get_residual_embed_dummy_inputs", ""),
    "tok_encoder":    ("get_tok_encoder_model", "get_tok_encoder_io_config", "get_tok_encoder_dummy_inputs", "speech_tokenizer"),
    "tok_decoder":    ("get_tok_decoder_model", "get_tok_decoder_io_config", "get_tok_decoder_dummy_inputs", "speech_tokenizer"),
    "speaker_encoder": ("get_speaker_encoder_model", "get_speaker_encoder_io_config", "get_speaker_encoder_dummy_inputs", ""),
}
TOKENIZER_COMPONENTS = {
    "tok_encoder": ("get_tok_encoder_model", "get_tok_encoder_io_config", "get_tok_encoder_dummy_inputs", ""),
    "tok_decoder": ("get_tok_decoder_model", "get_tok_decoder_io_config", "get_tok_decoder_dummy_inputs", ""),
}
# audio codecs + speaker encoder stay fp32 (DAC/conv int4/fp16 too lossy; ECAPA small)
FP32_ONLY = {"tok_encoder", "tok_decoder", "speaker_encoder"}


def output_root(device, precision):
    return HERE / "onnx" / f"{device}_{precision}"


def resolve_model(model: str, skip_download: bool) -> Path:
    """Local dir (relative or absolute) or HF id → local checkpoint dir."""
    p = Path(model)
    if p.is_dir():
        return p
    local = HERE / model.split("/")[-1].lower()
    if local.is_dir():
        return local
    if skip_download:
        raise FileNotFoundError(f"{model} not found locally and --skip-download set")
    from huggingface_hub import snapshot_download
    dst = HERE / model.split("/")[-1]
    print(f"  Downloading {model} → {dst}")
    snapshot_download(repo_id=model, local_dir=str(dst))
    return dst


def detect_type(model_dir: Path) -> str:
    cfg = json.loads((model_dir / "config.json").read_text())
    return cfg.get("model_type", "")


def build_component(name, funcs, model_dir, out_dir, device, precision):
    from olive import run
    loader, io_cfg, dummy, subdir = funcs
    src = model_dir / subdir if subdir else model_dir
    prec = "fp32" if name in FP32_ONLY else precision
    passes = {"c": {"type": "OnnxConversion", "target_opset": 20}}
    # Transformers using create_causal_mask aren't torch.onnx(TorchScript)-traceable
    # → dynamo exporter (codec parts + the MROPE talker).
    if name in FP32_ONLY or name in ("talker", "talker_cache"):
        passes["c"]["use_dynamo_exporter"] = True
    if prec == "int4":
        passes["q"] = {"type": "OnnxBlockwiseRtnQuantization", "bits": 4,
                       "block_size": 32, "is_symmetric": False}
    elif prec == "fp16":
        passes["h"] = {"type": "OnnxFloatToFloat16"}

    tmp = (out_dir / f"_{name}_tmp").resolve()
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(src.resolve()),
                        "model_loader": loader,
                        "model_script": str((HERE / "user_script.py").resolve()),
                        "io_config": io_cfg, "dummy_inputs_func": dummy},
        "systems": {"local_system": {"type": "LocalSystem", "accelerators": [
            {"device": OLIVE_DEV[device], "execution_providers": [PROVIDER[device]]}]}},
        "passes": passes, "target": "local_system", "log_severity_level": 2,
        "output_dir": str(tmp), "cache_dir": str((out_dir / "_ocache").resolve()),
        "no_artifacts": True,
    }
    tmp_json = out_dir / f"_{name}.json"
    tmp_json.write_text(json.dumps(cfg, indent=2))
    print(f"  Olive export {name} (precision={prec}) → {out_dir}/{name}.onnx")
    run(str(tmp_json))
    tmp_json.unlink(missing_ok=True)
    moved_data = False
    for f in tmp.glob("model.onnx*"):
        dst = out_dir / f.name.replace("model.onnx", f"{name}.onnx")
        if dst.exists():
            dst.unlink()
        shutil.move(str(f), str(dst))
        if dst.name.endswith(".onnx.data"):
            moved_data = True
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(out_dir / "_ocache", ignore_errors=True)
    if moved_data:
        relink_external_data(out_dir / f"{name}.onnx", f"{name}.onnx.data")


def relink_external_data(onnx_path, new_data_name):
    """Point a renamed model's external_data references at the renamed .data file.

    Olive writes `model.onnx` + `model.onnx.data`; we flatten to `{name}.onnx[.data]`
    but the proto still references `model.onnx.data`, so onnxruntime fails to load
    (`External data path does not exist`). Rewrite each tensor's `location` entry.
    Proto-only (load_external_data=False) so we never pull the multi-GB blob into RAM.
    """
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    n = 0
    for t in m.graph.initializer:
        if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL:
            for kv in t.external_data:
                if kv.key == "location":
                    kv.value = new_data_name
                    n += 1
    onnx.save(m, str(onnx_path))      # proto only; .data already on disk
    print(f"    relinked {n} external-data refs → {new_data_name}")


def write_manifest(out_dir, model, device, precision, model_kind, built):
    # include every sub-part present on disk (robust to incremental builds)
    present = sorted({p.stem for p in out_dir.glob("*.onnx")})
    manifest = {
        "model_id": model, "model_kind": model_kind, "device": device,
        "precision": precision, "execution_provider": PROVIDER[device],
        "sub_models": {n: {"filename": f"{n}.onnx"} for n in present},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote {out_dir / 'manifest.json'}")


def main():
    ap = argparse.ArgumentParser(description="Qwen3-TTS → ONNX sub-parts")
    ap.add_argument("--model", required=True, help="HF id or local dir")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["int4", "fp16", "fp32"], default="int4")
    ap.add_argument("--components", nargs="*", default=None, help="subset to build")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-isolate", action="store_true",
                    help="build everything in ONE process (may OOM on the talker). Default: "
                         "each component runs in its own subprocess for fresh memory.")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)  # internal
    args = ap.parse_args()

    model_dir = resolve_model(args.model, args.skip_download)
    mtype = detect_type(model_dir)
    if mtype == "qwen3_tts":
        kind, comp_map = "tts", TTS_COMPONENTS
    elif mtype.startswith("qwen3_tts_tokenizer"):
        kind, comp_map = "tokenizer", TOKENIZER_COMPONENTS
    else:
        raise ValueError(f"Unknown model_type '{mtype}' in {model_dir}")

    if args.components:
        components = [c for c in args.components if c in comp_map]
    else:
        components = list(comp_map)
        # speaker_encoder exists only in `base` checkpoints — skip it by default elsewhere
        # (it raises in the wrapper otherwise). Still buildable if requested explicitly.
        tts_type = json.loads((model_dir / "config.json").read_text()).get("tts_model_type")
        if "speaker_encoder" in components and tts_type != "base":
            components.remove("speaker_encoder")
        # talker and talker_cache hold the SAME transformer weights (no-cache vs KV-cache
        # forward). Shipping both duplicates ~870 MB. Default to talker_cache only (it does
        # prefill+decode, is faster O(n), and is what inference auto-uses); the plain no-cache
        # `talker` is still buildable explicitly if a simpler graph is wanted.
        if "talker" in components and "talker_cache" in components:
            components.remove("talker")
    out_dir = output_root(args.device, args.precision)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model {args.model} [{kind}] | {args.device}/{args.precision} → {out_dir}")
    print(f"Components: {components}\n")

    # ── memory-isolation: re-invoke this script once per component in a fresh subprocess.
    # The talker's big dynamo/float16 pass commits a lot of virtual memory; stacking it on
    # top of others in one process OOM/pagefile-kills on Windows. One process per component
    # releases everything between builds. (Reuses sys.executable = the resolved uv env.)
    if not args._child and not args.no_isolate and len(components) > 1:
        print(f"Isolating {len(components)} components in subprocesses (avoids OOM)…\n")
        ok, failed = [], []
        for comp in components:
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "--model", args.model, "--device", args.device,
                   "--precision", args.precision, "--components", comp, "--_child"]
            if args.skip_download:
                cmd.append("--skip-download")
            print(f"=== [subprocess] {comp} ===")
            rc = subprocess.run(cmd).returncode
            (ok if rc == 0 else failed).append(comp)
            if rc != 0:
                print(f"  [warn] component '{comp}' failed (rc {rc}) — continuing")
            print()
        write_manifest(out_dir, args.model, args.device, args.precision, kind, ok)
        print(f"\nDone → {out_dir}   built={ok}" + (f"  FAILED={failed}" if failed else ""))
        return

    # direct build (single component, --_child subprocess, or --no-isolate)
    built = []
    for name in components:
        print(f"=== {name} ===")
        build_component(name, comp_map[name], model_dir, out_dir, args.device, args.precision)
        built.append(name); print()
    if not args._child:                      # children skip manifest; parent writes it
        write_manifest(out_dir, args.model, args.device, args.precision, kind, built)
    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
