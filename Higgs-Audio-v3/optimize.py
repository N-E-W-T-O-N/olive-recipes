"""ONNX optimization pipeline for bosonai/higgs-audio-v3-tts-4b.

Project layout produced:

  model/                         # downloaded checkpoint (gitignored)
  qwen3_standalone/              # extracted Qwen3-4B LLM (HF dir; reference/embeds)
  onnx/{device}_{precision}/     # ALL sub-parts, flat, unique filenames:
      llm_decoder.onnx (+ .data) # Qwen3-4B decoder (ModelBuilder; inputs_embeds→hidden)
      genai_config.json          #   (onnxruntime-genai config for llm_decoder)
      tokenizer.json, ...
      audio_embed.onnx           # fused multi-codebook embedding  (Olive)
      audio_heads.onnx           # fused multi-codebook head (tied) (Olive)
      audio_tokenizer.onnx (+.data) # Higgs v2 codec decoder: codes→waveform (Olive)
      manifest.json

device   ∈ {cpu, cuda, openvino}   precision ∈ {int4, fp16, fp32}
The audio_tokenizer (neural codec) is always exported fp32 (INT4 is too lossy for a
DAC decoder), regardless of --precision.

device=openvino reuses the proven CPU build (EP-agnostic ONNX) and records the
OpenVINO EP in the manifest for Intel CPU/iGPU/NPU inference. Running on OpenVINO
needs a MATCHING `onnxruntime-openvino` build (it replaces plain `onnxruntime` —
they cannot coexist; mismatched versions hard-fail with Error 127). See SUPPORT.md.

Usage:
  python optimize.py --device cpu      --precision int4
  python optimize.py --device openvino --precision int4   # Intel; → onnx/openvino_int4/
  python optimize.py --device cpu --precision int4 --components audio_tokenizer --skip-download
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

MODEL_NAME = "bosonai/higgs-audio-v3-tts-4b"
MODEL_DIR = "model"
STANDALONE_DIR = "qwen3_standalone"          # top-level
ALL_COMPONENTS = ["llm_decoder", "audio_embed", "audio_heads", "audio_tokenizer"]

# ModelBuilder (onnxruntime-genai) execution_provider for the LLM build.
# genai has no OpenVINO EP, so 'openvino' builds the LLM as a CPU int4 model
# (it then runs under the OpenVINO EP at inference — the ONNX is EP-agnostic).
EP_FOR_DEVICE = {"cpu": "cpu", "cuda": "cuda", "openvino": "openvino"}

# Inference/runtime EP recorded in the manifest.
PROVIDER = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
            "openvino": "OpenVINOExecutionProvider"}

# Build-time Olive accelerator (device, EP) for the audio sub-parts. OnnxConversion
# /quant run on CPU regardless of the target EP, so 'openvino' builds on CPU and only
# the manifest provider switches to OpenVINO — no openvino.dll needed at build time.
OLIVE_ACCEL = {"cpu": ("cpu", "CPUExecutionProvider"),
               "cuda": ("gpu", "CUDAExecutionProvider"),
               "openvino": ("cpu", "CPUExecutionProvider")}

AUDIO_FUNCS = {
    "audio_embed": ("get_audio_embed_model", "get_audio_embed_io_config",
                    "get_audio_embed_dummy_inputs"),
    "audio_heads": ("get_audio_heads_model", "get_audio_heads_io_config",
                    "get_audio_heads_dummy_inputs"),
    "audio_tokenizer": ("get_audio_tokenizer_model", "get_audio_tokenizer_io_config",
                        "get_audio_tokenizer_dummy_inputs"),
}


def output_root(device: str, precision: str) -> Path:
    return Path("onnx") / f"{device}_{precision}"


# --------------------------------------------------------------------------- #
# download + LLM extraction
# --------------------------------------------------------------------------- #

def download_model(model_id: str, dest: str = MODEL_DIR) -> str:
    from huggingface_hub import snapshot_download
    dest_path = Path(dest).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {model_id} → {dest_path} ...")
    snapshot_download(repo_id=model_id, local_dir=str(dest_path),
                      allow_patterns=["*.json", "*.txt", "*.safetensors",
                                      "*.safetensors.index.json", "tokenizer*",
                                      "merges.txt", "vocab.json", "*.jinja"])
    print(f"  Download complete: {dest_path}")
    return str(dest_path)


def extract_llm(model_path: str) -> str:
    sys.path.insert(0, str(Path(__file__).parent))
    from user_script import extract_qwen3_standalone
    return str(Path(extract_qwen3_standalone(model_path, ".")).resolve())


def _set_genai_provider(genai_cfg_path: Path, device: str, ov_device: str = "CPU") -> None:
    """Record the runtime EP in the LLM's genai_config session_options.

    create_model writes empty provider_options (= CPU). For openvino we inject the
    OpenVINO provider with the chosen device_type (CPU/GPU/NPU/AUTO) so an
    OpenVINO-enabled onnxruntime-genai runtime picks it up. (genai still needs an
    OpenVINO-capable build to honor it — see SUPPORT.md.)
    """
    cfg = json.loads(genai_cfg_path.read_text())
    so = cfg["model"]["decoder"].setdefault("session_options", {})
    if device == "openvino":
        so["provider_options"] = [{"OpenVINO": {"device_type": ov_device}}]
    elif device == "cuda":
        so["provider_options"] = [{"cuda": {}}]
    else:
        so["provider_options"] = []
    genai_cfg_path.write_text(json.dumps(cfg, indent=4))


def build_llm(qwen3_abs: str, out_dir: Path, device: str, precision: str,
              ov_device: str = "CPU") -> None:
    """LLM decoder via onnxruntime-genai ModelBuilder → flat onnx/{dp}/llm_decoder.onnx.

    exclude_embeds + exclude_lm_head: decoder takes inputs_embeds → hidden_states
    (correct sub-part contract; avoids the onnx_ir serialization crash on the 4B
    embedding/lm_head). filename gives the unique flat name with correct external-data.

    device=openvino reuses an identical CPU-built LLM if present (genai builds the LLM
    on CPU either way — only the genai_config provider differs), skipping the slow,
    RAM-heavy int4 serialization.
    """
    import onnx
    # Reuse the byte-identical CPU build for openvino (only genai_config provider differs)
    if device == "openvino":
        cpu_dir = output_root("cpu", precision)
        if (cpu_dir / "llm_decoder.onnx").exists():
            print(f"  Reusing CPU-built LLM from {cpu_dir} (identical graph) + OpenVINO provider")
            for fn in ("llm_decoder.onnx", "llm_decoder.onnx.data", "genai_config.json",
                       "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                       "chat_template.jinja"):
                if (cpu_dir / fn).exists():
                    shutil.copy2(cpu_dir / fn, out_dir / fn)
            _set_genai_provider(out_dir / "genai_config.json", device, ov_device)
            print("  LLM done (reused).")
            return

    from onnxruntime_genai.models.builder import create_model
    cache = (out_dir / "_gbuild").resolve()
    tmp = (out_dir / "_llm_tmp").resolve()
    print(f"  ModelBuilder precision={precision} ep={EP_FOR_DEVICE[device]} → {out_dir}/llm_decoder.onnx")
    # exclude_embeds/exclude_lm_head MUST be direct kwargs. Use the DEFAULT output
    # filename (model.onnx): passing extra_options={'filename':...} re-triggers an
    # onnx_ir serialization crash on this 4B model. We rename to the flat unique
    # name afterwards via a proto-only external-data relink (no re-serialization).
    create_model(
        model_name="", input_path=qwen3_abs, output_dir=str(tmp),
        precision=precision, execution_provider=EP_FOR_DEVICE[device], cache_dir=str(cache),
        exclude_embeds=True, exclude_lm_head=True,
    )
    # relink external data: model.onnx(.data) → llm_decoder.onnx(.data)
    m = onnx.load(str(tmp / "model.onnx"), load_external_data=False)
    for t in m.graph.initializer:
        for ed in t.external_data:
            if ed.key == "location":
                ed.value = "llm_decoder.onnx.data"
    onnx.save(m, str(out_dir / "llm_decoder.onnx"))            # proto only; keeps external refs
    shutil.move(str(tmp / "model.onnx.data"), str(out_dir / "llm_decoder.onnx.data"))
    # genai_config + tokenizer alongside, with patched filename
    gcfg = json.loads((tmp / "genai_config.json").read_text())
    gcfg["model"]["decoder"]["filename"] = "llm_decoder.onnx"
    (out_dir / "genai_config.json").write_text(json.dumps(gcfg, indent=4))
    _set_genai_provider(out_dir / "genai_config.json", device, ov_device)
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "chat_template.jinja"):
        src_fn = tmp / fn if (tmp / fn).exists() else Path(STANDALONE_DIR) / fn
        if src_fn.exists():
            shutil.copy2(src_fn, out_dir / fn)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    print("  LLM done.")


# --------------------------------------------------------------------------- #
# audio sub-parts (Olive PyTorchModel) → flat {component}.onnx
# --------------------------------------------------------------------------- #

def build_audio_part(component: str, model_path: str, out_dir: Path,
                     device: str, precision: str) -> None:
    from olive import run
    loader, io_cfg, dummy = AUDIO_FUNCS[component]
    # neural codec stays fp32 even when the LLM is int4 (DAC int4 is too lossy)
    prec = "fp32" if component == "audio_tokenizer" else precision
    passes = {"c": {"type": "OnnxConversion", "target_opset": 20}}
    if prec == "int4":
        passes["q"] = {"type": "OnnxBlockwiseRtnQuantization", "bits": 4,
                       "block_size": 32, "is_symmetric": False}
    elif prec == "fp16":
        passes["h"] = {"type": "OnnxFloatToFloat16"}

    tmp_out = (out_dir / f"_{component}_tmp").resolve()
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(Path(model_path).resolve()),
                        "model_loader": loader,
                        "model_script": str((Path(__file__).parent / "user_script.py").resolve()),
                        "io_config": io_cfg, "dummy_inputs_func": dummy},
        "systems": {"local_system": {"type": "LocalSystem", "accelerators": [
            {"device": OLIVE_ACCEL[device][0],
             "execution_providers": [OLIVE_ACCEL[device][1]]}]}},
        "passes": passes, "target": "local_system", "log_severity_level": 2,
        "output_dir": str(tmp_out), "cache_dir": str((out_dir / "_ocache").resolve()),
        "no_artifacts": True,
    }
    tmp_json = out_dir / f"_{component}.json"
    tmp_json.write_text(json.dumps(cfg, indent=2))
    print(f"  Olive export {component} (precision={prec}) → {out_dir}/{component}.onnx")
    run(str(tmp_json))
    tmp_json.unlink(missing_ok=True)

    # flatten: move tmp_out/model.onnx(.data) → out_dir/{component}.onnx(.data)
    for src in tmp_out.glob("model.onnx*"):
        dst = out_dir / src.name.replace("model.onnx", f"{component}.onnx")
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    shutil.rmtree(tmp_out, ignore_errors=True)
    shutil.rmtree(out_dir / "_ocache", ignore_errors=True)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def write_manifest(out_dir: Path, device: str, precision: str, ov_device: str = "CPU") -> None:
    sm = {}
    if (out_dir / "llm_decoder.onnx").exists():
        sm["llm_decoder"] = {
            "filename": "llm_decoder.onnx", "runtime": "onnxruntime-genai",
            "genai_config": "genai_config.json",
            "io": {"inputs": ["inputs_embeds", "attention_mask", "past_key_values.*"],
                   "outputs": ["hidden_states", "present.*"]},
            "hidden_size": 2560, "num_layers": 36, "num_kv_heads": 8, "head_dim": 128,
            "vocab_size": 151936, "tie_word_embeddings": True,
            "note": "exclude_embeds+exclude_lm_head; text logits = hidden @ text_embedᵀ",
        }
    if (out_dir / "audio_embed.onnx").exists():
        sm["audio_embed"] = {"filename": "audio_embed.onnx",
            "io": {"inputs": ["codes[B,L,8] int64"], "outputs": ["audio_embeds[B,L,2560]"]},
            "num_codebooks": 8, "vocab_size": 1026,
            "note": "fused multi-codebook embedding; add to text embeds at audio positions"}
    if (out_dir / "audio_heads.onnx").exists():
        sm["audio_heads"] = {"filename": "audio_heads.onnx",
            "io": {"inputs": ["hidden_states[B,L,2560]"], "outputs": ["audio_logits[B,L,8,1026]"]},
            "num_codebooks": 8, "vocab_size": 1026,
            "note": "fused multi-codebook head (tied to audio_embed); apply delay pattern in gen loop"}
    if (out_dir / "audio_tokenizer.onnx").exists():
        sm["audio_tokenizer"] = {"filename": "audio_tokenizer.onnx",
            "io": {"inputs": ["audio_codes[B,8,T] int64"], "outputs": ["waveform[B,1,L] f32"]},
            "sample_rate": 24000, "num_codebooks": 8,
            "note": "Higgs v2 codec decoder (codes→waveform), fp32"}
    manifest = {"model_id": MODEL_NAME, "device": device, "precision": precision,
                "execution_provider": PROVIDER[device],
                "standalone_dir": "../../qwen3_standalone", "sub_models": sm}
    if device == "openvino":
        manifest["ov_device_type"] = ov_device   # CPU / GPU / NPU / AUTO
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote {out_dir / 'manifest.json'}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Optimize higgs-audio-v3-tts-4b → ONNX sub-parts")
    ap.add_argument("--device", choices=["cpu", "cuda","openvino"], default="cpu")
    ap.add_argument("--precision", choices=["int4", "fp16", "fp32"], default="int4")
    ap.add_argument("--ov-device", choices=["CPU", "GPU", "NPU", "AUTO"], default="CPU",
                    help="OpenVINO device_type when --device openvino (Intel CPU/iGPU/NPU/AUTO)")
    ap.add_argument("--model-id", default=MODEL_NAME)
    ap.add_argument("--model-path", default=MODEL_DIR)
    ap.add_argument("--components", nargs="*", default=ALL_COMPONENTS)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = output_root(args.device, args.precision)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device {args.device} | Precision {args.precision} | Output {out_dir}")
    print(f"Components: {args.components}\n")

    need_ckpt = any(c in args.components for c in ALL_COMPONENTS)
    if need_ckpt and not args.skip_download:
        print("=== Download checkpoint → ./model ==="); download_model(args.model_id, args.model_path); print()

    if "llm_decoder" in args.components:
        qwen3_abs = str(Path(STANDALONE_DIR).resolve())
        if not args.skip_extract:
            print("=== Extract standalone Qwen3 → ./qwen3_standalone ===")
            qwen3_abs = extract_llm(args.model_path); print()
        print("=== LLM via ModelBuilder ===")
        build_llm(qwen3_abs, out_dir, args.device, args.precision, args.ov_device); print()

    for comp in ("audio_embed", "audio_heads", "audio_tokenizer"):
        if comp in args.components:
            print(f"=== {comp} via Olive ===")
            build_audio_part(comp, args.model_path, out_dir, args.device, args.precision); print()

    print("=== Manifest ===")
    write_manifest(out_dir, args.device, args.precision, args.ov_device)
    print(f"\nDone → {out_dir}")
    if args.device == "openvino":
        print("NOTE: OpenVINO inference needs a matching onnxruntime-openvino build "
              "(replaces plain onnxruntime — not both). See SUPPORT.md.")


if __name__ == "__main__":
    main()
