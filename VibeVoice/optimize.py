"""VibeVoice family → ONNX sub-parts — unified builder for ALL checkpoints.

One entry point for every VibeVoice model. No download: point --model at a locally
downloaded checkpoint dir (defaults below). Each model is decomposed into standalone
ONNX sub-parts (LLM decoder via onnxruntime-genai ModelBuilder; audio / diffusion /
connector blocks via Olive), all wired from the loaders in user_script.py.

Models & components (all parity-verified vs PyTorch, cos ~1.0):
  1.5b     (VibeVoice-1.5B, TTS)      llm acoustic_encoder acoustic_decoder
                                       semantic_encoder diffusion_head
                                       acoustic_connector semantic_connector
  asr      (VibeVoice-ASR, 7B)        llm acoustic_encoder acoustic_decoder
                                       semantic_encoder acoustic_connector semantic_connector
  asr-hf   (VibeVoice-ASR-HF, 7B)     llm acoustic_encoder semantic_encoder multi_modal_projector
  realtime (Realtime-0.5B, streaming) llm acoustic_decoder diffusion_head acoustic_connector
  acoustic (VibeVoice-AcousticTokenizer)  acoustic_encoder acoustic_decoder  (standalone; no LLM)

Usage (model is the FINAL positional arg — a known key OR a path to a checkpoint dir):
  uv run optimize.py 1.5b                                  # all components, cpu int4
  uv run optimize.py --device cuda --precision fp16 asr-hf
  uv run optimize.py --components llm,acoustic_decoder realtime
  uv run optimize.py --device cuda --precision fp16 all
  uv run optimize.py --model D:/models/asr-hf asr-hf       # build from a checkpoint elsewhere
  uv run optimize.py --model D:/models/asr-hf              # same, type auto-detected
  uv run optimize.py ./asr-hf                              # a path → type auto-detected from config.json
  uv run optimize.py --list

Checkpoint resolution:
  keyword only          -> ./<dir> if present, else DOWNLOADED from HF_REPO[key] into ./<dir>
  keyword + --model P   -> build from P (no download); P's type must match the keyword
  --model P (no keyword)-> build from P, type auto-detected from P/config.json
  path as positional    -> same as --model P
A dir holding only config.json (no *.safetensors/*.bin) is rejected up front as incomplete
rather than failing later inside weight extraction.

Args:
  model                                     FINAL positional: {1.5b,asr,asr-hf,realtime,acoustic,all}
                                            or a filesystem path to a checkpoint dir (auto-detected)
  --model PATH                              build FROM this local transformers checkpoint dir
                                            (overrides the default ./<dir>; nothing is downloaded)
  --device {cpu,cuda}                       target device (default cpu)
  --precision {int4,fp16,fp32}              LLM quantization / audio dtype (default int4)
  --components a,b,c                        comma-separated subset (default: all for the model)
  --exclude-llm                             skip the LLM decoder build
  --repo ORG/NAME                           override the HF repo id to download from
  --output-dir PATH                         override output root (default onnx/{model}/{device}_{precision})

Precision note: --precision sets the LLM build (int4 / fp16 / fp32). Audio/diffusion/connector
blocks are VAE/DiT — int4 doesn't apply, so they export fp32 unless --precision fp16 (then fp16).
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Force UTF-8 stdout/stderr so emoji in dependency logs (Olive/onnxscript) don't crash a
# cp1252 Windows console — lets `uv run optimize.py ...` work without PYTHONIOENCODING set.
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

# LLM spec: (extract_fn, exclude_embeds, exclude_lm_head)
#   TTS backbones (1.5b, realtime): head is the diffusion head → exclude embeds + lm_head
#     (decoder consumes inputs_embeds, emits hidden_states).
#   ASR (asr, asr-hf): generate text → keep lm_head; audio+text fused upstream → exclude embeds.
# Olive spec per component: (loader, io_config, dummy, env-overrides)
MODELS = {
    "1.5b": {
        "dir": "model",
        "llm": ("extract_qwen2_standalone", True, True),
        "olive": {
            "acoustic_encoder": ("get_acoustic_encoder_model", "get_acoustic_encoder_io_config", "get_acoustic_encoder_dummy_inputs", {}),
            "acoustic_decoder": ("get_acoustic_decoder_model", "get_acoustic_decoder_io_config", "get_acoustic_decoder_dummy_inputs", {}),
            "semantic_encoder": ("get_semantic_tokenizer_encoder_model", "get_semantic_tokenizer_encoder_io_config", "get_semantic_tokenizer_encoder_dummy_inputs", {}),
            "diffusion_head": ("get_diffusion_head_model", "get_diffusion_head_io_config", "get_diffusion_head_dummy_inputs", {"VV_HEAD_HIDDEN": "1536"}),
            "acoustic_connector": ("get_acoustic_connector_model", "get_acoustic_connector_io_config", "get_acoustic_connector_dummy_inputs", {}),
            "semantic_connector": ("get_semantic_connector_model", "get_semantic_connector_io_config", "get_semantic_connector_dummy_inputs", {}),
        },
    },
    "asr": {
        "dir": "asr",
        "llm": ("extract_qwen2_asr", True, False),
        "olive": {
            "acoustic_encoder": ("get_acoustic_encoder_model", "get_acoustic_encoder_io_config", "get_acoustic_encoder_dummy_inputs", {}),
            "acoustic_decoder": ("get_acoustic_decoder_model", "get_acoustic_decoder_io_config", "get_acoustic_decoder_dummy_inputs", {}),
            "semantic_encoder": ("get_semantic_tokenizer_encoder_model", "get_semantic_tokenizer_encoder_io_config", "get_semantic_tokenizer_encoder_dummy_inputs", {}),
            "acoustic_connector": ("get_acoustic_connector_model", "get_acoustic_connector_io_config", "get_acoustic_connector_dummy_inputs", {}),
            "semantic_connector": ("get_semantic_connector_model", "get_semantic_connector_io_config", "get_semantic_connector_dummy_inputs", {}),
        },
    },
    "asr-hf": {
        "dir": "asr-hf",
        "llm": ("extract_qwen2_asrhf", True, False),
        "olive": {
            "acoustic_encoder": ("get_asrhf_acoustic_encoder_model", "get_asrhf_acoustic_encoder_io_config", "get_asrhf_acoustic_encoder_dummy_inputs", {}),
            "semantic_encoder": ("get_asrhf_semantic_encoder_model", "get_asrhf_semantic_encoder_io_config", "get_asrhf_semantic_encoder_dummy_inputs", {}),
            "multi_modal_projector": ("get_asrhf_projector_model", "get_asrhf_projector_io_config", "get_asrhf_projector_dummy_inputs", {}),
        },
    },
    "realtime": {
        "dir": "realtime",
        # NOTE: "llm" here is the 20-layer `tts_language_model` (the TTS backbone). The 4-layer
        # `language_model` TEXT encoder is a SEPARATE component ("text_lm" below) — without it the
        # driver feeds raw text embeds into the TTS backbone and the output is babble. See the
        # header comment on get_realtime_text_lm_model in user_script.py.
        "llm": ("extract_qwen2_realtime", True, True),
        "olive": {
            "text_lm": ("get_realtime_text_lm_model", "get_realtime_text_lm_io_config", "get_realtime_text_lm_dummy_inputs", {}),
            "acoustic_decoder": ("get_realtime_acoustic_decoder_model", "get_realtime_acoustic_decoder_io_config", "get_realtime_acoustic_decoder_dummy_inputs", {}),
            "diffusion_head": ("get_diffusion_head_model", "get_diffusion_head_io_config", "get_diffusion_head_dummy_inputs", {"VV_HEAD_HIDDEN": "896"}),
            "acoustic_connector": ("get_acoustic_connector_model", "get_acoustic_connector_io_config", "get_acoustic_connector_dummy_inputs", {}),
        },
    },
    # standalone Acoustic Tokenizer (encoder + decoder VAE) — no LLM. transformers-native loaders.
    "acoustic": {
        "dir": "acoustic",
        "llm": None,
        "olive": {
            "acoustic_encoder": ("get_acoustic_std_encoder_model", "get_acoustic_encoder_io_config", "get_acoustic_encoder_dummy_inputs", {}),
            "acoustic_decoder": ("get_acoustic_std_decoder_model", "get_acoustic_decoder_io_config", "get_acoustic_decoder_dummy_inputs", {}),
        },
    },
}

# HuggingFace repo per key — used to auto-download when the checkpoint dir is absent.
# Override any of these with --repo when the id differs.
HF_REPO = {
    "1.5b": "microsoft/VibeVoice-1.5B",
    "asr": "microsoft/VibeVoice-ASR",
    "asr-hf": "microsoft/VibeVoice-ASR-HF",
    "realtime": "microsoft/VibeVoice-Realtime-0.5B",
    "acoustic": "microsoft/VibeVoice-AcousticTokenizer",
}

# device → (olive device string, execution provider, genai device string)
DEVICES = {"cpu": ("cpu", "CPUExecutionProvider", "cpu"),
           "gpu": ("gpu", "CUDAExecutionProvider", "cuda"),
           "cuda": ("gpu", "CUDAExecutionProvider", "cuda")}


def ensure_checkpoint(key, repo_override=None):
    """Return the local checkpoint dir for `key`, downloading from HF if it isn't present."""
    src = (HERE / MODELS[key]["dir"]).resolve()
    if (src / "config.json").exists():
        # config.json alone is NOT enough: a partial/stub dir (config only, no weight shards)
        # used to pass this check and then blow up much later inside weight extraction with a
        # confusing FileNotFoundError on model-0000N-of-....safetensors. Require real weights.
        if not (list(src.glob("*.safetensors")) or list(src.glob("*.bin"))):
            sys.exit(
                f"[{key}] {src} has config.json but NO weight files (*.safetensors / *.bin) — "
                f"it's an incomplete checkpoint dir.\n"
                f"  Either delete it and re-run (to download {repo_override or HF_REPO.get(key)}), "
                f"or pass the path to a complete checkpoint instead, e.g.:\n"
                f"    uv run optimize.py --device cpu --precision fp16 /path/to/{key}"
            )
        return src
    repo = repo_override or HF_REPO.get(key)
    if not repo:
        sys.exit(f"[{key}] no checkpoint at {src} and no HF repo known — pass --repo <org/name>")
    print(f"[{key}] checkpoint not found at {src}; downloading {repo} ...")
    from huggingface_hub import snapshot_download
    src.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo, local_dir=str(src))
    print(f"[{key}] downloaded -> {src}")
    return src

VERBOSE = False    # set by --verbose: lowers Olive log severity + prints extra detail


def _log(*a):
    if VERBOSE:
        print(*a)


def all_components(model: str):
    llm = ["llm"] if MODELS[model].get("llm") else []   # some checkpoints (acoustic) have no LLM
    return llm + list(MODELS[model]["olive"].keys())


def detect_model_type(model_src: Path) -> str:
    """Infer the registry key from a checkpoint's config.json (model_type / architectures)."""
    cfg = json.loads((model_src / "config.json").read_text())
    mt = cfg.get("model_type"); arch = (cfg.get("architectures") or [""])[0]
    if mt == "vibevoice_streaming":
        return "realtime"
    if mt == "vibevoice_asr":
        return "asr-hf"
    if arch == "VibeVoiceForASRTraining":
        return "asr"
    if mt == "vibevoice_acoustic_tokenizer":
        return "acoustic"
    if mt == "vibevoice":
        return "1.5b"
    raise SystemExit(f"cannot auto-detect VibeVoice type at {model_src} "
                     f"(model_type={mt!r}, arch={arch!r})")


def resolve_target(arg: str):
    """Return (model_key, model_src) from a positional that is a registry key OR a path."""
    if arg in MODELS:
        return arg, (HERE / MODELS[arg]["dir"]).resolve()
    src = Path(arg).expanduser().resolve()
    if not (src / "config.json").exists():
        raise SystemExit(f"'{arg}' is not a known model key {list(MODELS)+['all']} "
                         f"nor a checkpoint dir with config.json")
    return detect_model_type(src), src


# Input token-embedding weight per model (the [vocab, hidden] lookup table). The genai decoders are
# built exclude_embeds=True (input = inputs_embeds), so this CONSTANT weight is shipped next to the
# ONNX to make the package self-contained — the driver turns token ids into vectors from this file
# (else it must read the multi-GB source checkpoint). Mirrors common.EMBED_KEY.
EMBED_WEIGHT_KEY = {
    "1.5b": "model.language_model.embed_tokens.weight",
    "asr": "model.language_model.embed_tokens.weight",
    "asr-hf": "language_model.model.embed_tokens.weight",
    "realtime": "model.tts_language_model.embed_tokens.weight",
}


def extract_embed_matrix(model: str, model_src: Path, target: Path, precision: str):
    """Export {target}/embeddings.onnx : input_ids[int64, B×S] -> inputs_embeds[float, B×S×H] via a
    Gather on the input token-embedding table. The genai decoder is built exclude_embeds=True, so this
    small companion graph provides the missing embed step and makes the package self-contained (no
    source checkpoint at inference — the genai-native embeddings.onnx pattern). fp16 for fp16 builds,
    else fp32; weights go to embeddings.onnx.data (external). Constant weight — a one-time snapshot."""
    import glob
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from safetensors import safe_open
    key = EMBED_WEIGHT_KEY.get(model)
    if key is None:
        print(f"  [embed] no embed key for '{model}' — skip"); return
    # Always fp16: a fp32 [vocab,hidden] table is >2 GB and trips the protobuf serialize limit, and the
    # decoder's OnnxLLM driver upcasts inputs_embeds to the graph dtype anyway (so fp32/int4 decoders
    # still receive fp32 input). Embedding values sit well within fp16 range — negligible loss.
    np_dt, tp = np.float16, TensorProto.FLOAT16
    W = None
    for sf in glob.glob(str(Path(model_src) / "*.safetensors")):
        with safe_open(sf, "pt") as f:
            if key in f.keys():
                W = f.get_tensor(key).float().numpy().astype(np_dt); break
    if W is None:
        print(f"  [embed][warn] {key} not found in {model_src} — package will need the source checkpoint"); return
    V, H = W.shape
    g = helper.make_graph(
        [helper.make_node("Gather", ["embed_weight", "input_ids"], ["inputs_embeds"], axis=0)],
        "embeddings",
        inputs=[helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])],
        outputs=[helper.make_tensor_value_info("inputs_embeds", tp, ["batch", "seq", H])],
        initializer=[numpy_helper.from_array(W, name="embed_weight")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    out = Path(target) / "embeddings.onnx"
    onnx.save(m, str(out), save_as_external_data=True, location="embeddings.onnx.data",
              size_threshold=1024, all_tensors_to_one_file=True)
    print(f"  [embed] embeddings.onnx {tuple(W.shape)} {np_dt.__name__} -> {target}")


def build_llm(model: str, model_src: Path, target: Path, precision: str, device: str):
    """Extract a standalone Qwen2 dir, then ModelBuilder → llm_decoder.onnx at --precision."""
    import user_script
    from onnxruntime_genai.models.builder import create_model

    extract_name, excl_embeds, excl_head = MODELS[model]["llm"]
    extract = getattr(user_script, extract_name)
    _log(f"    [llm] extract={extract_name} -> standalone Qwen2 dir")
    qwen2 = extract(str(model_src), str(HERE))
    target.mkdir(parents=True, exist_ok=True)
    genai_device = DEVICES[device][2]
    # Unique cache dir PER INVOCATION (not a shared `.mb_cache_{model}`). onnxruntime-genai's
    # save_model() ends with `if not os.listdir(cache): os.rmdir(cache)` — so a shared cache races
    # when two optimize.py runs build the same model concurrently (e.g. int4 + fp16, or cpu + cuda):
    # the first run rmdir's it, the second run's os.listdir(cache) then raises FileNotFoundError.
    # tempfile.mkdtemp is atomic+unique, so concurrent builds never collide; we clean up defensively
    # afterwards (genai already removes it when left empty).
    import tempfile, shutil
    cache = Path(tempfile.mkdtemp(prefix=f".mb_cache_{model}_{device}_{precision}_", dir=str(HERE)))
    print(f"  ModelBuilder: llm_decoder {precision} on {genai_device} "
          f"(exclude_embeds={excl_embeds}, exclude_lm_head={excl_head}) -> {target}/llm_decoder.onnx")
    try:
        create_model("", qwen2, str(target), precision, genai_device, cache_dir=str(cache),
                     filename="llm_decoder.onnx", exclude_embeds=excl_embeds, exclude_lm_head=excl_head)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    print("  llm_decoder done.")
    extract_embed_matrix(model, model_src, target, precision)   # ship the embedding table alongside


def build_olive(model: str, component: str, model_src: Path, target: Path, precision: str, device: str):
    """Olive export (OnnxConversion dynamo; +OnnxFloatToFloat16 when precision==fp16)."""
    import os
    from olive import run

    loader, io_cfg, dummy, env = MODELS[model]["olive"][component]
    os.environ.update(env)
    olive_dev, ep, _ = DEVICES[device]
    tmp = (target / f"_{component}_tmp").resolve()
    passes = {"c": {"type": "OnnxConversion", "use_dynamo_exporter": True}}
    if precision == "fp16":
        # keep shape/resample ops in fp32 — converting them yields invalid graphs (e.g. the acoustic
        # tokenizer's ConstantOfShape emitted fp16 where a float32 consumer expects it) and they gain
        # nothing from fp16. Same class of block-list the Higgs encoders needed.
        passes["f16"] = {"type": "OnnxFloatToFloat16",
                         "op_block_list": ["ConstantOfShape", "ConvTranspose", "Resize", "Range"]}
    cfg = {
        "input_model": {"type": "PyTorchModel", "model_path": str(model_src),
                        "model_loader": loader, "model_script": str(HERE / "user_script.py"),
                        "io_config": io_cfg, "dummy_inputs_func": dummy},
        "systems": {"local": {"type": "LocalSystem",
                    "accelerators": [{"device": olive_dev, "execution_providers": [ep]}]}},
        "passes": passes, "target": "local", "no_artifacts": True,
        "log_severity_level": 1 if VERBOSE else 3,
        "output_dir": str(tmp), "cache_dir": str((target / f"_ocache_{component}").resolve()),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(HERE)) as f:
        json.dump(cfg, f, indent=2); cfgp = f.name
    dtype = "fp16" if precision == "fp16" else "fp32"
    print(f"  Olive export {component} ({dtype}, {ep}) -> {target}/{component}.onnx")
    _log(f"    [olive] loader={loader} passes={list(passes)} env={env or '{}'}")
    try:
        run(cfgp)
    finally:
        Path(cfgp).unlink(missing_ok=True)
    for src in tmp.glob("model.onnx*"):
        dst = target / src.name.replace("model.onnx", f"{component}.onnx")
        if dst.exists(): dst.unlink()
        shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(target / f"_ocache_{component}", ignore_errors=True)
    print(f"  {component} done.")


def build_model(model: str, model_src: Path, device: str, precision: str, components, output_dir,
                exclude_llm=False):
    spec = MODELS[model]
    if not (model_src / "config.json").exists():
        sys.exit(f"[{model}] no checkpoint at {model_src} (download it first)")
    # Output layout: onnx/{model}/{device}_{precision}
    target = Path(output_dir) if output_dir else (HERE / "onnx" / model / f"{device}_{precision}")
    target.mkdir(parents=True, exist_ok=True)

    comps = [c.strip() for c in components.split(",")] if components else all_components(model)
    unknown = [c for c in comps if c != "llm" and c not in spec["olive"]]
    if unknown:
        sys.exit(f"[{model}] unknown components {unknown}; valid: {all_components(model)}")
    if exclude_llm and "llm" in comps:
        comps = [c for c in comps if c != "llm"]
        print(f"[{model}] --exclude-llm: skipping the LLM decoder build")

    # int4 (MatMulNBits) applies ONLY to the genai-built `llm` decoder. Every other component goes
    # through the Olive path, which emits fp32/fp16 only — conv-VAE blocks have no quantizable
    # MatMuls, and small transformer components like realtime's `text_lm` have no int4 path here
    # either. build_olive already treats int4 as fp32, so there is nothing to downgrade.
    #
    # Do NOT retarget to {device}_fp32. These components must still land in {device}_int4, because
    # that is the directory the driver loads. This mirrors embeddings.onnx, which is always written
    # fp16 no matter what the target dir is named, and it is why a full int4 build legitimately
    # contains fp32 audio/diffusion blocks (see README: int4 is larger than fp16 overall).
    #
    # This used to redirect the output dir, which only ever fired on PARTIAL builds that exclude the
    # llm — silently diverting e.g. `--precision int4 --components text_lm` into the fp32 dir (and
    # overwriting the fp32 copy) while leaving the int4 dir missing the file entirely.
    if precision == "int4" and "llm" not in comps:
        print(f"[{model}] note: int4 quantizes the `llm` decoder only; {comps} export as fp32 "
              f"into {target.name} — this is expected, not a mislabelled build.")

    print(f"\n=== {model}  src={model_src}  device={device}  precision={precision} ===")
    print(f"    target={target}  components={comps}")
    for comp in comps:
        if comp == "llm":
            build_llm(model, model_src, target, precision, device)
        else:
            build_olive(model, comp, model_src, target, precision, device)
    if model == "realtime" and not {"text_lm", "tts_lm"}.isdisjoint(comps):
        # Ship two tiny weight sets the driver needs but that don't justify their own ONNX graphs:
        #   type_embed.npy — the [2, H] tts_input_types lookup, to tag each TTS-LM step as
        #                    text(1)/speech(0).
        #   eos_head.npz   — the tts_eos_classifier, the model's learned stop signal. Without it
        #                    generation runs to --max-frames and everything past the end of the
        #                    text is drift.
        import user_script as _us
        _us.export_realtime_extras(model_src, target)
    print(f"[{model}] done -> {target}")


def main():
    ap = argparse.ArgumentParser(description="VibeVoice family -> ONNX (unified)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    ap.add_argument("--precision", choices=["int4", "fp16", "fp32"], default="int4")
    ap.add_argument("--components", help="comma-separated subset, e.g. llm,acoustic_decoder "
                                         "(default: all for the model)")
    ap.add_argument("--exclude-llm", action="store_true",
                    help="skip the LLM decoder build (e.g. the RAM-bound 7B ASR decoders)")
    ap.add_argument("--repo", help="override the HF repo id to download the checkpoint from")
    ap.add_argument("--model", dest="model_path", metavar="PATH",
                    help="path to an already-downloaded transformers checkpoint dir to build FROM "
                         "(overrides the default ./<dir> location; nothing is downloaded). "
                         "e.g. --model D:/models/asr-hf")
    ap.add_argument("--output-dir", help="override output dir (default onnx/{model}/{device}_{precision})")
    ap.add_argument("--verbose", "-v", action="store_true", help="verbose logs (Olive + extra detail)")
    ap.add_argument("--list", action="store_true", help="list models + components and exit")
    ap.add_argument("model", nargs="?",
                    help="FINAL positional: a keyword {1.5b,asr,asr-hf,realtime,acoustic,all} — picks "
                         "which model to build. Its checkpoint is used from ./<dir> if present, else "
                         "downloaded from HF. Use --model PATH to build from a checkpoint elsewhere. "
                         "A path may also be passed directly here (type auto-detected from config.json).")
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    device = "cuda" if args.device == "gpu" else args.device

    if args.list or not args.model:
        print("VibeVoice models & components:")
        for m in MODELS:
            print(f"  {m:9s} (dir '{MODELS[m]['dir']}', repo '{HF_REPO.get(m, 'local-only')}'): {' '.join(all_components(m))}")
        print("\nDevices: cpu cuda   Precisions: int4 fp16 fp32")
        print("Positional 'model': a keyword above or 'all'. Output -> onnx/{model}/{device}_{precision}")
        return

    # --model PATH: build from an already-downloaded checkpoint anywhere on disk. The positional
    # keyword still selects WHICH model (so the right extractor/loaders are used); if it's omitted
    # the type is auto-detected from the checkpoint's config.json.
    if args.model_path:
        src = Path(args.model_path).expanduser().resolve()
        if not (src / "config.json").exists():
            ap.error(f"--model {src} has no config.json — not a transformers checkpoint dir")
        if not (list(src.glob("*.safetensors")) or list(src.glob("*.bin"))):
            ap.error(f"--model {src} has config.json but NO weight files (*.safetensors / *.bin) "
                     f"— it's an incomplete/partial checkpoint")
        if args.model in (None, "all"):
            key = detect_model_type(src)
            print(f"[{key}] type auto-detected from {src}/config.json")
        elif args.model in MODELS:
            key = args.model
            detected = detect_model_type(src)
            if detected != key:                    # loaders are per-architecture — a mismatch here
                ap.error(f"--model {src} looks like '{detected}' but you asked for '{key}'; "
                         f"pass the matching keyword (or omit it to auto-detect)")
        else:
            ap.error(f"unknown model '{args.model}'; choose from {list(MODELS) + ['all']}")
        print(f"[{key}] building from {src} (no download)")
        build_model(key, src, device, args.precision, args.components, args.output_dir,
                    exclude_llm=args.exclude_llm)
        return

    if args.model == "all":
        if args.components:
            ap.error("--components cannot be combined with 'all'")
        if args.repo:
            ap.error("--repo cannot be combined with 'all'")
        keys = list(MODELS)
    elif args.model in MODELS:
        keys = [args.model]
    elif Path(args.model).expanduser().is_dir():
        # A filesystem path passed as the positional (documented in the module docstring, but the
        # dispatch below used to only accept registry keys — so paths silently errored out).
        # Equivalent to --model PATH; type is auto-detected and nothing is downloaded.
        key, src = resolve_target(args.model)
        print(f"[{key}] using checkpoint at {src} (auto-detected from config.json)")
        build_model(key, src, device, args.precision, args.components, args.output_dir,
                    exclude_llm=args.exclude_llm)
        return
    else:
        ap.error(f"unknown model '{args.model}'; choose from {list(MODELS) + ['all']} "
                 f"or pass a path to a checkpoint dir containing config.json")

    for m in keys:
        src = ensure_checkpoint(m, args.repo)      # download if the local dir is missing
        build_model(m, src, device, args.precision, args.components, args.output_dir,
                    exclude_llm=args.exclude_llm)


if __name__ == "__main__":
    main()
