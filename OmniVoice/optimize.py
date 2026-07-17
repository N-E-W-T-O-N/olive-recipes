"""End-to-end ONNX optimization pipeline for Prince-1/OmniVoice.

OmniVoice backbone consists of three ONNX sub-models:

  audio_embeddings_encoder  — text + audio embedding fusion
  llm_decoder               — Qwen3-28 layer backbone (inputs_embeds → hidden_states)
  audio_heads_decoder       — linear projection to per-codebook audio logits

The Higgs Audio V2 Tokenizer (acoustic/semantic encoder, quantizer, decoder)
requires the `boson-multimodal` package.  Export it separately:
  cd <omnivoice_model_dir>
  python convert_omnivoice_to_onnx.py --only higgs --out-dir ./higgs_onnx

Pipeline dataflow (32 iterative unmasking steps):
  (input_ids, audio_mask) → audio_embeddings_encoder → inputs_embeds
  (inputs_embeds, mask)   → llm_decoder              → hidden_states
  hidden_states           → audio_heads_decoder       → logits (B,8,S,1025)

Usage:
  python optimize.py --device cpu                  # CPU INT4 backbone (cpu_and_mobile/)
  python optimize.py --device cpu_fp16             # CPU FP16 backbone (cpu_fp16/)
  python optimize.py --device gpu                  # CUDA FP16 backbone (cuda/)
  python optimize.py --include-higgs               # also export Higgs Audio Tokenizer
  python optimize.py --higgs-only                  # export only Higgs tokenizer
  python optimize.py --skip-export                 # regenerate configs only
  python optimize.py --skip-llm                    # skip ModelBuilder step (slow)

Profiles:
  cpu       → INT4 weights for all sub-models.  Smallest footprint.
  cpu_fp16  → FP16 weights for all sub-models.  Better accuracy than INT4 on CPUs
               with AVX-512 FP16 support (Intel Sapphire Rapids+, AMD Zen5+).
               LLM uses ModelBuilder fp16; audio sub-models use OnnxFloatToFloat16.
  gpu       → FP16 audio sub-models + INT4 LLM via CUDAExecutionProvider.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.getLogger("onnxscript").setLevel(logging.WARNING)
logging.getLogger("onnx_ir").setLevel(logging.WARNING)

MODEL_NAME = "Prince-1/OmniVoice"
MODELS_DIR = "models"
HIDDEN_SIZE   = 1024
NUM_CODEBOOKS = 8
AUDIO_VOCAB   = 1025


# =============================================================================
# Step 0: Save Qwen3 standalone (required before ModelBuilder runs)
# =============================================================================

def prepare_qwen3_standalone(model_path: str, work_dir: str) -> str:
    """Save OmniVoice's internal Qwen3 LLM as a standalone HF directory.

    ModelBuilder (onnxruntime-genai create_model) needs a standard HF model
    directory with architectures=["Qwen3ForCausalLM"] to export the LLM.
    Returns the path to the saved directory.
    """
    qwen3_dir = str(Path(work_dir) / "qwen3_standalone")
    if Path(qwen3_dir).exists() and (Path(qwen3_dir) / "model.safetensors").exists():
        print(f"  Reusing existing {qwen3_dir}")
        return qwen3_dir

    print(f"  Saving Qwen3 standalone to {qwen3_dir} ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from user_script import save_qwen3_standalone
    return save_qwen3_standalone(model_path, work_dir)


# =============================================================================
# Step 1: Olive Export + Optimization
# =============================================================================

# Audio backbone sub-models — Olive config defined INLINE in Python (no external JSON).
# Each: PyTorchModel (loaded via user_script) → OnnxConversion → ORT-opt → peephole → fp16.
_USER_SCRIPT = str(Path(__file__).parent / "user_script.py")
AUDIO_SUBMODELS = (
    # stem,                        loader / io / dummy prefix,   fp16 saves external data?
    ("audio_embeddings_encoder",   "audio_embeddings",           True),
    ("audio_heads_decoder",        "audio_heads",                False),
)


def _audio_run_config(stem: str, prefix: str, model_path: str, models_dir: str,
                      precision: str, external_data: bool) -> dict:
    """Build the Olive RunConfig dict for one audio sub-model (replaces the old *.json).
    precision 'int4' → block-wise RTN int4 quantization; 'fp16' → float16 cast."""
    if precision == "int4":
        quant = {"type": "OnnxBlockWiseRtnQuantization", "block_size": 128,
                 "is_symmetric": True, "accuracy_level": 4, "save_as_external_data": external_data}
    else:
        quant = {"type": "OnnxFloatToFloat16", "save_as_external_data": external_data}
    if external_data:
        quant["external_data_name"] = f"{stem}.onnx.data"
    return {
        "input_model": {
            "type": "PyTorchModel",
            "model_path": str(Path(model_path).resolve()),
            "model_script": _USER_SCRIPT,
            "model_loader": f"get_{prefix}_model",
            "io_config": f"get_{prefix}_io_config",
            "dummy_inputs_func": f"get_{prefix}_dummy_inputs",
        },
        "passes": {
            "convert": {"type": "OnnxConversion", "use_dynamo_exporter": False},
            "ort": {"type": "OrtTransformersOptimization", "model_type": "",
                    "opt_level": 1, "only_onnxruntime": True},
            "cast": {"type": "OnnxPeepholeOptimizer", "onnxscript_optimize": False,
                     "onnxoptimizer_optimize": False, "fuse_reshape_operations": False,
                     "fix_com_microsoft_opset": True, "cast_chain_elimination": True},
            "quant": quant,
        },
        "no_artifacts": True,
        "output_dir": str(Path(models_dir) / f"{stem}.onnx"),
    }


def export_audio_models(model_path: str, models_dir: str, precision: str):
    """Run Olive on the two audio backbone sub-models (PyTorchModel → ONNX → int4/fp16),
    with the RunConfig built in Python — no external JSON files."""
    from olive.workflows import run as olive_run
    print(f"=== Running Olive pipelines (inline Python configs, audio={precision}) ===")
    for stem, prefix, external_data in AUDIO_SUBMODELS:
        print(f"  Building {stem}.onnx ...")
        olive_run(_audio_run_config(stem, prefix, model_path, models_dir, precision, external_data))


def export_llm(qwen3_dir: str, models_dir: str, source_model: str, execution_provider: str = "cpu"):
    """Export the Qwen3 LLM via onnxruntime-genai ModelBuilder `create_model` DIRECTLY
    (bypassing Olive, whose ModelBuilder pass can't emit FP16 on CPU).

    genai's valid precision×EP combos on CPU are only FP32 and INT4 — there is no FP16 CPU
    build — so on CPU the LLM is exported INT4 (the audio sub-models remain fp16). GPU uses FP16.
    exclude_embeds/exclude_lm_head give the `inputs_embeds → hidden_states` decoder the pipeline
    needs (the embeddings come from audio_embeddings_encoder, the heads from audio_heads_decoder)."""
    import tempfile, shutil
    from onnxruntime_genai.models.builder import create_model

    precision = "fp16" if execution_provider in ("cuda", "dml") else "int4"
    tmp = tempfile.mkdtemp()
    cache = tempfile.mkdtemp()
    print(f"  LLM via genai create_model (precision={precision}, ep={execution_provider}) ...")
    create_model(
        "", str(qwen3_dir), tmp, precision, execution_provider, cache,
        exclude_embeds=True, exclude_lm_head=True, filename="llm_decoder.onnx",
    )
    out = Path(models_dir)
    for suf in ("", ".data"):
        s = Path(tmp) / f"llm_decoder.onnx{suf}"
        if s.exists():
            d = out / f"llm_decoder.onnx{suf}"
            if d.exists():
                d.unlink()
            shutil.move(str(s), str(d))
    # genai_config.json describes the exported LLM for the genai runtime — keep it if produced
    genai_cfg = Path(tmp) / "genai_config.json"
    if genai_cfg.exists():
        shutil.copy(genai_cfg, out / "genai_config.json")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)

    # Tokenizer + chat template for the LLM come from the source OmniVoice checkpoint (Qwen3 tokenizer)
    src = Path(source_model)
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "vocab.json", "merges.txt", "tokenizer.model", "chat_template.jinja"):
        if (src / fn).exists():
            shutil.copy(src / fn, out / fn)
    print(f"  Saved LLM ({precision}) → {out / 'llm_decoder.onnx'}  (+ tokenizer & chat_template)")


# =============================================================================
# Step 1b: Higgs Audio Tokenizer export
# =============================================================================

# Higgs Audio V2 Tokenizer sub-models — Olive config defined INLINE in Python (no external JSON).
# INT4 is too lossy for the DAC encoder/decoder, so Higgs is exported FP16 (or FP32). The two DAC
# models (acoustic_encoder, higgs_decoder) are pre-traced inside user_script.py before Olive sees
# them, resolving Python control-flow branches that would otherwise crash torch.onnx.export.
HIGGS_SUBMODELS = (
    # stem,               loader/io/dummy prefix,  fp16 op_block_list,           fp16 external data?
    ("acoustic_encoder",  "higgs_acoustic",        ["ConvTranspose", "Resize"],  True),
    ("semantic_encoder",  "higgs_semantic",        ["LayerNormalization"],       True),
    ("quantizer_encoder", "higgs_quantizer",       None,                          False),
    ("higgs_decoder",     "higgs_decoder",         ["ConvTranspose", "Resize"],  True),
)


def _higgs_run_config(stem, prefix, model_path, out_dir, precision, block_list, external_data) -> dict:
    """Build the Olive RunConfig dict for one Higgs sub-model (replaces the old higgs/*.json)."""
    torch_dtype = "float16" if precision == "fp16" else "float32"
    passes = {
        "convert": {"type": "OnnxConversion", "use_dynamo_exporter": False,
                    "torch_dtype": torch_dtype, "target_opset": 20},
        "ort": {"type": "OrtTransformersOptimization", "model_type": "",
                "opt_level": 1, "only_onnxruntime": True},
        "cast": {"type": "OnnxPeepholeOptimizer", "onnxscript_optimize": False,
                 "onnxoptimizer_optimize": False, "fuse_reshape_operations": False,
                 "fix_com_microsoft_opset": True, "cast_chain_elimination": True},
    }
    if precision == "fp16":
        fp16 = {"type": "OnnxFloatToFloat16", "save_as_external_data": external_data}
        if block_list:
            fp16["op_block_list"] = block_list
        if external_data:
            fp16["external_data_name"] = f"{stem}.onnx.data"
        passes["fp16"] = fp16
    return {
        "input_model": {
            "type": "PyTorchModel",
            "model_path": str(Path(model_path).resolve()),
            "model_script": _USER_SCRIPT,
            "model_loader": f"get_{prefix}_model",
            "io_config": f"get_{prefix}_io_config",
            "dummy_inputs_func": f"get_{prefix}_dummy_inputs",
        },
        "passes": passes,
        "no_artifacts": True,
        "output_dir": str(Path(out_dir) / f"{stem}.onnx"),
    }


def export_higgs(model_path: str, out_dir: str, precision: str = "fp16"):
    """Export all 4 Higgs Audio V2 Tokenizer sub-models via inline Python configs (no JSON)."""
    from olive.workflows import run as olive_run
    print(f"=== Running Higgs Audio Tokenizer pipelines (inline Python configs, {precision}) → {out_dir} ===")
    for stem, prefix, block_list, external_data in HIGGS_SUBMODELS:
        print(f"  Building {stem}.onnx ...")
        olive_run(_higgs_run_config(stem, prefix, model_path, out_dir, precision, block_list, external_data))


# =============================================================================
# Step 2: Generate inference manifest
# =============================================================================

def write_inference_manifest(output_dir: str, device: str = "cpu"):
    """Write omnivoice_manifest.json describing all ONNX sub-model paths.

    The inference script reads this to locate each sub-model at runtime.
    This is analogous to the genai_config.json used in VL pipelines.
    """
    manifest_path = Path(output_dir) / "omnivoice_manifest.json"

    if device == "gpu":
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"   # both cpu and cpu_fp16 use CPU EP

    precision_map = {"cpu": "int4", "cpu_fp16": "fp16", "gpu": "fp16+int4_llm"}

    manifest = {
        "model_id": MODEL_NAME,
        "execution_provider": provider,
        "precision": precision_map.get(device, "int4"),
        "backbone": {
            "audio_embeddings_encoder": {
                "filename": "audio_embeddings_encoder.onnx",
                "description": "Fuses text + audio codec token embeddings → inputs_embeds",
                "inputs": {
                    "input_ids":  "int64 (batch, num_codebooks=8, seq)",
                    "audio_mask": "bool  (batch, seq)"
                },
                "outputs": {
                    "inputs_embeds": "float32 (batch, seq, hidden=1024)"
                },
            },
            "llm_decoder": {
                "filename": "llm_decoder.onnx",
                "description": "Qwen3 28-layer backbone (inputs_embeds → hidden_states)",
                "note": "Exported with exclude_embeds=True, exclude_lm_head=True",
                "inputs": {
                    "inputs_embeds":  "float32 (batch, seq, hidden=1024)",
                    "attention_mask": "int64   (batch, seq)",
                    "position_ids":   "int64   (batch, seq)",
                    "past_key_values": "float32 per layer — pass empty (shape [B,heads,0,head_dim]) for full-sequence forward"
                },
                "outputs": {
                    "hidden_states": "float32 (batch, seq, hidden=1024)"
                },
                "hidden_size":       1024,
                "num_layers":        28,
                "num_attn_heads":    16,
                "num_kv_heads":       8,
                "head_dim":          128,
            },
            "audio_heads_decoder": {
                "filename": "audio_heads_decoder.onnx",
                "description": "Projects hidden_states to per-codebook audio-token logits",
                "inputs": {
                    "hidden_states": "float32 (batch, seq, hidden=1024)"
                },
                "outputs": {
                    "logits": "float32 (batch, num_codebooks=8, seq, audio_vocab=1025)"
                },
                "num_codebooks":   8,
                "audio_vocab_size": 1025,
                "audio_mask_id":   1024,
            },
        },
        "higgs_tokenizer": {
            "note": "Exported via optimize.py --include-higgs (or --higgs-only).",
            "models_dir": "higgs/models/",
            "sub_models": {
                "acoustic_encoder":  "acoustic_encoder.onnx",
                "semantic_encoder":  "semantic_encoder.onnx",
                "quantizer_encoder": "quantizer_encoder.onnx",
                "higgs_decoder":     "higgs_decoder.onnx",
            },
            "sample_rate_acoustic": 24000,
            "sample_rate_semantic": 16000,
            "downsample_factor": 320,
            "num_codebooks": 8,
            "codebook_size": 1024,
            "pipeline": [
                "acoustic_encoder(waveform_24k) → acoustic_features",
                "semantic_encoder(waveform_16k) → semantic_features",
                "quantizer_encoder(acoustic_features, semantic_features) → codes",
                "  --- TTS inference: codes → audio_embeddings_encoder → ... → audio_codes ---",
                "higgs_decoder(audio_codes) → waveform_24k",
            ],
            "loading_priority": [
                "transformers.HiggsAudioV2TokenizerModel (transformers>=5.3.0)",
                "boson_multimodal.load_higgs_audio_tokenizer (pip install boson-multimodal @ git+...)",
            ],
        },
        "iterative_decoding": {
            "note": "OmniVoice uses 32-step iterative unmasking (non-autoregressive).",
            "steps": 32,
            "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
            "per_step_pipeline": [
                "audio_embeddings_encoder(input_ids, audio_mask) → inputs_embeds",
                "llm_decoder(inputs_embeds, attention_mask, position_ids) → hidden_states",
                "audio_heads_decoder(hidden_states) → logits",
                "sample audio tokens from logits using codebook weights",
                "unmask predicted positions in input_ids",
            ],
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {manifest_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimize Prince-1/OmniVoice backbone to ONNX"
    )
    parser.add_argument("--device", choices=["cpu", "cpu_fp16", "gpu"], default="cpu",
                        help="Target device/precision (default: cpu)\n"
                             "  cpu      → INT4 weights, CPUExecutionProvider (smallest, slowest)\n"
                             "  cpu_fp16 → FP16 weights, CPUExecutionProvider (balanced; best on AVX-512 FP16 CPUs)\n"
                             "  gpu      → FP16 audio + INT4 LLM, CUDAExecutionProvider (fastest)")
    parser.add_argument("--config-dir", default=None,
                        help="Directory with Olive JSON configs (default: auto from --device)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip Olive export (models already exist)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM (ModelBuilder) export — it takes several minutes")
    parser.add_argument("--include-higgs", action="store_true",
                        help="Also export Higgs Audio V2 Tokenizer (4 sub-models in higgs/)")
    parser.add_argument("--higgs-only", action="store_true",
                        help="Export ONLY the Higgs tokenizer; skip backbone")
    parser.add_argument("--higgs-precision", choices=["fp16", "fp32"], default="fp16",
                        help="Precision for the Higgs tokenizer sub-models (default: fp16)")
    parser.add_argument("--output", default=None,
                        help="Models output directory")
    parser.add_argument("--model", default=MODEL_NAME,
                        help=f"HF model ID or local path (default: {MODEL_NAME})")
    args = parser.parse_args()

    _default_dirs = {"cpu": "cpu_and_mobile", "cpu_fp16": "cpu_fp16", "gpu": "cuda"}
    config_dir = args.config_dir or _default_dirs[args.device]
    models_dir = args.output or str(Path(config_dir) / MODELS_DIR)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    run_backbone = not args.higgs_only
    run_higgs    = args.include_higgs or args.higgs_only

    print(f"Target device : {args.device}")
    print(f"Config dir    : {config_dir}")
    print(f"Models dir    : {models_dir}")
    print(f"Model         : {args.model}")
    print(f"Backbone      : {'yes' if run_backbone else 'no (--higgs-only)'}")
    print(f"Higgs tokenizer: {(args.higgs_precision + ' → ' + str(Path(models_dir) / 'audio_tokenizer')) if run_higgs else 'no (use --include-higgs / --higgs-only)'}")
    print()

    if run_backbone and not args.skip_export:
        # Step 0: Save Qwen3 standalone so ModelBuilder can find it
        print("=== Step 0: Preparing Qwen3 standalone ===")
        qwen3_dir = prepare_qwen3_standalone(args.model, config_dir)
        print(f"  Qwen3 standalone: {qwen3_dir}\n")

        # Step 1: Backbone sub-models — all configuration is defined in Python (no external JSON):
        #   • audio sub-models → Olive RunConfig dicts (export_audio_models)
        #   • LLM             → genai create_model directly (export_llm; Olive bypass)
        print("=== Step 1: Backbone export + optimization ===")
        audio_precision = "int4" if args.device == "cpu" else "fp16"  # cpu→int4, cpu_fp16/gpu→fp16
        export_audio_models(args.model, models_dir, audio_precision)
        if args.skip_llm:
            print("  Skipping LLM (--skip-llm).")
        else:
            ep = "cuda" if args.device == "gpu" else "cpu"
            export_llm(qwen3_dir, models_dir, args.model, execution_provider=ep)

    if run_higgs and not args.skip_export:
        # Step 1b: Higgs Audio V2 Tokenizer → <models_dir>/audio_tokenizer (inference default dir)
        print("=== Step 1b: Higgs Audio Tokenizer export ===")
        higgs_out = str(Path(models_dir) / "audio_tokenizer")
        Path(higgs_out).mkdir(parents=True, exist_ok=True)
        export_higgs(args.model, higgs_out, precision=args.higgs_precision)

    # Every model_config.json (Olive artifacts, incl. audio_tokenizer/) carries an absolute
    # model_path — rewrite each to the bare .onnx basename so the published repo is portable.
    for mc in Path(models_dir).rglob("model_config.json"):
        with open(mc) as f:
            cfg = json.load(f)
        p = cfg.get("config", {}).get("model_path")
        if p and p != Path(p).name:
            cfg["config"]["model_path"] = Path(p).name
            with open(mc, "w") as f:
                json.dump(cfg, f, indent=4)
            print(f"  Relativized {mc.relative_to(models_dir)} → model_path={Path(p).name}")

    # Step 2: Write inference manifest
    print("=== Step 2: Writing inference manifest ===")
    write_inference_manifest(output_dir=models_dir, device=args.device)
    print()
    print("Done.")
    if not run_higgs:
        print()
        print("TIP: To also export the Higgs Audio Tokenizer (needed for voice cloning):")
        print("     python optimize.py --include-higgs")


if __name__ == "__main__":
    main()
