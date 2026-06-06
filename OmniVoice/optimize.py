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
  cpu       → INT4 all sub-models.  Smallest footprint.
  cpu_fp16  → FP16 all sub-models (audio encoders/decoder + LLM via ModelBuilder fp16).
               onnxruntime-genai's create_model supports fp16 on CPU.
               Best on CPUs with native FP16 SIMD (Intel Sapphire Rapids+, AMD Zen5+).
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
    qwen3_dir = Path(work_dir).resolve() / "qwen3_standalone"
    # Check for both single-file and sharded safetensors layouts
    has_weights = (qwen3_dir / "model.safetensors").exists() or \
                  any(qwen3_dir.glob("model-*-of-*.safetensors"))
    if qwen3_dir.exists() and has_weights:
        print(f"  Reusing existing {qwen3_dir}")
        return str(qwen3_dir)

    print(f"  Saving Qwen3 standalone to {qwen3_dir} ...")
    sys.path.insert(0, str(Path(__file__).parent))
    from user_script import save_qwen3_standalone
    # save_qwen3_standalone saves under work_dir; return the resolved absolute path
    save_qwen3_standalone(model_path, str(qwen3_dir.parent))
    return str(qwen3_dir)


# =============================================================================
# Step 1: Olive Export + Optimization
# =============================================================================

def export_llm_fp16_cpu(qwen3_dir: str, models_dir: str) -> None:
    """Export the LLM in FP16 for CPU directly via onnxruntime-genai create_model().

    Olive's ModelBuilder pass explicitly rejects fp16 on CPU:
        [model_builder.py] FP16 is not supported on CPU. → pruned.

    However, onnxruntime-genai's create_model() itself DOES support fp16 on CPU
    (confirmed by user testing).  We bypass Olive's validation by calling
    create_model() directly, which is exactly what the original
    convert_omnivoice_to_onnx.py does.

    Uses exclude_embeds=True + exclude_lm_head=True so the exported model
    accepts inputs_embeds and outputs hidden_states — matching the three-model
    OmniVoice backbone pipeline.
    """
    try:
        from onnxruntime_genai.models.builder import create_model
    except ImportError:
        raise ImportError(
            "onnxruntime-genai is required for fp16 LLM export. "
            "Install: pip install onnxruntime-genai"
        )

    out_dir   = Path(models_dir).resolve()
    genai_out = out_dir / "_genai_llm"
    cache_dir = out_dir.parent / "_genai_cache"
    genai_out.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"  create_model() precision=fp16 ep=cpu  (bypassing Olive — it rejects fp16 on CPU)")
    print(f"  exclude_embeds=True, exclude_lm_head=True")
    print("  This may take several minutes ...")
    create_model(
        model_name   = str(qwen3_dir),
        input_path   = str(qwen3_dir),
        output_dir   = str(genai_out),
        precision    = "fp16",
        execution_provider = "cpu",
        cache_dir    = str(cache_dir),
        exclude_embeds   = True,
        exclude_lm_head  = True,
    )
    print(f"  create_model() done → {genai_out}")

    # Copy canonical output files to models_dir with expected name
    import shutil
    onnx_files = sorted(genai_out.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"create_model() produced no .onnx files in {genai_out}")
    dst = out_dir / "llm_decoder.onnx"
    shutil.copy2(onnx_files[0], dst)
    print(f"  Copied {onnx_files[0].name} → {dst.name}")
    for data_file in genai_out.glob("*.onnx.data"):
        shutil.copy2(data_file, out_dir / data_file.name)
        print(f"  Copied external data: {data_file.name}")


def export_models(config_dir: str, models_dir: str = None,
                  device: str = "cpu", skip_llm: bool = False) -> None:
    """Run Olive on all three backbone sub-models.

    For cpu_fp16: audio sub-models go through Olive (fp16 is fine for small
    audio models), but the LLM uses export_llm_fp16_cpu() directly because
    Olive's ModelBuilder pass explicitly rejects fp16 on CPU EP.
    """
    from olive import run

    config_path  = Path(config_dir)
    _models_dir  = models_dir or str(config_path / MODELS_DIR)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")

    # audio_embeddings_encoder + audio_heads_decoder — Olive handles these fine
    for config in ("audio_embeddings_encoder.json", "audio_heads_decoder.json"):
        print(f"  Running {config}...")
        run(str(config_path / config))

    # LLM decoder
    if skip_llm:
        print("  Skipping LLM (--skip-llm).")
    elif device == "cpu_fp16":
        # Olive rejects fp16 on CPU; call create_model() directly
        print("  Running llm_decoder (fp16 CPU via create_model directly) ...")
        qwen3_abs = Path(config_path / "qwen3_standalone").resolve()
        if not qwen3_abs.exists():
            # Try the patched absolute path stored in the JSON
            import json
            llm_json = config_path / "llm_decoder.json"
            with open(llm_json) as f:
                llm_cfg = json.load(f)
            qwen3_abs = Path(llm_cfg["input_model"]["model_path"])
        export_llm_fp16_cpu(str(qwen3_abs), _models_dir)
    else:
        llm_cfg = str(config_path / "llm_decoder.json")
        print(f"  Running llm_decoder.json...")
        run(llm_cfg)
    print()


# =============================================================================
# Step 1b: Higgs Audio Tokenizer export
# =============================================================================

def export_higgs(higgs_config_dir: str = "higgs"):
    """Run Olive on all 4 Higgs Audio V2 Tokenizer sub-models.

    Configs are in higgs/ and all use FP16 (best balance of quality + speed
    for audio codec operations — INT4 is too lossy for DAC encoder/decoder).
    Both DAC models (acoustic_encoder, higgs_decoder) are pre-traced inside
    user_script.py before Olive sees them, resolving Python control-flow branches
    that would otherwise crash torch.onnx.export.
    """
    from olive import run

    config_path = Path(higgs_config_dir)
    print(f"=== Running Higgs Audio Tokenizer pipelines (configs from {config_path}) ===")
    for config in (
        "acoustic_encoder.json",
        "semantic_encoder.json",
        "quantizer_encoder.json",
        "higgs_decoder.json",
    ):
        print(f"  Running {config}...")
        run(str(config_path / config))
    print()


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
            "loading": "transformers.AutoModel (transformers>=5.4.0, no external deps needed)",
            "docs": "https://huggingface.co/docs/transformers/v5.4.0/en/model_doc/higgs_audio_v2_tokenizer",
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
                             "  cpu      → INT4 all models, CPUExecutionProvider\n"
                             "  cpu_fp16 → FP16 all models (audio+LLM), CPUExecutionProvider\n"
                             "             (onnxruntime-genai create_model supports fp16 on CPU)\n"
                             "  gpu      → FP16 audio + INT4 LLM, CUDAExecutionProvider")
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
    parser.add_argument("--models-dir", default=None,
                        help="Models output directory")
    parser.add_argument("--model-path", default=MODEL_NAME,
                        help=f"HF model ID or local path (default: {MODEL_NAME})")
    args = parser.parse_args()

    _default_dirs = {"cpu": "cpu_and_mobile", "cpu_fp16": "cpu_fp16", "gpu": "cuda"}
    config_dir = args.config_dir or _default_dirs[args.device]
    models_dir = args.models_dir or str(Path(config_dir) / MODELS_DIR)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    print(f"Target device : {args.device}")
    print(f"Config dir    : {config_dir}")
    print(f"Models dir    : {models_dir}")
    print(f"Model         : {args.model_path}")
    print()

    run_backbone = not args.higgs_only
    run_higgs    = args.include_higgs or args.higgs_only

    if run_backbone and not args.skip_export:
        # Step 0: Save Qwen3 standalone so ModelBuilder can find it
        print("=== Step 0: Preparing Qwen3 standalone ===")
        qwen3_dir = prepare_qwen3_standalone(args.model_path, config_dir)
        print(f"  Qwen3 standalone: {qwen3_dir}\n")

        # Patch llm_decoder.json with the ABSOLUTE path to qwen3_standalone.
        # Olive resolves HfModel's model_path from the config file's location, not
        # the working directory — so a relative path like "cpu_fp16/qwen3_standalone"
        # in cpu_fp16/llm_decoder.json would be misread as
        # "cpu_fp16/cpu_fp16/qwen3_standalone".  An absolute path is unambiguous.
        llm_json     = Path(config_dir) / "llm_decoder.json"
        qwen3_abs    = str(Path(qwen3_dir).resolve())
        if llm_json.exists():
            with open(llm_json) as f:
                llm_cfg = json.load(f)
            llm_cfg["input_model"]["model_path"] = qwen3_abs
            with open(llm_json, "w") as f:
                json.dump(llm_cfg, f, indent=4)
            print(f"  Patched {llm_json} → model_path={qwen3_abs}")

        # Step 1: Backbone sub-models
        print("=== Step 1: Backbone Olive export + optimization ===")
        if args.device == "cpu_fp16":
            print("  NOTE: Olive's ModelBuilder rejects fp16 on CPU — LLM will use")
            print("        create_model() directly via export_llm_fp16_cpu().")
        export_models(
            config_dir,
            models_dir=models_dir,
            device=args.device,
            skip_llm=args.skip_llm,
        )

    if run_higgs and not args.skip_export:
        # Step 1b: Higgs Audio V2 Tokenizer
        print("=== Step 1b: Higgs Audio Tokenizer export ===")
        export_higgs("higgs")

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
