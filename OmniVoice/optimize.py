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
  python optimize.py --device cpu                  # CPU INT4 (cpu_and_mobile/)
  python optimize.py --device cpu_fp16             # CPU FP16 (cpu_fp16/)
  python optimize.py --device gpu                  # CUDA FP16 (cuda/)
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

def export_models(config_dir: str, llm_config_path: str = None):
    """Run Olive on all three backbone sub-models."""
    from olive import run

    config_path = Path(config_dir)
    print(f"=== Running Olive pipelines (configs from {config_path}) ===")

    # audio_embeddings_encoder + audio_heads_decoder (PyTorchModel → ONNX → quantize)
    for config in ("audio_embeddings_encoder.json", "audio_heads_decoder.json"):
        print(f"  Running {config}...")
        run(str(config_path / config))

    # llm_decoder (HfModel → ModelBuilder with exclude_embeds + exclude_lm_head)
    llm_cfg = llm_config_path or str(config_path / "llm_decoder.json")
    print(f"  Running {Path(llm_cfg).name}...")
    run(llm_cfg)
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
            "note": "Exported separately via convert_omnivoice_to_onnx.py --only higgs",
            "requires": "boson-multimodal @ git+https://github.com/boson-ai/higgs-audio.git",
            "sub_models": [
                "higgs_acoustic_encoder.onnx",
                "higgs_semantic_encoder.onnx",
                "higgs_quantizer_encoder.onnx",
                "higgs_decoder.onnx",
            ],
            "audio_tokenizer_dir": "audio_tokenizer/",
            "sample_rate_acoustic": 24000,
            "sample_rate_semantic": 16000,
            "hop_length": 320,
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

    if not args.skip_export:
        # Step 0: Save Qwen3 standalone so ModelBuilder can find it
        print("=== Step 0: Preparing Qwen3 standalone ===")
        qwen3_dir = prepare_qwen3_standalone(args.model_path, config_dir)
        print(f"  Qwen3 standalone: {qwen3_dir}\n")

        # Patch llm_decoder.json to point to the saved standalone
        llm_json = Path(config_dir) / "llm_decoder.json"
        if llm_json.exists():
            with open(llm_json) as f:
                llm_cfg = json.load(f)
            llm_cfg["input_model"]["model_path"] = qwen3_dir
            with open(llm_json, "w") as f:
                json.dump(llm_cfg, f, indent=4)
            print(f"  Patched {llm_json} → model_path={qwen3_dir}")

        # Step 1: Run Olive on all sub-models
        print("=== Step 1: Olive export + optimization ===")
        if args.skip_llm:
            print("  Skipping LLM (--skip-llm).")
            from olive import run
            for config in ("audio_embeddings_encoder.json", "audio_heads_decoder.json"):
                print(f"  Running {config}...")
                run(str(Path(config_dir) / config))
        else:
            export_models(config_dir)

    # Step 2: Write inference manifest
    print("=== Step 2: Writing inference manifest ===")
    write_inference_manifest(output_dir=models_dir, device=args.device)
    print()
    print("Done.")
    print()
    print("NOTE: To export the Higgs Audio Tokenizer (voice cloning components),")
    print("      download the OmniVoice model directory and run:")
    print("      python convert_omnivoice_to_onnx.py --only higgs --out-dir ./higgs_onnx")
    print("      (requires: pip install boson-multimodal @ git+https://github.com/boson-ai/higgs-audio.git)")


if __name__ == "__main__":
    main()
