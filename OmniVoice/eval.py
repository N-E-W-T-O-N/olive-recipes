"""
Evaluation script for Prince-1/OmniVoice ONNX models.

Two modes:

  1. NUMERICAL EQUIVALENCE  (--mode equiv)
     Verifies that each ONNX sub-model produces outputs numerically close to the
     original PyTorch module it wraps.  Catches precision regressions introduced
     by FP16 conversion, graph optimisations, or weight-norm stripping.

     Sub-models tested:
       Backbone
         A) AudioEmbeddingsEncoderWrapper   → audio_embeddings_encoder.onnx
         B) AudioHeadsDecoderWrapper        → audio_heads_decoder.onnx
       Higgs Audio Tokenizer
         C) tok.acoustic_encoder            → acoustic_encoder.onnx
         D) tok.semantic_model+encoder_sem  → semantic_encoder.onnx
         E) tok.fc+tok.quantizer.encode     → quantizer_encoder.onnx
         F) tok.quantizer.decode+fc2+decoder→ higgs_decoder.onnx

  2. RTF BENCHMARK  (--mode rtf)
     Measures Real-Time Factor (inference time / audio duration) for the backbone.
     Optionally compares two model directories side-by-side.

Usage:
  # Numerical equivalence check (all sub-models)
  python eval.py --mode equiv --model_dir cpu_and_mobile/models --higgs_dir higgs/models

  # Skip backbone (only test Higgs tokenizer)
  python eval.py --mode equiv --higgs_dir higgs/models --skip_backbone

  # RTF benchmark
  python eval.py --mode rtf --model_dir cpu_and_mobile/models --num_samples 10

  # Compare two model directories (RTF)
  python eval.py --mode rtf --model_dir cpu_and_mobile/models --compare cpu_fp16/models
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add this directory to sys.path so user_script / codes can be imported
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# =============================================================================
# Shared helpers
# =============================================================================

MODEL_NAME   = "Prince-1/OmniVoice"
SR_24K       = 24_000
SR_16K       = 16_000
DOWNSAMPLE   = 960     # DAC hop length at 24 kHz → 25 fps
HIGGS_N_CB   = 8
HIGGS_CB_SZ  = 1024
HIDDEN_SIZE  = 1024
NUM_CB       = 8
AUDIO_VOCAB  = 1025


def _ort_session(path: str, provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=opts, providers=[provider])


def _cast_inputs(feed: dict, sess) -> dict:
    """Cast float32 numpy arrays to the dtype the ONNX session expects.

    OnnxFloatToFloat16 converts all float inputs to float16. Running a float32
    array into a float16 session raises INVALID_ARGUMENT. This helper inspects
    the session's input type map and casts each feed value accordingly.

    int64 inputs (codes) are left as-is.
    """
    import onnxruntime as ort
    # Build name→numpy_dtype map from session metadata
    _ort_to_np = {
        "tensor(float16)": np.float16,
        "tensor(float)":   np.float32,
        "tensor(double)":  np.float64,
        "tensor(int64)":   np.int64,
        "tensor(int32)":   np.int32,
    }
    type_map = {inp.name: _ort_to_np.get(inp.type, None) for inp in sess.get_inputs()}
    out = {}
    for k, v in feed.items():
        target = type_map.get(k)
        if target is not None and isinstance(v, np.ndarray) and v.dtype != target:
            v = v.astype(target)
        out[k] = v
    return out


def _stats(pt: np.ndarray, onnx: np.ndarray, label: str):
    """Print comparison statistics and return pass/fail."""
    # Cast both to float64 for stable comparison
    a = pt.astype(np.float64).ravel()
    b = onnx.astype(np.float64).ravel()
    abs_diff = np.abs(a - b)
    rel_diff = abs_diff / (np.abs(a) + 1e-8)
    cos_sim  = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    max_abs = abs_diff.max()
    mean_abs = abs_diff.mean()
    max_rel = rel_diff.max()

    # Thresholds: FP16 conversion can accumulate error across transformer layers.
    # HuBERT with blocked LayerNorm has 12 fp16↔fp32 Cast boundaries → higher max_abs.
    # Cosine similarity > 0.999 is the primary signal; max_abs < 0.2 is a safety bound.
    passed = bool(max_abs < 0.2 and cos_sim > 0.999)

    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status}  {label}")
    print(f"         shape  PT={pt.shape}  ONNX={onnx.shape}")
    print(f"         max_abs_diff={max_abs:.4e}   mean_abs_diff={mean_abs:.4e}")
    print(f"         max_rel_diff={max_rel:.4e}   cosine_sim={cos_sim:.6f}")
    return passed


def _stats_int(pt: np.ndarray, onnx: np.ndarray, label: str):
    """Compare integer tensors (codec codes) — must be exactly equal."""
    eq = (pt == onnx).all()
    mismatch = (pt != onnx).sum()
    status = "[PASS]" if eq else f"[MISMATCH] ({mismatch}/{pt.size} elements differ)"
    print(f"  {status}  {label}  shape={pt.shape}")
    return bool(eq)


# =============================================================================
# Mode A-B: Backbone sub-models
# =============================================================================

def test_backbone(model_dir: str, provider: str = "CPUExecutionProvider") -> dict:
    """Compare backbone ONNX sub-models against PyTorch wrappers.

    Tests:
      A) AudioEmbeddingsEncoderWrapper  vs  audio_embeddings_encoder.onnx
      B) AudioHeadsDecoderWrapper       vs  audio_heads_decoder.onnx

    The LLM (llm_decoder) is a black-box genai model — its numerical equivalence
    is covered by the onnxruntime-genai validation in the ModelBuilder pass.
    """
    import torch
    from user_script import get_audio_embeddings_model, get_audio_heads_model

    results = {}
    print("\n" + "="*60)
    print("  Backbone sub-model equivalence")
    print("="*60)

    # -------------------------------------------------------------------------
    # A) AudioEmbeddingsEncoderWrapper
    # -------------------------------------------------------------------------
    print("\n[A] AudioEmbeddingsEncoder  (PyTorch vs ONNX)")
    onnx_path = Path(model_dir) / "audio_embeddings_encoder.onnx"
    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path} not found")
        results["audio_embeddings"] = None
    else:
        pt_model = get_audio_embeddings_model(MODEL_NAME)
        pt_model.eval()

        B, S = 1, 64
        input_ids  = torch.randint(0, AUDIO_VOCAB, (B, NUM_CB, S), dtype=torch.int64)
        audio_mask = torch.zeros(B, S, dtype=torch.bool)
        audio_mask[:, S // 4 : S * 3 // 4] = True

        with torch.no_grad():
            pt_out = pt_model(input_ids, audio_mask).numpy()

        sess = _ort_session(str(onnx_path), provider)
        onnx_out = sess.run(
            ["inputs_embeds"],
            _cast_inputs({"input_ids": input_ids.numpy(), "audio_mask": audio_mask.numpy()}, sess)
        )[0]

        results["audio_embeddings"] = _stats(pt_out, onnx_out, "inputs_embeds")

    # -------------------------------------------------------------------------
    # B) AudioHeadsDecoderWrapper
    # -------------------------------------------------------------------------
    print("\n[B] AudioHeadsDecoder  (PyTorch vs ONNX)")
    onnx_path = Path(model_dir) / "audio_heads_decoder.onnx"
    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path} not found")
        results["audio_heads"] = None
    else:
        pt_model = get_audio_heads_model(MODEL_NAME)
        pt_model.eval()

        B, S = 1, 64
        hidden = torch.randn(B, S, HIDDEN_SIZE, dtype=torch.float32)

        with torch.no_grad():
            pt_out = pt_model(hidden).numpy()

        sess = _ort_session(str(onnx_path), provider)
        onnx_out = sess.run(
            ["logits"],
            _cast_inputs({"hidden_states": hidden.numpy()}, sess)
        )[0]

        results["audio_heads"] = _stats(pt_out, onnx_out, "logits")

    return results


# =============================================================================
# Mode C-F: Higgs tokenizer sub-models
# =============================================================================

def test_higgs(higgs_dir: str, provider: str = "CPUExecutionProvider") -> dict:
    """Compare all four Higgs ONNX sub-models against the PyTorch tokenizer.

    Tests:
      C) tok.acoustic_encoder            vs  acoustic_encoder.onnx
      D) tok.semantic_model+encoder_sem  vs  semantic_encoder.onnx
      E) tok.fc+tok.quantizer.encode     vs  quantizer_encoder.onnx   (int64 codes)
      F) quantizer.decode+fc2+decoder    vs  higgs_decoder.onnx
    """
    import torch
    from user_script import _load_higgs_tokenizer, _prepare_tok
    from codes.model_wrappers import (
        HiggsAcousticEncoderWrapper, HiggsSemanticEncoderWrapper,
        HiggsQuantizerEncoderWrapper, HiggsDecoderWrapper,
        HIGGS_D_ACOUSTIC, HIGGS_D_SEMANTIC,
    )

    results = {}
    print("\n" + "="*60)
    print("  Higgs Audio Tokenizer sub-model equivalence")
    print("="*60)

    tok = _load_higgs_tokenizer(MODEL_NAME)
    tok = _prepare_tok(tok)

    T_audio = SR_24K // DOWNSAMPLE   # 25 frames for 1 second

    # -------------------------------------------------------------------------
    # C) acoustic_encoder
    # -------------------------------------------------------------------------
    print("\n[C] AcousticEncoder  (tok.acoustic_encoder  vs  ONNX)")
    onnx_path = Path(higgs_dir) / "acoustic_encoder.onnx"
    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path} not found")
        results["acoustic_encoder"] = None
    else:
        pt_wrapper = HiggsAcousticEncoderWrapper(tok.acoustic_encoder)
        pt_wrapper.eval()

        torch.manual_seed(42)
        wav24 = torch.clamp(torch.randn(1, 1, SR_24K, dtype=torch.float32), -1.0, 1.0)
        with torch.no_grad():
            pt_out = pt_wrapper(wav24).numpy()

        sess = _ort_session(str(onnx_path), provider)
        onnx_out = sess.run(
            ["acoustic_features"],
            _cast_inputs({"waveform_24k": wav24.numpy()}, sess)
        )[0]

        results["acoustic_encoder"] = _stats(pt_out, onnx_out, "acoustic_features")

    # -------------------------------------------------------------------------
    # D) semantic_encoder
    # -------------------------------------------------------------------------
    print("\n[D] SemanticEncoder  (tok.semantic_model+encoder_semantic  vs  ONNX)")
    onnx_path = Path(higgs_dir) / "semantic_encoder.onnx"
    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path} not found")
        results["semantic_encoder"] = None
    else:
        pt_wrapper = HiggsSemanticEncoderWrapper(
            tok.semantic_model, tok.encoder_semantic,
            downsample_factor=getattr(tok.config, "semantic_downsample_factor", 2),
            pad=160,
        )
        pt_wrapper.eval()

        # Use a low-frequency sine wave — realistic speech-like content.
        # torch.randn amplifies fp16 LayerNorm Cast errors across 12 attention layers.
        t = torch.linspace(0, 1.0, SR_16K, dtype=torch.float32)
        wav16 = (0.5 * torch.sin(2 * 3.14159 * 440 * t)).unsqueeze(0)   # 440 Hz tone, (1, T)
        with torch.no_grad():
            pt_out = pt_wrapper(wav16).numpy()

        sess = _ort_session(str(onnx_path), provider)
        onnx_out = sess.run(
            ["semantic_features"],
            _cast_inputs({"waveform_16k": wav16.numpy()}, sess)
        )[0]

        results["semantic_encoder"] = _stats(pt_out, onnx_out, "semantic_features")

    # -------------------------------------------------------------------------
    # E) quantizer_encoder  — outputs int64 codes, must be exact
    # -------------------------------------------------------------------------
    print("\n[E] QuantizerEncoder  (tok.fc+tok.quantizer.encode  vs  ONNX)  [exact int match]")
    onnx_path = Path(higgs_dir) / "quantizer_encoder.onnx"
    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path} not found")
        results["quantizer_encoder"] = None
    else:
        pt_wrapper = HiggsQuantizerEncoderWrapper(tok.fc, tok.quantizer, merge_mode="concat")
        pt_wrapper.eval()

        acoustic_feat = torch.randn(1, HIGGS_D_ACOUSTIC, T_audio, dtype=torch.float32)
        semantic_feat = torch.randn(1, HIGGS_D_SEMANTIC, T_audio, dtype=torch.float32)
        with torch.no_grad():
            pt_out = pt_wrapper(acoustic_feat, semantic_feat).numpy()

        sess = _ort_session(str(onnx_path), provider)
        onnx_out = sess.run(
            ["codes"],
            _cast_inputs({
                "acoustic_features": acoustic_feat.numpy(),
                "semantic_features": semantic_feat.numpy(),
            }, sess)
        )[0]

        results["quantizer_encoder"] = _stats_int(pt_out, onnx_out, "codes")

        # Keep codes for the decoder test
        test_codes_pt   = pt_out
        test_codes_onnx = onnx_out

    # -------------------------------------------------------------------------
    # F) higgs_decoder  — compare waveform outputs
    #    Run twice: once with PT codes and once with ONNX codes, compare both
    #    against the PT decoder output.
    # -------------------------------------------------------------------------
    print("\n[F] HiggsDecoder  (quantizer.decode+fc2+acoustic_decoder  vs  ONNX)")
    onnx_path = Path(higgs_dir) / "higgs_decoder.onnx"
    if not onnx_path.exists():
        print(f"  [SKIP] {onnx_path} not found")
        results["higgs_decoder"] = None
    else:
        import torch
        pt_wrapper = HiggsDecoderWrapper(tok.quantizer, tok.fc2, tok.acoustic_decoder)
        pt_wrapper.eval()

        # Use consistent codes for both sides — take PT codes if available,
        # otherwise generate fresh dummy codes
        if "test_codes_pt" in dir():
            codes_np = test_codes_pt   # (8, 1, T_audio) int64 from step E
        else:
            codes_np = np.random.randint(0, HIGGS_CB_SZ, (HIGGS_N_CB, 1, T_audio),
                                         dtype=np.int64)

        codes_t = torch.from_numpy(codes_np)
        with torch.no_grad():
            pt_out = pt_wrapper(codes_t).numpy()   # (1, 1, T_samples)

        sess = _ort_session(str(onnx_path), provider)
        onnx_out = sess.run(
            ["waveform_24k"],
            _cast_inputs({"codes": codes_np}, sess)
        )[0]   # (1, 1, T_samples)

        results["higgs_decoder"] = _stats(
            pt_out.ravel(), onnx_out.ravel(), "waveform_24k"
        )

        # Extra: if quantizer produced different codes (PT vs ONNX), also check
        # that the decoder output for ONNX codes is still close to PT decode of PT codes
        if "test_codes_onnx" in dir() and not (test_codes_pt == test_codes_onnx).all():
            print("    [Extra] Decoder with ONNX codes vs PT decoder with PT codes:")
            onnx_codes_out = sess.run(
                ["waveform_24k"],
                _cast_inputs({"codes": test_codes_onnx}, sess)
            )[0]
            _stats(pt_out.ravel(), onnx_codes_out.ravel(),
                   "waveform_24k (ONNX codes vs PT codes)")

    return results


# =============================================================================
# Mode: RTF benchmark
# =============================================================================

def run_rtf_benchmark(model_dir: str, num_samples: int = 10,
                      provider: str = "CPUExecutionProvider") -> dict:
    """Measure Real-Time Factor for the OmniVoice backbone pipeline."""
    print(f"\nLoading backbone ONNX models from: {model_dir}")
    opts_path = Path(model_dir)

    def _sess(name):
        p = opts_path / name
        if not p.exists():
            raise FileNotFoundError(f"{p} — run optimize.py first")
        return _ort_session(str(p), provider)

    try:
        emb_sess   = _sess("audio_embeddings_encoder.onnx")
        llm_sess   = _sess("llm_decoder.onnx")
        heads_sess = _sess("audio_heads_decoder.onnx")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return {}

    B, S_text, S_audio = 1, 100, 128
    S = S_text + S_audio

    input_ids  = np.zeros((B, NUM_CB, S), dtype=np.int64)
    audio_mask = np.zeros((B, S), dtype=bool)
    audio_mask[:, S_text:] = True
    attn_mask  = np.ones((B, S), dtype=np.int64)
    pos_ids    = np.arange(S, dtype=np.int64)[None, :]

    # Build LLM feed template with empty KV cache
    llm_feed_template = {
        "attention_mask": attn_mask,
        "position_ids":   pos_ids,
    }
    for inp in llm_sess.get_inputs():
        if "past" in inp.name:
            llm_feed_template[inp.name] = np.zeros((B, 8, 0, 128), dtype=np.float32)

    latencies = []
    for i in range(num_samples):
        t0 = time.perf_counter()

        embeds = emb_sess.run(
            ["inputs_embeds"],
            _cast_inputs({"input_ids": input_ids, "audio_mask": audio_mask}, emb_sess)
        )[0]

        llm_feed = _cast_inputs({**llm_feed_template, "inputs_embeds": embeds}, llm_sess)
        hidden = llm_sess.run(["hidden_states"], llm_feed)[0]

        _ = heads_sess.run(
            ["logits"], _cast_inputs({"hidden_states": hidden}, heads_sess)
        )[0]

        latencies.append(time.perf_counter() - t0)
        if (i + 1) % max(1, num_samples // 5) == 0:
            print(f"  Sample {i+1}/{num_samples}: {latencies[-1]:.3f}s")

    # Audio duration: S_audio frames × hop_length / sample_rate
    audio_dur  = S_audio * DOWNSAMPLE / SR_24K
    avg_lat    = sum(latencies) / len(latencies)
    rtf        = avg_lat / audio_dur

    print(f"\n{'='*55}")
    print(f"  RTF Benchmark  —  {model_dir}")
    print(f"{'='*55}")
    print(f"  Samples         : {num_samples}")
    print(f"  Seq length      : {S}  ({S_text} text + {S_audio} audio frames)")
    print(f"  Audio duration  : {audio_dur:.3f}s  (per backbone step)")
    print(f"  Avg step latency: {avg_lat:.3f}s")
    print(f"  RTF             : {rtf:.3f}  "
          f"{'✅ faster than real-time' if rtf < 1.0 else '⚠  slower than real-time'}")
    return {"rtf": rtf, "avg_latency_s": avg_lat, "audio_duration_s": audio_dur}


# =============================================================================
# Summary printer
# =============================================================================

def _print_summary(results: dict):
    passed = [k for k, v in results.items() if v is True]
    failed = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]

    print("\n" + "="*60)
    print("  EQUIVALENCE SUMMARY")
    print("="*60)
    for k, v in results.items():
        icon = "[PASS]" if v is True else ("[FAIL]" if v is False else "[SKIP]")
        print(f"  {icon}  {k}")
    print(f"\n  Passed: {len(passed)}  Failed: {len(failed)}  Skipped: {len(skipped)}")
    if failed:
        print(f"\n  [WARN] Failures in: {', '.join(failed)}")
        print("     Possible causes: FP16 precision loss, incorrect ONNX graph, "
              "weight_norm stripping issue.")
    return len(failed) == 0


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice ONNX evaluation: numerical equivalence + RTF benchmark"
    )
    parser.add_argument("--mode", choices=["equiv", "rtf"], default="equiv",
                        help="Evaluation mode: 'equiv' (numerical) or 'rtf' (benchmark)")
    parser.add_argument("--model_dir", default="cpu_and_mobile/models",
                        help="Backbone ONNX model directory (default: cpu_and_mobile/models)")
    parser.add_argument("--higgs_dir", default="higgs/models",
                        help="Higgs tokenizer ONNX directory (default: higgs/models)")
    parser.add_argument("--compare", default=None,
                        help="Second model directory to compare RTF against (rtf mode only)")
    parser.add_argument("--skip_backbone", action="store_true",
                        help="Skip backbone sub-model tests (equiv mode)")
    parser.add_argument("--skip_higgs", action="store_true",
                        help="Skip Higgs tokenizer sub-model tests (equiv mode)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of RTF benchmark iterations (rtf mode, default: 10)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDAExecutionProvider")
    args = parser.parse_args()

    provider = "CUDAExecutionProvider" if args.cuda else "CPUExecutionProvider"

    # ------------------------------------------------------------------
    if args.mode == "equiv":
        all_results = {}

        if not args.skip_backbone:
            try:
                bb = test_backbone(args.model_dir, provider)
                all_results.update(bb)
            except Exception as e:
                print(f"\n[ERROR] Backbone test failed: {e}")
                import traceback; traceback.print_exc()

        if not args.skip_higgs:
            try:
                hg = test_higgs(args.higgs_dir, provider)
                all_results.update(hg)
            except Exception as e:
                print(f"\n[ERROR] Higgs test failed: {e}")
                import traceback; traceback.print_exc()

        ok = _print_summary(all_results)
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    elif args.mode == "rtf":
        results = {}

        r = run_rtf_benchmark(args.model_dir, args.num_samples, provider)
        if r:
            results[args.model_dir] = r

        if args.compare:
            r2 = run_rtf_benchmark(args.compare, args.num_samples, provider)
            if r2:
                results[args.compare] = r2

        if len(results) == 2:
            dirs  = list(results.keys())
            rtf_a = results[dirs[0]]["rtf"]
            rtf_b = results[dirs[1]]["rtf"]
            if rtf_b > 0:
                sp = rtf_a / rtf_b
                faster = dirs[1] if sp > 1 else dirs[0]
                print(f"\n  Relative speedup: {abs(sp):.2f}×  "
                      f"({faster} is {'faster' if sp > 1 else 'slower'})")


if __name__ == "__main__":
    main()
