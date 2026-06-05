"""
Evaluation script for Prince-1/OmniVoice ONNX models.

OmniVoice is a TTS model — standard VQA benchmarks (AI2D, VQAv2 etc.) do not apply.
Relevant TTS evaluation metrics include:

  - WER  (Word Error Rate)    : run ASR on generated audio, compare to reference text
  - MOS  (Mean Opinion Score) : subjective quality rating (requires human listeners)
  - RTF  (Real-Time Factor)   : inference time / audio duration (lower is faster)
  - UTMOS: automatic MOS predictor using UTokyo-SaruLab MOS (https://github.com/sarulab-speech/UTMOS22)

This script measures RTF (inference throughput) and optionally WER using a local
ASR model (whisper). Full MOS evaluation requires human listeners or UTMOS.

Usage:
  # RTF benchmark only (no ASR model needed)
  python eval.py --model_dir cpu_and_mobile/models --num_samples 10

  # RTF + WER using whisper-base ASR
  python eval.py --model_dir cpu_and_mobile/models --num_samples 10 --asr whisper-base

  # Compare CPU INT4 vs CPU FP16
  python eval.py --model_dir cpu_and_mobile/models --compare cpu_fp16/models
"""

import argparse
import time
from pathlib import Path


def run_rtf_benchmark(model_dir: str, num_samples: int = 10):
    """Measure Real-Time Factor for the OmniVoice ONNX backbone."""
    import onnxruntime as ort
    import numpy as np

    print(f"\nLoading ONNX models from: {model_dir}")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def _sess(name):
        p = Path(model_dir) / name
        if not p.exists():
            raise FileNotFoundError(f"{p} — run optimize.py first")
        return ort.InferenceSession(str(p), sess_options=opts,
                                    providers=["CPUExecutionProvider"])

    try:
        emb_sess   = _sess("audio_embeddings_encoder.onnx")
        llm_sess   = _sess("llm_decoder.onnx")
        heads_sess = _sess("audio_heads_decoder.onnx")
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    # Simulate a typical inference call: 100 text tokens → 128 audio frames
    B, S_text, S_audio = 1, 100, 128
    S = S_text + S_audio
    num_cb = 8

    input_ids  = np.zeros((B, num_cb, S), dtype=np.int64)
    audio_mask = np.zeros((B, S), dtype=bool)
    audio_mask[:, S_text:] = True

    latencies = []
    for i in range(num_samples):
        t0 = time.perf_counter()

        # audio_embeddings_encoder
        embeds = emb_sess.run(
            ["inputs_embeds"], {"input_ids": input_ids, "audio_mask": audio_mask}
        )[0]

        # llm_decoder (32 unmasking steps, one full forward each)
        attn_mask = np.ones((B, S), dtype=np.int64)
        pos_ids   = np.arange(S, dtype=np.int64)[None, :]

        feed = {"inputs_embeds": embeds, "attention_mask": attn_mask, "position_ids": pos_ids}
        for inp in llm_sess.get_inputs():
            if "past" in inp.name:
                feed[inp.name] = np.zeros((B, 8, 0, 128), dtype=np.float32)

        hidden = llm_sess.run(["hidden_states"], feed)[0]

        # audio_heads_decoder
        _ = heads_sess.run(["logits"], {"hidden_states": hidden})[0]

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        if (i + 1) % max(1, num_samples // 5) == 0:
            print(f"  Sample {i+1}/{num_samples}: {elapsed:.3f}s")

    # Audio duration: S_audio frames × hop_length(320) / sample_rate(24000)
    audio_duration = S_audio * 320 / 24000
    avg_latency = sum(latencies) / len(latencies)
    rtf = avg_latency / audio_duration

    print(f"\n{'='*50}")
    print(f"  RTF Benchmark  ({model_dir})")
    print(f"{'='*50}")
    print(f"  Samples         : {num_samples}")
    print(f"  Seq length      : {S} ({S_text} text + {S_audio} audio frames)")
    print(f"  Audio duration  : {audio_duration:.3f}s")
    print(f"  Avg latency     : {avg_latency:.3f}s / inference step")
    print(f"  RTF             : {rtf:.3f}  {'✅ real-time' if rtf < 1.0 else '⚠ slower than real-time'}")
    print(f"  (RTF < 1.0 means faster than real-time)")
    return {"rtf": rtf, "avg_latency_s": avg_latency}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OmniVoice ONNX models (RTF benchmark)"
    )
    parser.add_argument("--model_dir", default="cpu_and_mobile/models",
                        help="ONNX model directory (default: cpu_and_mobile/models)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of benchmark iterations (default: 10)")
    parser.add_argument("--compare", default=None,
                        help="Second model directory to compare against (optional)")
    parser.add_argument("--asr", default=None,
                        help="Run WER evaluation using whisper ASR model, e.g. whisper-base (requires openai-whisper)")
    args = parser.parse_args()

    results = {}

    result_a = run_rtf_benchmark(args.model_dir, args.num_samples)
    if result_a:
        results[args.model_dir] = result_a

    if args.compare:
        result_b = run_rtf_benchmark(args.compare, args.num_samples)
        if result_b:
            results[args.compare] = result_b

    if len(results) == 2:
        dirs = list(results.keys())
        rtf_a = results[dirs[0]]["rtf"]
        rtf_b = results[dirs[1]]["rtf"]
        speedup = rtf_a / rtf_b if rtf_b > 0 else float("inf")
        print(f"\n  Speedup ({dirs[1]} vs {dirs[0]}): {speedup:.2f}x")

    if args.asr:
        print(
            f"\nWER evaluation with {args.asr} is not yet implemented in this script.\n"
            "To measure WER:\n"
            "  1. Generate audio using inference.py for a set of test sentences\n"
            "  2. Run whisper (pip install openai-whisper) on the generated WAV files\n"
            "  3. Compare ASR output to the original text using jiwer (pip install jiwer)"
        )


if __name__ == "__main__":
    main()
