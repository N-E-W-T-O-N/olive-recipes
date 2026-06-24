# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "onnxruntime>=1.20", "numpy", "soundfile", "librosa", "transformers",
#   "numba>=0.60.0", "llvmlite>=0.43.0",
# ]
# ///
"""Parity + speed: KV-cache talker loop vs no-cache loop. Greedy → must match exactly.

Usage:
  uv run eval_cache.py --model-path onnx/voicedesign/cpu_int4 --tts-dir voicedesign
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tts-dir", required=True)
    ap.add_argument("--text", default="Hello, this is a KV cache parity test.")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    from inference import Pipeline
    pipe = Pipeline(args.model_path, tts_dir=args.tts_dir)
    if pipe.talker_cache is None:
        print("no talker_cache.onnx in this dir — nothing to compare"); return

    kw = dict(language="Auto", max_new_tokens=args.max_new_tokens, do_sample=False,
              sub_do_sample=False, seed=0, verbose=False)
    t0 = time.time(); codes_cached = pipe.generate(args.text, **kw); t_cached = time.time() - t0
    pipe.talker_cache = None                         # force no-cache path
    t0 = time.time(); codes_nocache = pipe.generate(args.text, **kw); t_nocache = time.time() - t0

    n = min(len(codes_cached), len(codes_nocache))
    match = float((codes_cached[:n] == codes_nocache[:n]).mean()) if n else -1
    print(f"cached frames={len(codes_cached)} ({t_cached:.1f}s)  "
          f"no-cache frames={len(codes_nocache)} ({t_nocache:.1f}s)  "
          f"speedup={t_nocache/max(t_cached,1e-9):.2f}x")
    print(f"all-16-codes exact match over {n} frames: {match*100:.2f}%")


if __name__ == "__main__":
    main()
