# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3", "torch", "torchvision", "torchaudio",
#   "onnx", "onnxruntime>=1.20", "numpy", "safetensors",
#   "huggingface_hub", "accelerate", "librosa", "soundfile",
#   "numba>=0.60.0", "llvmlite>=0.43.0",
# ]
# ///
"""Parity: speaker_encoder.onnx (x-vector, with reimplemented mel/STFT) vs PyTorch
extract_speaker_embedding. Validates the one genuinely-new ONNX component for Base
voice cloning across several audio lengths.

Usage:
  uv run eval_speaker.py --tts-dir base/1.7B --onnx onnx/base17_cpu_int4/speaker_encoder.onnx
"""
import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "codes"))


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-dir", default="base/1.7B")
    ap.add_argument("--onnx", default="onnx/base17_cpu_int4/speaker_encoder.onnx")
    args = ap.parse_args()

    import torch, onnxruntime as ort
    from user_script import _load_tts

    model = _load_tts(args.tts_dir)
    assert model.speaker_encoder is not None, "not a base checkpoint"
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    for secs in (2.0, 3.5, 6.0):
        n = int(24000 * secs)
        audio = (0.5 * rng.standard_normal(n)).astype(np.float32)
        with torch.no_grad():
            ref = model.extract_speaker_embedding(audio, 24000).cpu().numpy().ravel()
        got = sess.run(None, {"audio": audio[None]})[0].ravel()
        print(f"len={secs:>4}s  onnx{got.shape} ref{ref.shape}  "
              f"cosine={cosine(got, ref):.6f}  max|Δ|={np.abs(got - ref).max():.3e}")


if __name__ == "__main__":
    main()
