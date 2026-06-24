# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3", "torch", "torchvision", "torchaudio",
#   "onnx", "onnxruntime>=1.20", "numpy", "safetensors",
#   "huggingface_hub", "accelerate", "librosa", "soundfile",
# ]
# ///
"""Parity for text_embed / codec_embed ONNX vs the PyTorch wrappers.

Usage:
  uv run eval_embed.py --model-path onnx/cpu_fp32 --tts-path voicedesign
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "codes"))


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tts-path", default="voicedesign")
    args = ap.parse_args()

    import torch, onnxruntime as ort
    from user_script import (get_text_embed_model, get_codec_embed_model, _tts_dims, _load_tts)

    mdir = Path(args.model_path)
    sub = json.loads((mdir / "manifest.json").read_text())["sub_models"]
    dims = _tts_dims(_load_tts(args.tts_path))

    for name, get_model, lo, hi, in_name, out_name in [
        ("text_embed",  get_text_embed_model,  0, 1000,            "text_ids",  "text_embeds"),
        ("codec_embed", get_codec_embed_model, 0, dims["codec_vocab"], "codec_ids", "codec_embeds"),
    ]:
        sess = ort.InferenceSession(str(mdir / sub[name]["filename"]),
                                    providers=["CPUExecutionProvider"])
        wrap = get_model(args.tts_path)
        ids = torch.randint(lo, hi, (1, 16), dtype=torch.int64)
        with torch.no_grad():
            ref = wrap(ids).numpy()
        got = sess.run(None, {in_name: ids.numpy()})[0]
        print(f"=== {name} ===  onnx={got.shape} ref={ref.shape}  "
              f"cosine={cosine(got, ref):.6f}  max|Δ|={np.abs(got-ref).max():.3e}")


if __name__ == "__main__":
    main()
