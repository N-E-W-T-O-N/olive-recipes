# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3", "torch", "torchvision", "torchaudio",
#   "onnx", "onnxruntime>=1.20", "numpy", "safetensors",
#   "huggingface_hub", "accelerate", "librosa", "soundfile",
# ]
# ///
"""Parity for the teacher-forced code_predictor ONNX.

Two checks:
  (1) wrapper vs model.forward_sub_talker_finetune  — is the in-graph wrapper
      faithful to the original PyTorch composition?
  (2) ONNX vs wrapper                               — did export preserve it?

Usage:
  uv run eval_predictor.py --model-path onnx/cpu_fp32 --tts-path voicedesign
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
    from user_script import get_code_predictor_model, _load_tts

    mdir = Path(args.model_path)
    fn = json.loads((mdir / "manifest.json").read_text())["sub_models"]["code_predictor"]["filename"]
    sess = ort.InferenceSession(str(mdir / fn), providers=["CPUExecutionProvider"])

    wrap = get_code_predictor_model(args.tts_path)
    talker = _load_tts(args.tts_path).talker
    ng = wrap.n_groups

    torch.manual_seed(0)
    hidden = torch.randn(1, 2048)
    codes = torch.randint(0, 2048, (1, ng), dtype=torch.int64)

    with torch.no_grad():
        wrap_logits = wrap(hidden, codes).numpy()
        native_logits, _ = talker.forward_sub_talker_finetune(codes, hidden)  # [B,15,vocab]
        native_logits = native_logits.numpy()
    onnx_logits = sess.run(None, {"talker_hidden": hidden.numpy(),
                                  "codec_ids": codes.numpy()})[0]

    print("=== (1) wrapper vs native forward_sub_talker_finetune ===")
    print(f"  shapes wrap={wrap_logits.shape} native={native_logits.shape}")
    print(f"  cosine={cosine(wrap_logits, native_logits):.6f}  "
          f"max|Δ|={np.abs(wrap_logits-native_logits).max():.3e}  "
          f"argmax agree={(wrap_logits.argmax(-1)==native_logits.argmax(-1)).mean():.3%}")

    print("=== (2) ONNX vs wrapper ===")
    print(f"  shapes onnx={onnx_logits.shape}")
    print(f"  cosine={cosine(onnx_logits, wrap_logits):.6f}  "
          f"max|Δ|={np.abs(onnx_logits-wrap_logits).max():.3e}  "
          f"argmax agree={(onnx_logits.argmax(-1)==wrap_logits.argmax(-1)).mean():.3%}")


if __name__ == "__main__":
    main()
