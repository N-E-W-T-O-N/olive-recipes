"""Run a jina-embeddings-v4 ONNX sub-part build (text or image) → embedding vector.

  uv run inference.py --onnx-dir onnx/cpu_fp16 --text "Overview of climate change impacts"
  uv run inference.py --onnx-dir onnx/cpu_fp16 --text "..." --prefix Passage --truncate-dim 256
  uv run inference.py --onnx-dir onnx/cpu_fp16 --image doc.png

No PyTorch model load: text uses embeddings→backbone; image uses vision→embeddings→backbone with
the fixed prompt tensors from image_meta.npz (only pixel_values come from the processor). CPU only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import (embed_image_onnx, embed_text_onnx, load_sessions, load_tokenizer, npdt_of, quiet)


def main():
    ap = argparse.ArgumentParser(description="jina-embeddings-v4 ONNX sub-part inference")
    ap.add_argument("--onnx-dir", default="onnx/cpu_fp16")
    ap.add_argument("--text", default="hello")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prefix", default="Query")
    ap.add_argument("--tokenizer", default=None, help="override tokenizer/processor dir (default: onnx-dir)")
    ap.add_argument("--truncate-dim", type=int, default=None, help="Matryoshka: slice + renormalize")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    quiet()

    out = Path(args.onnx_dir)
    man = json.loads((out / "manifest.json").read_text())
    npdt = npdt_of(man)
    if args.image:
        sess = load_sessions(out, need_vision=True)
        meta = dict(np.load(out / "image_meta.npz"))
        emb = embed_image_onnx(sess, args.tokenizer or str(out), args.image, man["image_size"], npdt, meta)
        src = f"image {args.image}"
    else:
        sess = load_sessions(out, need_vision=False)
        tok = load_tokenizer(args.tokenizer or str(out))
        emb = embed_text_onnx(sess, tok, args.text, args.prefix, npdt)
        src = f"text {args.text!r}"
    if args.truncate_dim:
        emb = emb[:, :args.truncate_dim]
        emb = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-12)
    print(f"{src} → embedding{emb.shape}  [:8]={np.round(emb[0, :8], 4)}")
    if args.save:
        np.save(args.save, emb); print(f"saved → {args.save}")


if __name__ == "__main__":
    main()
