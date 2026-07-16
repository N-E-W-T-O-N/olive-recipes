"""Eval composed ONNX sub-parts vs full PyTorch — accepts MULTIPLE build dirs at once and reports
each one's detected precision (from its manifest) alongside its pooled-embedding cosine.

  uv run eval.py --model vllm-retrieval --onnx-dir onnx/cpu_fp16
  uv run eval.py --model vllm-retrieval --onnx-dir onnx/cpu_fp16 onnx/cpu_fp32 onnx/cpu_fp16-int8

The full PyTorch reference is computed ONCE (model freed before any ORT session loads — they don't
co-fit in RAM), then every dir is scored against it. Precision/quant is read per-dir from manifest,
so you can eye the fidelity vs size trade-off across builds in one run. CPU only.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from common import (cosine, describe_precision, embed_image_onnx, embed_text_onnx, hf_name,
                    image_rope_and_mask, load_model, load_sessions, load_tokenizer, make_image_inputs,
                    make_inputs, mean_pool, npdt_of, quiet, text_position_ids)

TEXTS = [("Query", "capital of France?"), ("Passage", "Paris is the capital of France."),
         ("Query", "def add(a,b): return a+b")]


def pytorch_refs(model_dir, image, size):
    """(text_refs, img_ref) from the full model; the model is freed before this returns.
    Loaded in fp16 to fit RAM (the fp32 model is ~12 GB and won't co-fit with the ORT sessions)."""
    import torch as _t
    model = load_model(model_dir, dtype=_t.float16, attn="eager")
    tok = load_tokenizer(model_dir)
    text_refs = []
    for prefix, t in TEXTS:
        ids, am = make_inputs(tok, [t], prefix=prefix)
        pos = text_position_ids(am)
        with torch.no_grad():
            h = model.model.language_model(inputs_embeds=model.model.language_model.embed_tokens(ids),
                                           attention_mask=am, position_ids=pos, use_cache=False).last_hidden_state
        text_refs.append((prefix, t, mean_pool(h, am).float().numpy()))
    batch = make_image_inputs(model_dir, image, size)
    ipos, vm = image_rope_and_mask(model, batch)
    with torch.no_grad():
        full = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           position_ids=ipos, pixel_values=batch["pixel_values"],
                           image_grid_thw=batch["image_grid_thw"], use_cache=False).last_hidden_state
    img_ref = mean_pool(full, vm).float().numpy()
    del model, full, h; gc.collect()
    return tok, text_refs, img_ref


def eval_dir(onnx_dir, model_dir, tok, text_refs, img_ref, image):
    out = Path(onnx_dir)
    man = json.loads((out / "manifest.json").read_text())
    npdt = npdt_of(man); size = man["image_size"]
    prec = describe_precision(man)
    sess = load_sessions(out, need_vision=True)
    meta = dict(np.load(out / "image_meta.npz"))
    worst = 1.0
    print(f"\n--- {out}   precision={prec} ---")
    for prefix, t, ref in text_refs:
        c = cosine(embed_text_onnx(sess, tok, t, prefix, npdt), ref); worst = min(worst, c)
        print(f"  [text ] {prefix+': '+t[:34]!r:44} cos={c:.6f}")
    c = cosine(embed_image_onnx(sess, model_dir, image, size, npdt, meta), img_ref); worst = min(worst, c)
    print(f"  [image] {'synthetic' if not image else image:44} cos={c:.6f}")
    return prec, worst


def main():
    ap = argparse.ArgumentParser(description="eval jina-embeddings-v4 ONNX sub-parts (multi-dir) vs PyTorch")
    ap.add_argument("--model", default="vllm-retrieval")
    ap.add_argument("dirs", nargs="*", help="one or more build dirs (positional)")
    ap.add_argument("--onnx-dir", nargs="+", default=None, help="one or more build dirs (flag form)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--tol", type=float, default=0.999)
    args = ap.parse_args()
    quiet()
    # dedupe by resolved path (keep first occurrence) so the same dir isn't eval'd twice —
    # e.g. "onnx/fp32" and "onnx/fp32/" are one model
    raw = args.dirs or args.onnx_dir or ["onnx/cpu_fp16"]
    seen, onnx_dirs = set(), []
    for d in raw:
        key = Path(d).resolve()
        if key not in seen:
            seen.add(key); onnx_dirs.append(d)
    print(f"=== eval composed ONNX vs full PyTorch | {hf_name(args.model)} (cpu) ===")

    tok, text_refs, img_ref = pytorch_refs(args.model, args.image, 224)
    rows = []
    for d in onnx_dirs:
        prec, worst = eval_dir(d, args.model, tok, text_refs, img_ref, args.image)
        rows.append((d, prec, worst, worst >= args.tol))

    print(f"\n=== summary (tol {args.tol}) ===")
    w = max(len(d) for d, *_ in rows)
    for d, prec, worst, ok in rows:
        print(f"  {d:<{w}}  {prec:<22} worst cos {worst:.6f}  {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if all(ok for *_, ok in rows) else 1)


if __name__ == "__main__":
    main()
