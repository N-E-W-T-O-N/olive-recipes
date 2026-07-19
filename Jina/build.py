"""Build jina-embeddings-v4 (vLLM merged, per-task) → ONNX sub-parts (vision / embeddings / backbone).

  uv run build.py --model vllm-retrieval --output onnx/cpu_fp16              # fp16 (default)
  uv run build.py --model vllm-retrieval --output onnx/cpu_fp32 --precision fp32
  uv run build.py --model vllm-retrieval --output onnx/cpu_int8 --precision int8
  uv run build.py --model vllm-retrieval --output onnx/cpu_int4 --precision int4

--precision {fp16,fp32,int8,int4}. int8/int4 build a fp16 graph then weight-quantize the backbone
in place (block-wise MatMulNBits; vision/embeddings stay fp16). int8 keeps pooled-cosine ≥0.999;
int4 does NOT (embedding drift ~5-8%) so it only warns — measure with eval.py before use.
CPU only (no device flag).
"""
import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch

from common import (HIDDEN, MATRYOSHKA, BackboneSub, EmbeddingsSub, VisionSub, cosine, hf_name,
                    image_rope_and_mask, load_model, load_tokenizer, make_image_inputs, make_session,
                    mean_pool, quiet)


def _export(enc, sample, names_in, names_out, dyn, raw):
    with torch.no_grad():
        torch.onnx.export(enc, sample, str(raw), input_names=names_in, output_names=names_out,
                          dynamic_axes=dyn, opset_version=19, do_constant_folding=True, dynamo=False)


def _consolidate(out, stem, raw):
    """temp graph+loose externals (in raw's dir) → one {out}/{stem}.onnx + {stem}.onnx.data.
    Only the two final files land in `out`; sibling sub-parts are never touched."""
    final = out / f"{stem}.onnx"
    for p in (final, out / f"{stem}.onnx.data"):
        if p.exists(): p.unlink()
    m = onnx.load(str(raw))
    onnx.save_model(m, str(final), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=f"{stem}.onnx.data", size_threshold=1024, convert_attribute=False)
    del m; gc.collect()
    shutil.rmtree(raw.parent, ignore_errors=True)
    return sum(p.stat().st_size for p in out.glob(f"{stem}.onnx*")) / 1e9


def _tmp(stem):
    return Path(tempfile.mkdtemp()) / f"{stem}.onnx"


TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
                   "chat_template.json")


def task_prompts(hf_model):
    """Per-task text prompt convention (from jina's model cards). text-matching is symmetric (both
    sides use 'Query:'); retrieval/code are asymmetric (query 'Query:', document 'Passage:')."""
    if "text-matching" in hf_model:
        return {"query": "Query:", "document": "Query:", "symmetric": True}
    return {"query": "Query:", "document": "Passage:", "symmetric": False}


QUANT_BITS = {"int8": 8, "int4": 4}     # precision → MatMulNBits bit-width (int* → fp16 graph + quant)


def _quant_backbone_inplace(out, bits, block_size):
    """Block-wise n-bit (MatMulNBits) on backbone.onnx MatMul weights, overwriting it in `out`.
    Compute stays float — only the weight matrices are quantized. vision/embeddings stay fp16
    (small, and more precision-sensitive). Returns the new backbone size in GB."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    m = onnx.load(str(out / "backbone.onnx"))
    q = MatMulNBitsQuantizer(m, block_size=block_size, is_symmetric=True, bits=bits)
    q.process()
    for p in (out / "backbone.onnx", out / "backbone.onnx.data"):
        if p.exists(): p.unlink()
    onnx.save_model(q.model.model, str(out / "backbone.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="backbone.onnx.data",
                    size_threshold=1024, convert_attribute=False)
    del m, q; gc.collect()
    return sum(p.stat().st_size for p in out.glob("backbone.onnx*")) / 1e9


def build(model_dir, output, precision, image, image_size, block_size=32):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    bits = QUANT_BITS.get(precision)                       # None for fp16/fp32
    base = "fp32" if precision == "fp32" else "fp16"       # int8/int4 quantize a fp16 graph
    dtype = torch.float16 if base == "fp16" else torch.float32
    npdt = np.float16 if base == "fp16" else np.float32
    print(f"=== build SUB-PARTS | {hf_name(model_dir)} → {out} ({precision}, cpu) ===")
    model = load_model(model_dir, dtype=dtype, attn="eager")

    # image sample (fixed resolution) drives vision + exercises the fusion in embeddings/backbone
    batch = make_image_inputs(model_dir, image, image_size)
    ipos, vmask = image_rope_and_mask(model, batch)
    px = batch["pixel_values"].to(dtype)
    grid = batch["image_grid_thw"]

    sizes = {}
    vsub = VisionSub(model, grid).eval()
    with torch.no_grad():
        vfeat = vsub(px)
    print(f"  vision: pixel_values{tuple(px.shape)} grid{grid[0].tolist()} → image_features{tuple(vfeat.shape)}")
    raw = _tmp("vision")
    _export(vsub, (px,), ["pixel_values"], ["image_features"], {"pixel_values": {0: "num_patches"}}, raw)
    sizes["vision"] = _consolidate(out, "vision", raw)

    esub = EmbeddingsSub(model).eval()
    iid = batch["input_ids"]
    with torch.no_grad():
        emb_ref = esub(iid, vfeat)
    raw = _tmp("embeddings")
    _export(esub, (iid, vfeat), ["input_ids", "image_features"], ["inputs_embeds"],
            {"input_ids": {0: "b", 1: "s"}, "image_features": {0: "n"}, "inputs_embeds": {0: "b", 1: "s"}}, raw)
    sizes["embeddings"] = _consolidate(out, "embeddings", raw)

    bsub = BackboneSub(model).eval()
    iam = batch["attention_mask"]
    with torch.no_grad():
        hid_ref = bsub(emb_ref, iam, ipos)
    raw = _tmp("backbone")
    _export(bsub, (emb_ref, iam, ipos), ["inputs_embeds", "attention_mask", "position_ids"],
            ["last_hidden"], {"inputs_embeds": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                              "position_ids": {2: "s"}, "last_hidden": {0: "b", 1: "s"}}, raw)
    sizes["backbone"] = _consolidate(out, "backbone", raw)

    quantized = {}
    if bits:  # int8/int4: quantize the shipped backbone in place, then sanity-check the real artifact
        sizes["backbone"] = _quant_backbone_inplace(out, bits, block_size)
        quantized["backbone"] = precision

    img_ref = mean_pool(hid_ref, vmask).float().cpu().numpy()
    # the image-prompt layout is FIXED for a given resolution (only pixels vary) → save the host
    # tensors so image inference needs NO model load (just the processor for pixel_values).
    np.savez(out / "image_meta.npz", input_ids=iid.cpu().numpy(), attention_mask=iam.cpu().numpy(),
             position_ids=ipos.cpu().numpy(), vision_mask=vmask.cpu().numpy())
    del model, vsub, esub, bsub; gc.collect()

    for fn in TOKENIZER_FILES:
        s = Path(model_dir) / fn
        if s.exists(): shutil.copy(s, out / fn)
    for stem, gb in sizes.items():
        print(f"  {stem}.onnx (+.data): {gb:.1f} GB")

    # composed sanity: vision → embeddings → backbone → pool  vs  the PyTorch image embedding
    vs = make_session(out / "vision.onnx"); es = make_session(out / "embeddings.onnx")
    bs = make_session(out / "backbone.onnx")
    f = vs.run(None, {"pixel_values": px.float().cpu().numpy().astype(npdt)})[0]
    e = es.run(None, {"input_ids": iid.cpu().numpy(), "image_features": f})[0]
    h = bs.run(None, {"inputs_embeds": e, "attention_mask": iam.cpu().numpy(),
                      "position_ids": ipos.cpu().numpy()})[0]
    c = cosine(mean_pool(h, vmask.cpu().numpy()), img_ref)
    print(f"  composed sanity (image, ORT chain vs PyTorch): cos={c:.6f}")

    man = {"hf_model": hf_name(model_dir), "local_dir": str(Path(model_dir).name),
           "precision": base, "embedding_dim": HIDDEN, "matryoshka_dims": MATRYOSHKA,
           "image_size": image_size, "image_grid_thw": grid[0].tolist(),
           "sub_models": {
               "vision": {"file": "vision.onnx", "in": "pixel_values[N,1176]", "out": "image_features[N,2048]",
                          "task_agnostic": True},
               "embeddings": {"file": "embeddings.onnx", "in": "input_ids[B,S] + image_features[N,2048]",
                              "out": "inputs_embeds[B,S,2048]"},
               "backbone": {"file": "backbone.onnx", "in": "inputs_embeds + attention_mask + position_ids[3,B,S]",
                            "out": "last_hidden[B,S,2048]"}},
           "compose_text": "embeddings(ids,empty) -> backbone -> mean-pool(attn_mask) -> L2norm",
           "compose_image": "vision(px) -> embeddings(ids,feats) -> backbone -> mean-pool(vision-span) -> L2norm",
           "prompts": task_prompts(hf_name(model_dir))}
    if quantized:
        man["quantized"] = quantized     # I/O dtype stays fp16; precision field holds the base
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    # fp16/fp32 must hit parity; int4 is knowingly lossy (embedding drift), so warn instead of fail.
    if c < 0.999:
        if bits and bits <= 4:
            print(f"  [WARN] {precision} composed cos {c:.4f} < 0.999 — expected for int4 (lossy); "
                  f"verify with eval.py before use")
        else:
            raise SystemExit(f"  [FAIL] composed sanity cos {c:.4f} < 0.999")
    print("  build OK")
    return out


def main():
    ap = argparse.ArgumentParser(description="jina-embeddings-v4 → ONNX sub-parts")
    ap.add_argument("--model", default="vllm-retrieval")
    ap.add_argument("--output", default="onnx/cpu_fp16")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "int8", "int4"],
                    help="int8/int4 build a fp16 graph then quantize the backbone in place (block-wise)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--block-size", type=int, default=32, help="int8/int4 quantization block size")
    args = ap.parse_args()
    quiet()
    build(args.model, args.output, args.precision, args.image, args.image_size, args.block_size)


if __name__ == "__main__":
    main()
