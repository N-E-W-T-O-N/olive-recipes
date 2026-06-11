"""Evaluate the exported Higgs-Audio-v3 sub-parts (ONNX) vs the original PyTorch.

--model-path points at an onnx/{device}_{precision} dir (same as inference.py).
Reference = that dir's qwen3_standalone, loaded as Qwen3ForCausalLM in fp32.

(a) Parity: ONNX `hidden_states` vs PyTorch `last_hidden_state` for the same
    inputs_embeds — cosine, max-abs-diff, next-token argmax agreement (tied head).
(b) Task: greedy continuation prefix-agreement, ONNX vs PyTorch, over sample prompts.

Audio-side WER eval needs the audio sub-parts (pending — STATUS.md).

Usage:
  python eval.py --model-path onnx/cpu_int4
"""
import argparse
from pathlib import Path

import numpy as np

from inference import Pipeline

PROMPTS = [
    "The future of on-device speech AI is",
    "In a quiet village by the sea,",
    "Large language models can",
]


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="onnx/{device}_{precision} dir")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    pipe = Pipeline(args.model_path)
    ref_dir = str(pipe.standalone_dir())
    print(f"Loading PyTorch reference Qwen3 from {ref_dir} ...")
    ref = AutoModelForCausalLM.from_pretrained(ref_dir, dtype=torch.float32).eval()

    print("\n=== (a) Hidden-state parity (ONNX vs PyTorch) ===")
    cos_all, maxd_all, hits, tot = [], [], 0, 0
    for p in PROMPTS:
        ids = pipe.tok(p, return_tensors="np").input_ids.astype(np.int64)
        onnx_h = pipe.hidden_states(ids)
        with torch.no_grad():
            emb = torch.from_numpy(pipe.embed[ids])
            torch_h = ref.model(inputs_embeds=emb).last_hidden_state.numpy()
        cos_all.append(cosine(onnx_h, torch_h)); maxd_all.append(float(np.abs(onnx_h - torch_h).max()))
        ot = int((onnx_h[:, -1] @ pipe.embed.T)[0].argmax())
        tt = int((torch_h[:, -1] @ pipe.embed.T)[0].argmax())
        hits += int(ot == tt); tot += 1
        print(f"  {p!r}: cos={cos_all[-1]:.4f}  max|d|={maxd_all[-1]:.3f}  "
              f"next-tok {'match' if ot==tt else 'DIFFER'}")
    print(f"  mean cosine={np.mean(cos_all):.4f}  mean max|d|={np.mean(maxd_all):.3f}  "
          f"next-tok agreement={hits}/{tot}")

    print("\n=== (b) Greedy text prefix agreement (ONNX vs PyTorch) ===")
    tot_agree, tot_len = 0, 0
    for p in PROMPTS:
        ids0 = pipe.tok(p, return_tensors="np").input_ids.astype(np.int64)
        ids = ids0.copy(); onnx_new = []
        for _ in range(args.max_new_tokens):
            nxt = int(pipe.logits_last(ids)[0].argmax())
            onnx_new.append(nxt); ids = np.concatenate([ids, [[nxt]]], axis=1)
        with torch.no_grad():
            gen = ref.generate(torch.from_numpy(ids0), max_new_tokens=args.max_new_tokens, do_sample=False)
        torch_new = gen[0, ids0.shape[1]:].tolist()
        agree = 0
        for x, y in zip(onnx_new, torch_new):
            if x != y:
                break
            agree += 1
        tot_agree += agree; tot_len += min(len(onnx_new), len(torch_new))
        print(f"  {p!r}: prefix agreement {agree}/{min(len(onnx_new), len(torch_new))}")
    print(f"\nOverall greedy-prefix agreement: {tot_agree}/{tot_len} "
          f"({100*tot_agree/max(tot_len,1):.1f}%)")


if __name__ == "__main__":
    main()
