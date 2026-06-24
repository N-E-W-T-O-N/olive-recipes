# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3", "torch", "torchvision", "torchaudio",
#   "onnx", "onnxruntime>=1.20", "numpy", "safetensors",
#   "huggingface_hub", "accelerate", "librosa", "soundfile",
# ]
# ///
"""Parity: ONNX inference.py generate() vs the PyTorch reference (greedy).

Compares, on identical text/instruct, greedy (do_sample=False):
  (1) tokenization  — reference processor ids  vs  inference.py AutoTokenizer ids
  (2) first-codebook code sequence — reference talker codes vs ONNX codes
      (frame count, first divergence step, % match up to min length)

Greedy isolates correctness from sampling RNG. Small fp diffs can still flip an
argmax and diverge late; report first-divergence step so a late split (good
prefill) is distinguishable from step-0 (broken prefill).

Usage:
  uv run eval_generate.py --model-path onnx/customvoice/cpu_fp32 --tts-dir voicedesign \
      --text "Hello, this is a test." --instruct "A calm female voice." --max-new-tokens 60
"""
import argparse, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "codes"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tts-dir", default="voicedesign")
    ap.add_argument("--text", default="Hello, this is a test.")
    ap.add_argument("--instruct", default="A calm female voice.")
    ap.add_argument("--language", default="Auto")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    import torch
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from inference import Pipeline

    # ── reference (PyTorch) ──────────────────────────────────────────────────
    ref = Qwen3TTSModel.from_pretrained(args.tts_dir)
    assistant = f"<|im_start|>assistant\n{args.text}<|im_end|>\n<|im_start|>assistant\n"
    ref_ids = ref._tokenize_texts([assistant])[0]           # [1,1,L] or [1,L]
    ref_ids_flat = ref_ids.reshape(-1).tolist()

    input_ids = ref._tokenize_texts([ref._build_assistant_text(args.text)])
    instruct_ids = [ref._tokenize_texts([ref._build_instruct_text(args.instruct)])[0]]
    with torch.no_grad():
        codes_list, _ = ref.model.generate(
            input_ids=input_ids, instruct_ids=instruct_ids, languages=[args.language],
            non_streaming_mode=True, do_sample=False, subtalker_dosample=False,
            max_new_tokens=args.max_new_tokens, output_hidden_states=True,
            return_dict_in_generate=True)
    ref_codes = codes_list[0].cpu().numpy().astype(np.int64)   # [T,16]
    ref_first = ref_codes[:, 0]

    # ── ONNX (inference.py) ──────────────────────────────────────────────────
    pipe = Pipeline(args.model_path, tts_dir=args.tts_dir)
    onnx_ids = pipe._ids(assistant).reshape(-1).tolist()
    onnx_codes = pipe.generate(
        args.text, language=args.language, instruct=args.instruct,
        max_new_tokens=args.max_new_tokens, do_sample=False, sub_do_sample=False,
        verbose=False)
    onnx_first = onnx_codes[:, 0]

    # ── report ───────────────────────────────────────────────────────────────
    print("\n=== tokenization ===")
    print(f"  ref  ids ({len(ref_ids_flat)}): {ref_ids_flat}")
    print(f"  onnx ids ({len(onnx_ids)}): {onnx_ids}")
    print(f"  MATCH: {ref_ids_flat == onnx_ids}")

    print("\n=== first-codebook codes (greedy) ===")
    print(f"  ref  frames={len(ref_first)}  onnx frames={len(onnx_first)}")
    n = min(len(ref_first), len(onnx_first))
    if n:
        eq = ref_first[:n] == onnx_first[:n]
        div = int(np.argmax(~eq)) if not eq.all() else n
        print(f"  match up to min-len: {int(eq.sum())}/{n} ({100*eq.mean():.1f}%)")
        print(f"  first divergence step: {div}{' (none)' if div==n else ''}")
        print(f"  ref [:12]: {ref_first[:12].tolist()}")
        print(f"  onnx[:12]: {onnx_first[:12].tolist()}")

    print("\n=== ALL 16 codebooks (greedy) ===")
    m = min(len(ref_codes), len(onnx_codes))
    if m:
        full_eq = (ref_codes[:m] == onnx_codes[:m])
        print(f"  full-grid match: {int(full_eq.sum())}/{full_eq.size} "
              f"({100*full_eq.mean():.2f}%) over {m} frames × 16 groups")


if __name__ == "__main__":
    main()
