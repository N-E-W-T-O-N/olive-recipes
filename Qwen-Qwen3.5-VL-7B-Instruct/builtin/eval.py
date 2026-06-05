"""Evaluate quantized ONNX vs PyTorch unsloth/Qwen3.5-0.8B on AI2D.

AI2D is a multiple-choice visual QA benchmark on scientific diagrams.

Usage:
    # ONNX only (fastest)
    python eval.py

    # ONNX + PyTorch side-by-side
    python eval.py --pytorch_model unsloth/Qwen3.5-0.8B

    # CUDA models
    python eval.py --model_path cuda/models --num_samples 200
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import onnxruntime_genai as og
from datasets import load_dataset
from PIL import Image

NUMBERS = ["1", "2", "3", "4"]
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise multiple-choice answering assistant. "
    "When given a question with numbered options, respond with ONLY a single digit (1, 2, 3, or 4). "
    "Do not include any explanation, reasoning, or other text — just the digit."
)


def build_messages(question: str, options: list, system_prompt: str = "") -> str:
    option_text = "\n".join(f"{N}. {o}" for N, o in zip(NUMBERS, options))
    content = (
        f"Look at the diagram and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\nOptions:\n{option_text}\n\nReply with the number only (1, 2, 3, or 4)."
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": content}]})
    return json.dumps(messages)


def parse_answer(text: str):
    text = text.strip()
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return m.group(1)
    for ch in text:
        if ch in NUMBERS:
            return ch
    return None


def ground_truth_number(sample: dict):
    answer = sample.get("answer", "")
    try:
        idx = int(answer)
        if 0 <= idx < 4:
            return NUMBERS[idx]
    except (ValueError, TypeError):
        pass
    return None


def pil_from_sample(sample: dict):
    img = sample.get("image")
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, bytes):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    return None


def load_ai2d(num_samples: int):
    print(f"Loading AI2D dataset ({num_samples} samples)...")
    ds = load_dataset("lmms-lab/ai2d", split="test")
    ds = ds.select(range(min(num_samples, len(ds))))
    print(f"  Loaded {len(ds)} samples.")
    return ds


def build_onnx_runner(model_path: str):
    print(f"\nLoading ONNX model from: {model_path}")
    model = og.Model(model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    print("  ONNX model loaded.")
    return model, processor, tokenizer


def run_onnx(model, processor, tokenizer, pil_image: Image.Image, messages_json: str) -> str:
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_image.save(f, format="PNG")
        tmp_path = f.name
    try:
        images = og.Images.open(tmp_path)
        prompt = tokenizer.apply_chat_template(messages_json, add_generation_prompt=True)
        inputs = processor(prompt, images=images)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=3000, do_sample=False)
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)
        tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            tokens.append(generator.get_next_tokens()[0])
        del generator
        return tokenizer.decode(tokens)
    finally:
        os.unlink(tmp_path)


def build_pytorch_runner(model_id: str):
    print(f"\nLoading PyTorch model: {model_id}")
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")
    pt_model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pt_proc = AutoProcessor.from_pretrained(model_id)
    print("  PyTorch model loaded.")
    return pt_model, pt_proc, device


def run_pytorch(pt_model, pt_proc, pil_image, question, options, device, system_prompt=""):
    import torch
    option_text = "\n".join(f"{N}. {o}" for N, o in zip(NUMBERS, options))
    content = (
        f"Look at the diagram and answer the multiple-choice question.\n\n"
        f"Question: {question}\n\nOptions:\n{option_text}\n\nReply with the number only (1, 2, 3, or 4)."
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": content}]})
    text = pt_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pt_proc(text=[text], images=[pil_image], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = pt_model.generate(**inputs, max_new_tokens=8, do_sample=False)
    out_ids = out[0][inputs["input_ids"].shape[-1]:]
    return pt_proc.decode(out_ids, skip_special_tokens=True)


def evaluate(dataset, runner_fn, label: str) -> dict:
    correct, skipped = 0, 0
    total = len(dataset)
    latencies = []
    print(f"\n{'='*60}\n  Evaluating: {label}  ({total} samples)\n{'='*60}")

    for i, sample in enumerate(dataset):
        gt = ground_truth_number(sample)
        pil_image = pil_from_sample(sample)
        question = sample.get("question", "")
        options = sample.get("options", [])
        if gt is None or pil_image is None or len(options) < 2:
            skipped += 1
            continue
        try:
            t0 = time.perf_counter()
            raw = runner_fn(pil_image, question, options)
            latencies.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"  [WARN] sample {i}: {e}")
            skipped += 1
            continue
        pred = parse_answer(raw)
        hit = pred == gt
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:4d}/{total}] gt={gt} pred={pred}  {'OK' if hit else 'X'}  "
                  f"running_acc={correct/(i+1-skipped+1e-9):.3f}")
        if hit:
            correct += 1

    evaluated = total - skipped
    accuracy = correct / evaluated if evaluated > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\n  {label}: {correct}/{evaluated}  accuracy={accuracy:.4f}  avg_lat={avg_lat:.2f}s")
    return {"label": label, "accuracy": accuracy, "correct": correct,
            "evaluated": evaluated, "avg_latency_s": avg_lat, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="Eval ONNX vs PyTorch unsloth/Qwen3.5-0.8B on AI2D")
    parser.add_argument("--model_path", default="cpu_and_mobile/models")
    parser.add_argument("--pytorch_model", default=None,
                        help="HuggingFace model ID, e.g. unsloth/Qwen3.5-0.8B")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--skip_onnx", action="store_true")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    args = parser.parse_args()

    ds = load_ai2d(args.num_samples)
    results = []
    sys_prompt = args.system_prompt

    if not args.skip_onnx:
        onnx_model, onnx_proc, onnx_tok = build_onnx_runner(args.model_path)
        def onnx_runner(pil_image, question, options):
            return run_onnx(onnx_model, onnx_proc, onnx_tok, pil_image,
                            build_messages(question, options, sys_prompt))
        results.append(evaluate(ds, onnx_runner, f"ONNX INT4 @ {args.model_path}"))

    if args.pytorch_model:
        pt_model, pt_proc, device = build_pytorch_runner(args.pytorch_model)
        def pt_runner(pil_image, question, options):
            return run_pytorch(pt_model, pt_proc, pil_image, question, options, device, sys_prompt)
        results.append(evaluate(ds, pt_runner, f"PyTorch @ {args.pytorch_model}"))

    print(f"\n{'='*60}\n  EVALUATION SUMMARY\n{'='*60}")
    for r in results:
        print(f"  {r['label']}")
        print(f"    Accuracy: {r['accuracy']*100:.2f}%  ({r['correct']}/{r['evaluated']})")
        print(f"    Avg lat : {r['avg_latency_s']:.2f}s/sample\n")

    if len(results) == 2:
        delta = results[1]["accuracy"] - results[0]["accuracy"]
        print(f"  Accuracy delta (PyTorch - ONNX): {delta*100:+.2f} pp")


if __name__ == "__main__":
    main()
