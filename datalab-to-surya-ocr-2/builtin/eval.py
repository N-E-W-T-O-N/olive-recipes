"""Evaluate surya-ocr-2 ONNX builds on its real tasks (layout / OCR / table recognition).

Compares one or more built ONNX targets against the PyTorch checkpoint on real document images
(prompts sourced verbatim from datalab-to/surya — see tasks.py; do NOT invent prompts for this
model). Structured tasks (layout, table_rec) are checked for valid, schema-conforming JSON;
all tasks are compared against the PyTorch baseline via text similarity when available.

Usage:
    python eval.py --model model cpu_int4/models cpu_fp32/models
    python eval.py --model model                                   # auto-discovers built targets
    python eval.py --model model --task layout --limit 5 cpu_int4/models
    python eval.py --model model --skip-pytorch cpu_int4/models cpu_fp32/models
    python eval.py --model model --images D:/my-test-docs --task all cpu_int4/models
"""

import argparse
import difflib
import glob
import json
import os
import re
import time

import onnxruntime_genai as og
from PIL import Image

from inference import generate_response, load_id_to_token
from tasks import FULL_PAGE_TASKS, IMAGE_TOKEN_HEADROOM, LAYOUT_LABEL_SET, TABLE_REC_LABEL_SET

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

# The unsuffixed plain document pages shipped in model/assets/. The *_layout / *_reading /
# *_tablerec / *_text siblings are datalab's README reference-OUTPUT visualizations, not inputs —
# using them as input was an earlier mistake in this project; don't repeat it.
DEFAULT_IMAGE_NAMES = ["corporate.png", "excerpt.png", "form.png", "handwritten.png",
                       "newspaper.png", "textbook.png"]

# surya's own README Examples table only shows a "Table Rec" output for Tax Form / Handwritten
# Notes / Corporate Doc — NOT excerpt/newspaper/textbook. Running table_rec on a page with no
# table isn't a meaningful correctness test (the model has nothing to recognize), so restrict
# that task to the pages the model was actually demonstrated on.
TASK_IMAGE_NAMES = {"table_rec": ["corporate.png", "form.png", "handwritten.png"]}


def resolve_images(images_arg: str | None, limit: int, task: str) -> list[str]:
    if images_arg:
        # Custom folder: trust the caller, use every image in it for every task.
        paths = sorted(
            os.path.join(images_arg, f) for f in os.listdir(images_arg)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )
    else:
        default_dir = os.path.join(HERE, "model", "assets")
        names = TASK_IMAGE_NAMES.get(task, DEFAULT_IMAGE_NAMES)
        paths = [os.path.join(default_dir, n) for n in names
                 if os.path.exists(os.path.join(default_dir, n))]
    return paths[:limit] if limit else paths


def resolve_onnx_dirs(explicit: list[str]) -> list[str]:
    """Explicit dirs win; otherwise auto-discover every '*/models' under HERE with a
    genai_config.json (i.e. every target optimize.py has built so far)."""
    if explicit:
        return explicit
    found = sorted(
        os.path.dirname(p) for p in glob.glob(os.path.join(HERE, "*", "models", "genai_config.json"))
    )
    return found


def validate_structured(text: str, task: str) -> tuple[bool, str]:
    """For layout/table_rec: strip code fences, parse as JSON, check it matches surya's schema
    (list of {label, bbox[, count]} dicts; label in the real label set; bbox = 'x0 y0 x1 y1'
    normalized 0-1000). Returns (valid, reason)."""
    if task not in ("layout", "table_rec"):
        return True, "n/a (not a structured task)"
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    if not isinstance(parsed, list):
        return False, f"not a JSON array (got {type(parsed).__name__})"
    label_set = LAYOUT_LABEL_SET if task == "layout" else TABLE_REC_LABEL_SET
    bbox_re = re.compile(r"^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$")
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            return False, f"item {i} is not an object"
        if item.get("label") not in label_set:
            return False, f"item {i} label {item.get('label')!r} not in {task} label set"
        if not bbox_re.match(str(item.get("bbox", ""))):
            return False, f"item {i} bbox {item.get('bbox')!r} doesn't match 'x0 y0 x1 y1'"
    return True, f"{len(parsed)} valid entries"


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def build_pytorch_runner(model_path: str):
    print(f"\nLoading PyTorch model: {model_path}")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")
    pt_model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pt_proc = AutoProcessor.from_pretrained(model_path)
    print(f"  PyTorch model loaded ({type(pt_model).__name__}).")
    return pt_model, pt_proc, device


def run_pytorch(pt_model, pt_proc, device, prompt: str, image_path: str, max_tokens: int):
    """No qwen_vl_utils dependency — the processor accepts a plain PIL image directly."""
    import torch

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text_input = pt_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = pt_proc(text=[text_input], images=[image], return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = pt_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    elapsed = time.perf_counter() - t0

    out_ids = out[0][prompt_len:]
    text = pt_proc.decode(out_ids, skip_special_tokens=True)
    tokens = len(out_ids)
    tps = max(tokens - 1, 1) / max(elapsed, 1e-9)
    return text, tokens, elapsed, tps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate surya-ocr-2 ONNX builds against PyTorch on real OCR/layout/table tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="model",
                        help="PyTorch model path or HF id for the baseline (default: ./model)")
    parser.add_argument("--task", choices=[*FULL_PAGE_TASKS, "all"], default="all",
                        help="Which real surya task(s) to evaluate (default: all)")
    parser.add_argument("--images", default=None,
                        help="Folder of test images (default: model/assets/, curated plain pages)")
    parser.add_argument("--limit", type=int, default=2,
                        help="Max images to evaluate per task (default: 2 — full-page OCR is slow "
                             "on CPU; raise once you trust the setup)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Override max_length for every generation (default: per-task budget "
                             f"+ {IMAGE_TOKEN_HEADROOM} image-token headroom)")
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip the PyTorch baseline (ONNX-only: timing + JSON-schema checks)")
    parser.add_argument("--verbose", action="store_true", help="Print full raw output per run")
    parser.add_argument("onnx_dirs", nargs="*",
                        help="ONNX model dir(s) to evaluate, e.g. cpu_int4/models cpu_fp32/models. "
                             "If omitted, auto-discovers every built '*/models' dir under this project.")
    args = parser.parse_args()

    tasks = list(FULL_PAGE_TASKS) if args.task == "all" else [args.task]
    images_by_task = {task: resolve_images(args.images, args.limit, task) for task in tasks}
    onnx_dirs = resolve_onnx_dirs(args.onnx_dirs)

    if not any(images_by_task.values()):
        parser.error("no test images found (check --images / model/assets/)")
    if not onnx_dirs:
        parser.error("no ONNX model dirs given and none auto-discovered "
                      "(build one first, e.g. `python optimize.py --device cpu --precision int4`)")

    print(f"Tasks     : {tasks}")
    for task in tasks:
        print(f"  {task}: {[os.path.basename(p) for p in images_by_task[task]]}")
    print(f"ONNX dirs : {onnx_dirs}")

    # ---- PyTorch baseline: computed ONCE, reused for every onnx_dir ----
    baseline = {}  # (task, image_path) -> (text, tokens, elapsed, tps)
    if not args.skip_pytorch:
        try:
            pt_model, pt_proc, device = build_pytorch_runner(args.model)
            for task in tasks:
                prompt, budget = FULL_PAGE_TASKS[task]
                max_tokens = args.max_length or budget
                for img in images_by_task[task]:
                    print(f"  [pytorch] {task} / {os.path.basename(img)} ...", end="", flush=True)
                    text, tokens, elapsed, tps = run_pytorch(pt_model, pt_proc, device, prompt, img, max_tokens)
                    baseline[(task, img)] = (text, tokens, elapsed, tps)
                    print(f" {tokens} tok, {elapsed:.1f}s")
            del pt_model
        except Exception as e:
            print(f"\n[WARN] PyTorch baseline unavailable ({e}); continuing ONNX-only.\n")
            baseline = {}

    # ---- ONNX: each dir evaluated against the same images/tasks ----
    all_rows = []  # dicts: onnx_dir, task, image, tokens, ttft, tps, json_valid, json_reason, similarity
    for onnx_dir in onnx_dirs:
        print(f"\n{'=' * 70}\nEvaluating: {onnx_dir}\n{'=' * 70}")
        try:
            model = og.Model(onnx_dir)
            processor = model.create_multimodal_processor()
            tokenizer = og.Tokenizer(model)
            tokenizer_stream = processor.create_stream()
        except Exception as e:
            # e.g. a cuda/webgpu target auto-discovered on a CPU-only box (no matching EP
            # installed) — skip it rather than aborting the whole batch.
            print(f"  [skip] couldn't load {onnx_dir}: {e}")
            continue
        id_to_token = load_id_to_token(onnx_dir)

        for task in tasks:
            prompt, budget = FULL_PAGE_TASKS[task]
            max_length = args.max_length or (budget + IMAGE_TOKEN_HEADROOM)
            for img in images_by_task[task]:
                name = os.path.basename(img)
                try:
                    text, tokens, ttft, tps = generate_response(
                        model, processor, tokenizer, tokenizer_stream, prompt, img, max_length,
                        quiet=True, id_to_token=id_to_token,
                    )
                except Exception as e:
                    print(f"  [{task}/{name}] ERROR: {e}")
                    continue

                valid, reason = validate_structured(text, task)
                sim = None
                if (task, img) in baseline:
                    sim = similarity(text, baseline[(task, img)][0])

                row = {"onnx_dir": onnx_dir, "task": task, "image": name, "tokens": tokens,
                      "ttft_ms": ttft * 1000, "tps": tps, "json_valid": valid,
                      "json_reason": reason, "similarity": sim}
                all_rows.append(row)

                sim_str = f"sim={sim:.2f}" if sim is not None else "sim=n/a"
                valid_str = "valid" if valid else f"INVALID ({reason})"
                print(f"  [{task}/{name}] {tokens} tok  TTFT={ttft*1000:.0f}ms  TPS={tps:.1f}  "
                      f"{valid_str}  {sim_str}")
                if args.verbose:
                    print(f"    -> {text.strip()[:400]!r}")

        del model

    # ---- Summary ----
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for onnx_dir in onnx_dirs:
        rows = [r for r in all_rows if r["onnx_dir"] == onnx_dir]
        if not rows:
            print(f"\n  {onnx_dir}: no successful runs")
            continue
        n = len(rows)
        n_valid = sum(1 for r in rows if r["json_valid"])
        sims = [r["similarity"] for r in rows if r["similarity"] is not None]
        avg_tps = sum(r["tps"] for r in rows) / n
        avg_ttft = sum(r["ttft_ms"] for r in rows) / n
        print(f"\n  {onnx_dir}  ({n} runs)")
        print(f"    Avg TPS         : {avg_tps:.1f} tokens/sec")
        print(f"    Avg TTFT        : {avg_ttft:.0f} ms")
        print(f"    JSON valid      : {n_valid}/{n} (layout/table_rec only)")
        if sims:
            print(f"    Avg similarity  : {sum(sims) / len(sims):.3f} vs PyTorch baseline ({len(sims)} runs)")

    if len(onnx_dirs) > 1:
        print(f"\n  Cross-target note: compare 'Avg similarity' and 'JSON valid' rates above to see "
              f"whether a lower-precision target (e.g. int4) degrades output vs a higher one (fp32).")


if __name__ == "__main__":
    main()
