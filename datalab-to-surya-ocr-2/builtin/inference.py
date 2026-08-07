"""ONNX Runtime GenAI inference for surya-ocr-2 (Qwen3.5-VL OCR).

Usage:
    python inference.py --image model/assets/corporate.png --task ocr
    python inference.py --image model/assets/corporate.png --task layout
    python inference.py --image model/assets/form.png --task table_rec
    python inference.py --image photo.jpg --prompt "custom prompt overrides --task"
    python inference.py --interactive
    python inference.py --benchmark model/assets --task layout --verbose
    python inference.py --benchmark model/assets --task ocr --pytorch_model datalab-to/surya-ocr-2
    python inference.py --model_path cpu_fp32/models --task ocr --image model/assets/form.png

--task selects one of surya's REAL trained prompts (see tasks.py — sourced verbatim from
datalab-to/surya's surya/inference/prompts.py; do not paraphrase). --prompt overrides it.
"""

import argparse
import json
import os
import time

import onnxruntime_genai as og

from tasks import FULL_PAGE_TASKS, IMAGE_TOKEN_HEADROOM

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


def load_id_to_token(onnx_dir: str) -> dict:
    """Build an id->token-string map straight from the built tokenizer.json, as a fallback
    decoder for a confirmed genai bug: id 10147's vocab entry is the literal Unicode replacement
    character U+FFFD ('\\ufffd') — present in the ORIGINAL upstream tokenizer.json, not introduced
    by this recipe's WordLevel->BPE conversion — and genai's C++ tokenizer fails to index that one
    entry into its internal reverse map, so ANY decode of a generated token with that id raises
    'invalid map<K, T> key' (confirmed via isolated repro: even `tokenizer.decode([10147])` alone
    fails). Only surfaces on long/varied real text (the full-page 'ocr' task) since layout/
    table_rec never generate it. added_tokens override the base vocab for overlapping ids, same
    resolution order tokenizers.js/HF use."""
    path = os.path.join(onnx_dir, "tokenizer.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        tk = json.load(f)
    id_to_token = {v: k for k, v in tk.get("model", {}).get("vocab", {}).items()}
    for t in tk.get("added_tokens", []):
        id_to_token[t["id"]] = t["content"]
    return id_to_token


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for VL models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="cpu_int4/models",
        help="Path to the model directory containing genai_config.json and ONNX models",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file",
    )
    parser.add_argument(
        "--task",
        choices=list(FULL_PAGE_TASKS),
        default=None,
        help="Use one of surya's real trained prompts (see tasks.py). Ignored if --prompt is set.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom text prompt (overrides --task)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum total tokens (prompt + generated). Default: task's suggested budget "
             f"+ {IMAGE_TOKEN_HEADROOM} for image tokens, or 4096 if neither --task nor --prompt "
             "implies one.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Path to a folder of images. Runs inference on each and reports avg TPS/TTFT.",
    )
    parser.add_argument(
        "--benchmark_prompt",
        type=str,
        default=None,
        help="Custom prompt for benchmark mode (overrides --task). Default: --task's prompt, "
             "or the real 'ocr' (full-page) prompt if neither is set.",
    )
    parser.add_argument(
        "--pytorch_model",
        type=str,
        default=None,
        help="HuggingFace model ID for PyTorch comparison (e.g. datalab-to/surya-ocr-2)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generated text in benchmark mode",
    )
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()
    id_to_token = load_id_to_token(args.model_path)

    if args.benchmark:
        benchmark_folder(model, processor, tokenizer, tokenizer_stream, args, id_to_token)
    elif args.interactive:
        interactive_mode(model, processor, tokenizer, tokenizer_stream, args, id_to_token)
    elif args.prompt or args.task:
        prompt, max_length = _resolve_prompt(args.prompt, args.task, args.max_length)
        generate_response(model, processor, tokenizer, tokenizer_stream, prompt, args.image, max_length,
                          id_to_token=id_to_token)
    else:
        print("Please provide --prompt/--task, --interactive, or --benchmark <folder>")
        parser.print_help()


def _resolve_prompt(prompt: str | None, task: str | None, max_length: int | None):
    """--prompt wins if set; else --task's real prompt; else a generic fallback.
    max_length: explicit value wins; else task's suggested budget + image-token headroom."""
    if prompt:
        return prompt, max_length or 4096
    if task:
        task_prompt, task_tokens = FULL_PAGE_TASKS[task]
        return task_prompt, max_length or (task_tokens + IMAGE_TOKEN_HEADROOM)
    return "Describe this image in detail.", max_length or 4096


def generate_response(model, processor, tokenizer, tokenizer_stream, prompt, image_path,
                      max_length=4096, quiet=False, id_to_token=None):
    """Run a single generation. Returns (text, token_count, ttft, tps).

    id_to_token: optional {token_id: str} fallback (see load_id_to_token) used ONLY when genai's
    own decode raises on a specific token — currently known to happen for id 10147 (U+FFFD) on
    long real-world text. Pass None to skip the fallback (decode failures then propagate as-is)."""
    t_preproc_start = time.perf_counter()
    images = None
    if image_path:
        if not quiet:
            print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    full_prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)

    if not quiet:
        print(f"\nPrompt: {prompt}")
        if image_path:
            print(f"Image: {image_path}")
        print("\nGenerating response...")

    inputs = processor(full_prompt, images=images)

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)
    t_preproc = time.perf_counter() - t_preproc_start
    t_start = time.perf_counter()

    token_count = 0
    ttft = None
    tokens = []
    pieces = []  # per-token tokenizer_stream.decode() output — see the fallback note below

    if not quiet:
        print("\nResponse: ", end="", flush=True)
    while not generator.is_done():
        try:
            generator.generate_next_token()
        except Exception as e:
            tail = "".join(pieces[-30:])
            raise RuntimeError(
                f"{e} (after {token_count} tokens; last ~30 decoded: {tail!r})"
            ) from e
        if ttft is None:
            ttft = time.perf_counter() - t_start
        token_count += 1
        new_token = generator.get_next_tokens()[0]
        tokens.append(new_token)
        try:
            piece = tokenizer_stream.decode(new_token)
        except Exception:
            if id_to_token is None or int(new_token) not in id_to_token:
                raise
            piece = id_to_token[int(new_token)]
        pieces.append(piece)
        if not quiet:
            print(piece, end="", flush=True)

    t_model = time.perf_counter() - t_start
    if not quiet:
        print()

    # ROOT CAUSE (confirmed via isolated repro, 2026-07-30): token id 10147's vocab entry is the
    # literal Unicode replacement character U+FFFD ('�') — present in the ORIGINAL upstream
    # tokenizer.json, not introduced by this recipe's WordLevel->BPE conversion. genai's C++
    # tokenizer fails to index that one entry into its internal reverse map, so ANY decode of a
    # generated token with that id raises "invalid map<K, T> key" — confirmed to fail identically
    # whether hit via tokenizer_stream.decode() mid-loop, tokenizer.decode() in bulk, or (having
    # never been decoded at all) apparently during Generator teardown. Only surfaces on long,
    # varied real text (the full-page "ocr" task can legitimately emit U+FFFD for a
    # low-confidence/unrecognized glyph); layout/table_rec's short structured JSON never does.
    # id_to_token (see load_id_to_token) is the same fallback used in the per-token loop above.
    try:
        del generator
    except Exception as e:
        if id_to_token is None:
            raise
        print(f"\n[warn] generator teardown raised (known genai bug, id 10147/U+FFFD): {e}")

    try:
        text = tokenizer.decode(tokens)
    except Exception as e:
        if id_to_token is None:
            raise
        # Rebuild from the incrementally-decoded pieces instead (each already used the same
        # per-token fallback above, so this is a faithful reconstruction, not a guess).
        text = "".join(pieces)

    decode_tokens = max(token_count - 1, 1)
    decode_time = t_model - (ttft or 0)
    tps = decode_tokens / decode_time if decode_time > 0 else 0

    if not quiet:
        print(f"\n  Tokens generated : {token_count}")
        print(f"  Preprocess       : {t_preproc * 1000:.1f} ms  (image/template/processor; excluded from TTFT)")
        print(f"  TTFT (model)     : {ttft * 1000:.1f} ms")
        print(f"  Decode TPS       : {tps:.1f} tokens/sec")
        print(f"  Model time       : {t_model:.2f} s")
        print(f"  End-to-end time  : {t_preproc + t_model:.2f} s")

    return text, token_count, ttft or 0, tps


def benchmark_folder(model, processor, tokenizer, tokenizer_stream, args, id_to_token=None):
    """Run inference on every image in a folder and report avg TPS/TTFT."""
    folder = args.benchmark
    prompt, max_length = _resolve_prompt(args.benchmark_prompt, args.task, args.max_length)
    verbose = args.verbose

    image_files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in {folder}")
        return

    print(f"\nBenchmark: {len(image_files)} images from {folder}")
    print(f"Prompt   : {prompt}")
    print(f"{'=' * 70}")

    onnx_results = _run_benchmark(
        image_files, prompt, max_length, verbose,
        lambda img, p, ml: generate_response(model, processor, tokenizer, tokenizer_stream, p, img, ml,
                                             quiet=not verbose, id_to_token=id_to_token),
        label="ONNX",
    )

    pt_results = None
    if args.pytorch_model:
        pt_model, pt_proc, device = _build_pytorch_runner(args.pytorch_model)
        pt_results = _run_benchmark(
            image_files, prompt, max_length, verbose,
            lambda img, p, ml: _run_pytorch(pt_model, pt_proc, device, p, img, ml),
            label="PyTorch",
        )

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK SUMMARY ({len(image_files)} images)")
    print(f"{'=' * 70}")
    _print_summary(onnx_results, "ONNX")
    if pt_results:
        _print_summary(pt_results, f"PyTorch ({args.pytorch_model})")

    avg_onnx_tps = sum(onnx_results["tps"]) / len(onnx_results["tps"]) if onnx_results["tps"] else 0
    if pt_results and pt_results["tps"] and avg_onnx_tps:
        avg_pt_tps = sum(pt_results["tps"]) / len(pt_results["tps"])
        print(f"\n  ONNX / PyTorch TPS speedup   : {avg_onnx_tps / max(avg_pt_tps, 1e-9):.2f}x")


def _run_benchmark(image_files, prompt, max_length, verbose, run_fn, label=""):
    """Run a generate function over all images, collect metrics."""
    all_ttft, all_tps, all_tokens = [], [], []

    if label:
        print(f"\n--- {label} ---")

    for i, img_path in enumerate(image_files):
        print(f"\n[{i + 1}/{len(image_files)}] {os.path.basename(img_path)}")
        try:
            text, token_count, ttft, tps = run_fn(img_path, prompt, max_length)
            all_ttft.append(ttft)
            all_tps.append(tps)
            all_tokens.append(token_count)
            print(f"  tokens={token_count}  TTFT={ttft * 1000:.1f}ms  TPS={tps:.1f}")
            if verbose:
                display = text.strip()[:500]
                print(f"  Output: {display}{'...' if len(text.strip()) > 500 else ''}")
        except Exception as e:
            print(f"  ERROR: {e}")

    return {"ttft": all_ttft, "tps": all_tps, "tokens": all_tokens}


def _print_summary(results, label):
    """Print avg/min/max for a set of benchmark results."""
    tps_list = results["tps"]
    ttft_list = results["ttft"]
    tokens_list = results["tokens"]
    if not tps_list:
        print(f"\n  {label}: no successful runs")
        return
    n = len(tps_list)
    print(f"\n  {label} ({n} images):")
    print(f"    Avg TTFT         : {sum(ttft_list) / n * 1000:.1f} ms")
    print(f"    Avg Decode TPS   : {sum(tps_list) / n:.1f} tokens/sec")
    print(f"    Avg tokens/image : {sum(tokens_list) / n:.0f}")
    print(f"    Min / Max TPS    : {min(tps_list):.1f} / {max(tps_list):.1f}")
    print(f"    Min / Max TTFT   : {min(ttft_list) * 1000:.1f} / {max(ttft_list) * 1000:.1f} ms")


def _build_pytorch_runner(model_id: str):
    """Load a HuggingFace VL model for comparison."""
    print(f"\nLoading PyTorch model: {model_id}")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Device: {device}, dtype: {dtype}")

    pt_model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pt_proc = AutoProcessor.from_pretrained(model_id)
    print(f"  PyTorch model loaded ({type(pt_model).__name__}).")
    return pt_model, pt_proc, device


def _run_pytorch(pt_model, pt_proc, device, prompt, image_path, max_length):
    """Run PyTorch generation on one image. Returns (text, token_count, ttft, tps).

    No qwen_vl_utils dependency — the processor accepts a plain PIL image directly (verified:
    a single-image chat-template prompt round-trips through AutoProcessor with no vision_info
    helper needed)."""
    import torch
    from PIL import Image as PILImage

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }
    ]
    text_input = pt_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = PILImage.open(image_path).convert("RGB")
    inputs = pt_proc(text=[text_input], images=[image], return_tensors="pt").to(device)

    prompt_len = inputs["input_ids"].shape[-1]

    t_start = time.perf_counter()
    with torch.no_grad():
        out = pt_model.generate(**inputs, max_new_tokens=max_length, do_sample=False)
    t_total = time.perf_counter() - t_start

    out_ids = out[0][prompt_len:]
    token_count = len(out_ids)
    text = pt_proc.decode(out_ids, skip_special_tokens=True)

    # PyTorch doesn't expose per-token TTFT, approximate
    ttft = t_total / max(token_count, 1)
    tps = max(token_count - 1, 1) / max(t_total - ttft, 1e-9)

    return text, token_count, ttft, tps


def interactive_mode(model, processor, tokenizer, tokenizer_stream, args, id_to_token=None):
    """Run in interactive mode with text and optional image inputs."""
    print("\n" + "=" * 50)
    print("Interactive Mode - Enter 'quit' or 'exit' to stop")
    print("To include an image, type: image:/path/to/image.jpg your prompt")
    print("=" * 50 + "\n")

    max_length = args.max_length or 4096  # args.max_length defaults to None (task-driven elsewhere)

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            print("Please enter a prompt.")
            continue

        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]
            prompt = parts[1] if len(parts) > 1 else "Describe this image"

        try:
            generate_response(
                model, processor, tokenizer, tokenizer_stream,
                prompt, image_path, max_length, id_to_token=id_to_token,
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
