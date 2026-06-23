import argparse
import json
import re
from pathlib import Path

import onnxruntime_genai as og


def describe_model(model_path: str):
    """Infer (device, precision, execution_provider) from the model dir alone.

    device/precision come from the `<device>_<precision>` folder name (cpu_int4,
    cuda_fp16, …); the EP is read from genai_config.json's provider_options (and, if
    the folder name is non-standard, precision is sniffed from text.onnx)."""
    p = Path(model_path)
    device = precision = None
    for part in [p.name] + [a.name for a in p.parents]:
        m = re.fullmatch(r"(cpu|cuda)_(int4|fp16|fp32)", part)
        if m:
            device, precision = m.group(1), m.group(2)
            break

    ep = "CPUExecutionProvider"
    cfg_file = p / "genai_config.json"
    if cfg_file.exists():
        txt = cfg_file.read_text()
        ep = "CUDAExecutionProvider" if '"cuda"' in txt else "CPUExecutionProvider"
    if device is None:
        device = "cuda" if ep == "CUDAExecutionProvider" else "cpu"

    if precision is None:  # fall back to sniffing the text decoder weights
        try:
            import onnx
            t = onnx.load(str(p / "text.onnx"), load_external_data=False)
            ops = {n.op_type for n in t.graph.node}
            if "MatMulNBits" in ops or "GatherBlockQuantized" in ops:
                precision = "int4"
            elif any(i.data_type == onnx.TensorProto.FLOAT16 for i in t.graph.initializer):
                precision = "fp16"
            else:
                precision = "fp32"
        except Exception:
            precision = "unknown"
    return device, precision, ep


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for Qwen3-VL"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="cpu_int4/models",
        help="Model dir with genai_config.json + ONNX models, e.g. cpu_int4/models, "
             "cpu_fp16/models, cpu_fp32/models, cuda_fp16/models, cuda_fp32/models"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Load model
    device, precision, ep = describe_model(args.model_path)
    print(f"Loading model from: {args.model_path}")
    print(f"  type: {device}  |  precision: {precision}  |  execution provider: {ep}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer, tokenizer_stream, args)
    elif args.prompt:
        generate_response(model, processor, tokenizer, tokenizer_stream, args.prompt, args.image)
    else:
        print("Please provide --prompt or use --interactive mode")
        parser.print_help()


def generate_response(model, processor, tokenizer, tokenizer_stream, prompt, image_path):
    # Build messages for chat template
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        # Message with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    else:
        # Text-only message
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

    # Apply chat template (requires JSON string)
    full_prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)

    print(f"\nPrompt: {prompt}")
    if image_path:
        print(f"Image: {image_path}")
    print("\nGenerating response...")

    # Process inputs
    inputs = processor(full_prompt, images=images)

    # Set up generation parameters
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)

    # Generate
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print("\nResponse: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)
    print()
    del generator


def interactive_mode(model, processor, tokenizer, tokenizer_stream, args):
    """Run in interactive mode."""
    print("\n" + "="*50)
    print("Interactive Mode - Enter 'quit' or 'exit' to stop")
    print("To include an image, type: image:/path/to/image.jpg")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            print("Please enter a prompt.")
            continue

        # Check for image path
        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]  # Remove "image:" prefix
            prompt = parts[1] if len(parts) > 1 else "Describe this image"

        try:
            generate_response(
                model, processor, tokenizer, tokenizer_stream,
                prompt, image_path
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        print("-"*50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
