import argparse
import json
import onnxruntime_genai as og


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for datalab-to/surya-ocr-2"
    )
    parser.add_argument("--model_path", type=str, default="cpu_and_mobile/models",
                        help="Path to the model directory containing genai_config.json and ONNX models")
    parser.add_argument("--image", type=str, default=None, help="Path to image/document file")
    parser.add_argument("--prompt", type=str, default=None, help="OCR or document analysis prompt")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer, tokenizer_stream)
    elif args.prompt:
        generate_response(model, processor, tokenizer, tokenizer_stream, args.prompt, args.image)
    else:
        # Default OCR prompt for document images
        default_prompt = "OCR the full text of this document."
        if args.image:
            generate_response(model, processor, tokenizer, tokenizer_stream, default_prompt, args.image)
        else:
            print("Please provide --prompt and/or --image, or use --interactive mode")
            parser.print_help()


def generate_response(model, processor, tokenizer, tokenizer_stream, prompt, image_path):
    images = None
    if image_path:
        print(f"Loading image: {image_path}")
        images = og.Images.open(image_path)
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]

    full_prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)
    print(f"\nPrompt: {prompt}")
    if image_path:
        print(f"Image:  {image_path}")
    print("\nGenerating response...")

    inputs = processor(full_prompt, images=images)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=4096)
    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    print("\nResponse: ", end="", flush=True)
    while not generator.is_done():
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end="", flush=True)
    print()
    del generator


def interactive_mode(model, processor, tokenizer, tokenizer_stream):
    print("\n" + "=" * 55)
    print("Surya OCR-2 Interactive Mode")
    print("Enter 'quit' or 'exit' to stop")
    print("To include an image: image:/path/to/doc.png  <prompt>")
    print("Default prompt when only image given: 'OCR the full text of this document.'")
    print("=" * 55 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            print("Please enter a prompt or image path.")
            continue

        image_path = None
        prompt = user_input
        if user_input.startswith("image:"):
            parts = user_input.split(" ", 1)
            image_path = parts[0][6:]
            prompt = parts[1] if len(parts) > 1 else "OCR the full text of this document."

        try:
            generate_response(model, processor, tokenizer, tokenizer_stream, prompt, image_path)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        print("-" * 55 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
