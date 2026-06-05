import argparse
import json
import os
 
 

import onnxruntime_genai as og


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI inference for unsloth/Qwen3.5-0.8B"
    )
    parser.add_argument("--model_path", type=str, default="cpu_and_mobile/models",
                        help="Path to the model directory containing genai_config.json and ONNX models")
    parser.add_argument("--image", type=str, default=None, help="Path to image file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    config = og.Config(f"{args.model_path}")

   # config.clear_providers()
    #config.append_provider("OpenVINOExecutionProvider")
    model = og.Model(config)
    #model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = processor.create_stream()

    if args.interactive:
        interactive_mode(model, processor, tokenizer, tokenizer_stream)
    elif args.prompt:
        generate_response(model, processor, tokenizer, tokenizer_stream, args.prompt, args.image)
    else:
        print("Please provide --prompt or use --interactive mode")
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
        print(f"Image: {image_path}")
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
    print("\n" + "=" * 50)
    print("Interactive Mode — Enter 'quit' or 'exit' to stop")
    print("To include an image, prefix with: image:/path/to/image.jpg  <prompt>")
    print("=" * 50 + "\n")

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
            generate_response(model, processor, tokenizer, tokenizer_stream, prompt, image_path)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        print("-" * 50 + "\n")

    print("Goodbye!")


if __name__ == "__main__":
    main()
