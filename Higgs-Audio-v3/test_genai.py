"""Empirical test: can onnxruntime-genai (og) drive our llm_decoder?

Claim under test: it cannot run the TTS model end-to-end, because the decoder is
built with exclude_embeds + exclude_lm_head — so og has (a) no embedding to turn
tokens into `inputs_embeds`, and (b) no lm_head/logits to sample from. og's
Generator loop requires both.

This script tries the normal og path and reports exactly where/why it fails (or, if
a build env ever changes, whether it works). Run:

  uv run python test_genai.py --model-path onnx/cpu_int4
"""
import argparse
import json
from pathlib import Path


def inspect_genai_config(model_path: Path):
    cfg = json.loads((model_path / "genai_config.json").read_text())
    dec = cfg["model"]["decoder"]
    print("genai_config decoder:")
    print("  inputs :", list(dec["inputs"].values()))
    print("  outputs:", list(dec["outputs"].values()))
    has_logits = any("logits" in v.lower() for v in dec["outputs"].values())
    has_embeds_input = any("embed" in v.lower() for v in dec["inputs"].values())
    print(f"  → produces logits?   {has_logits}")
    print(f"  → consumes inputs_embeds (embeds excluded)? {has_embeds_input}")
    return has_logits, has_embeds_input


def try_og_generation(model_path: Path):
    import onnxruntime_genai as og
    print(f"\nonnxruntime-genai {og.__version__}")
    print("Loading og.Model ...")
    model = og.Model(str(model_path))          # reads genai_config.json
    print("  og.Model loaded OK")

    tokenizer = og.Tokenizer(model)
    tokens = tokenizer.encode("Hello, world.")
    print(f"  tokenized prompt → {len(tokens)} token ids")

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(tokens) + 8, temperature=0.0)
    gen = og.Generator(model, params)
    print("  appending tokens (og will try to embed them internally) ...")
    gen.append_tokens(tokens)          # needs the embedding matrix (excluded!)
    print("  generating ...")
    out = []
    while not gen.is_done() and len(out) < 8:
        gen.generate_next_token()      # needs logits/lm_head (excluded!)
        out.append(int(gen.get_next_tokens()[0]))
    print(f"  generated tokens: {out}")
    print(f"  decoded: {tokenizer.decode(out)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="onnx/cpu_int4")
    args = ap.parse_args()
    mp = Path(args.model_path)

    print("=" * 70)
    print("TEST: onnxruntime-genai driving the Higgs-v3 llm_decoder")
    print("=" * 70)
    has_logits, embeds_excluded = inspect_genai_config(mp)

    print("\nAttempting og.Generator text generation ...")
    try:
        try_og_generation(mp)
        print("\nRESULT: og generation RAN (unexpected for the exclude-head build).")
    except Exception as e:
        print(f"\nRESULT: og generation FAILED — {type(e).__name__}: {str(e)[:300]}")
        print("\nWhy (by design):")
        if embeds_excluded:
            print("  • exclude_embeds → no embedding in the graph, so og cannot convert")
            print("    its token ids into inputs_embeds (the model's only input).")
        if not has_logits:
            print("  • exclude_lm_head → the model outputs hidden_states, not logits, so")
            print("    og's sampler has nothing to sample from.")
        print("  ⇒ This is exactly why inference.py drives the loop itself: raw ORT runs")
        print("    the decoder block, then WE apply the tied embedding, the fused audio")
        print("    head, the delay-pattern sampler, audio_embed feedback, and the codec.")
        print("  (To use og for plain TEXT gen you'd rebuild WITHOUT the excludes — a full")
        print("   Qwen3 LM with lm_head — but that is not the TTS sub-part pipeline.)")


if __name__ == "__main__":
    main()
