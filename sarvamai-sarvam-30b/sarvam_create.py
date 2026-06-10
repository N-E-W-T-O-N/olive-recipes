# -------------------------------------------------------------------------
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI generated content.
# -------------------------------------------------------------------------
"""Convenience wrapper around builder.py for sarvamai/sarvam-30b.

Usage
-----
# INT4 on CPU (no GPU needed, slow but works):
python create.py --device cpu

# INT4 on CUDA GPU:
python create.py --device gpu

# fp16 on GPU:
python create.py --device gpu --precision fp16

# Override output directory and cache:
python create.py --device cpu --output ./my_output --cache_dir ./my_cache

# Dry-run: generate GenAI config only (no ONNX conversion):
python create.py --device gpu --config_only

# Pass any extra builder options:
python create.py --device gpu --extra_options int4_algo_config=k_quant_mixed

This script hard-codes:
  model_name  = sarvamai/sarvam-30b
  input_path  = ""  (download from HuggingFace)
  precision   = int4   (default; override with --precision)
  trust_remote_code is always enabled via hf_remote extra option
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# builder.py lives in the same folder — make sure it is importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from builder import create_model, parse_extra_options  # noqa: E402

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_NAME = "sarvamai/sarvam-30b"

# Map --device to the execution_provider string expected by create_model
DEVICE_TO_EP = {
    "cpu":  "cpu",
    "gpu":  "cuda",
}

# Default output directories per device
DEFAULT_OUTPUT = {
    "cpu":  os.path.join(".", "output", "sarvam-30b-cpu-int4"),
    "gpu":  os.path.join(".", "output", "sarvam-30b-cuda-int4"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Build the sarvamai/sarvam-30b ONNX model for ONNX Runtime GenAI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--device",
        required=True,
        choices=["cpu", "gpu"],
        help="Target device.  'cpu' → CPU execution provider.  'gpu' → CUDA execution provider.",
    )

    parser.add_argument(
        "--precision",
        required=False,
        default="int4",
        choices=["int4", "fp16", "bf16", "fp32"],
        help="Model precision (default: int4).",
    )

    parser.add_argument(
        "--output",
        required=False,
        default=None,
        help="Output directory for the ONNX model and GenAI config.  "
             "Defaults to ./output/sarvam-30b-{device}-{precision}.",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        default=os.path.join(".", "cache_dir"),
        help="Directory used to cache HuggingFace downloads and temporary files (default: ./cache_dir).",
    )

    parser.add_argument(
        "--input",
        required=False,
        default="",
        help="Optional path to a local HuggingFace model directory.  "
             "Leave empty to download from HuggingFace Hub (default).",
    )

    parser.add_argument(
        "--config_only",
        action="store_true",
        help="Generate GenAI config files only — skip ONNX conversion.  "
             "Useful when you already have the ONNX model.",
    )

    parser.add_argument(
        "--extra_options",
        required=False,
        metavar="KEY=VALUE",
        nargs="+",
        help="Additional key=value options forwarded to the model builder.  "
             "Example: --extra_options int4_algo_config=k_quant_mixed qmoe_block_size=128",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    execution_provider = DEVICE_TO_EP[args.device]

    # Build output path if not specified
    if args.output is None:
        output_dir = os.path.join(
            ".", "output", f"sarvam-30b-{args.device}-{args.precision}"
        )
    else:
        output_dir = args.output

    # Parse user-supplied extra_options and inject our required defaults
    extra_options = parse_extra_options(args.extra_options, execution_provider)

    # Always trust remote code — sarvam_moe is a custom architecture
    extra_options.setdefault("hf_remote", True)

    # Propagate --config_only flag
    if args.config_only:
        extra_options["config_only"] = True

    # ------------------------------------------------------------------
    # Print a clear summary before starting (conversion takes a long time)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  sarvamai/sarvam-30b  ONNX Model Builder")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Device     : {args.device}  ({execution_provider} EP)")
    print(f"  Precision  : {args.precision}")
    print(f"  Output     : {os.path.abspath(output_dir)}")
    print(f"  Cache      : {os.path.abspath(args.cache_dir)}")
    print(f"  Config only: {args.config_only}")
    if extra_options:
        print(f"  Extra opts : {extra_options}")
    print("=" * 60)

    if args.device == "cpu":
        print(
            "INFO: Running on CPU.  This will use system RAM for the full "
            "model (~60 GB for fp16 load + ONNX graph).  Conversion may "
            "take several hours.  Ensure you have >=64 GB of free RAM."
        )
    else:
        print(
            "INFO: Running on CUDA GPU.  Ensure your GPU has sufficient "
            "VRAM (>=24 GB for int4, >=40 GB for fp16)."
        )
    print()

    # ------------------------------------------------------------------
    # Call the builder
    # ------------------------------------------------------------------
    create_model(
        model_name=MODEL_NAME,
        input_path=args.input,
        output_dir=output_dir,
        precision=args.precision,
        execution_provider=execution_provider,
        cache_dir=args.cache_dir,
        **extra_options,
    )

    print()
    print("=" * 60)
    print(f"  Done.  ONNX model saved to: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
