# Chandra ONNX Runtime GenAI Example

This example demonstrates how to convert [datalab-to/chandra](https://huggingface.co/datalab-to/chandra) to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Chandra is a vision-language model fine-tuned from [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), sharing the same architecture (hidden_size=4096, 36 layers, patch_size=16). The key preprocessing difference is a larger pixel range: `min_pixels=65536`, `max_pixels=16777216` (vs 3136/12845056 in the base model).

The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations (Cast chain elimination, Gemm→MatMul conversion), and quantizes/converts per **target = device × precision**.

### Build targets (`--device {cpu,cuda} --precision {int4,fp16,fp32}`)

| Target | text decoder | vision + embedding | output dir |
|--------|--------------|--------------------|------------|
| `cpu_int4`  | INT4 | INT4 | `cpu_int4/models/`  |
| `cpu_fp16`  | FP16 | FP16 | `cpu_fp16/models/`  |
| `cpu_fp32`  | FP32 | FP32 (optimized float) | `cpu_fp32/models/` |
| `cuda_fp16` | FP16 | FP16 | `cuda_fp16/models/` |
| `cuda_fp32` | FP32 | FP32 | `cuda_fp32/models/` |

- The **text decoder** is built with onnxruntime-genai **ModelBuilder (`create_model`) directly** — not via an Olive pass — because Olive's ModelBuilder pass restricts the cpu/precision combos (e.g. cpu + fp32). Calling `create_model` directly allows every device × precision.
- The **vision encoder** and **embedding** are built via Olive (configs generated in-Python by `optimize.py`); the final pass is swapped per precision: INT4 `OnnxBlockWiseRtnQuantization`, FP16 `OnnxFloatToFloat16`, FP32 none.
- ⚠️ **Size:** these are an 8B model — FP32 text ≈ 32 GB, FP16 ≈ 16 GB, INT4 ≈ 5 GB per target. Build only the targets you need.

## Architecture & Preprocessing Summary

| Property | Value |
|----------|-------|
| Base model | Qwen3-VL-8B-Instruct |
| Text hidden_size | 4096 |
| Text num_hidden_layers | 36 |
| Vision patch_size | 16 |
| Vision spatial_merge_size | 2 |
| Vision temporal_patch_size | 2 |
| Vision channels/patch | 1536 (3×16×16×2) |
| Vision out_hidden_size | 4096 |
| image_mean / image_std | [0.5, 0.5, 0.5] |
| rescale_factor | 1/255 |
| min_pixels | 65536 (256×256) |
| max_pixels | 16777216 (4096×4096) |
| image_token_id | 151655 |
| video_token_id | 151656 |
| vision_start_token_id | 151652 |

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI based on your target device:

| Device | Install Command |
|--------|-----------------|
| GPU (CUDA) | `pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |
| CPU | `pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |

## Steps

### 1. Export & Optimize Models

`optimize.py` generates the Olive configs in-Python and orchestrates the three sub-model
builds for a chosen `--device` × `--precision`. Output goes to `<device>_<precision>/models/`.

| Command | Target |
|---------|--------|
| `python optimize.py --device cpu  --precision int4` | `cpu_int4/models`  |
| `python optimize.py --device cpu  --precision fp16` | `cpu_fp16/models`  |
| `python optimize.py --device cpu  --precision fp32` | `cpu_fp32/models`  |
| `python optimize.py --device cuda --precision fp16` | `cuda_fp16/models` |
| `python optimize.py --device cuda --precision fp32` | `cuda_fp32/models` |

Subset / regen:
```bash
python optimize.py --device cpu --precision int4 --components vision     # one sub-model
python optimize.py --device cpu --precision int4 --skip-export           # regen genai/processor configs only
```

> **Notes**
> - The text decoder is built with **ModelBuilder (`create_model`) directly** — this is what makes cpu + fp32 possible (Olive's ModelBuilder pass restricts that combo).
> - The vision encoder is exported for a single image with the Dynamo exporter; at runtime GenAI calls it once per image and concatenates results.

### 2. Run Inference

```bash
# Text-only (default cpu_int4/models)
python inference.py --prompt "What is the capital of France?"

# With a single image
python inference.py --prompt "Describe this image" --image cat.jpeg

# A specific target
python inference.py --model_path cuda_fp16/models --prompt "Describe this image" --image cat.jpeg

# Interactive mode
python inference.py --interactive
```

**Multi-image inference** via `model-mm.py` from the `onnxruntime-genai` examples:

```bash
python <onnxruntime-genai>/examples/python/model-mm.py \
    -m <path-to-builtin>/cpu_and_mobile/models \
    -up "Are these two images the same?" \
    --image_paths image1.jpeg image2.jpeg \
    --non_interactive
```

## Evaluation

`eval.py` measures model quality on [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) — a multiple-choice visual QA benchmark on scientific diagrams.

```bash
# one target (default cpu_int4/models)
python eval.py --num_samples 100 --model_path cpu_int4/models

# sweep + compare accuracy across several targets
python eval.py --num_samples 100 \
    --targets cpu_int4/models,cpu_fp16/models,cpu_fp32/models

# ONNX + PyTorch reference comparison
python eval.py --num_samples 100 --model_path cpu_int4/models --pytorch_model datalab-to/chandra
```

`--targets` evaluates each model dir on the same AI2D subset and prints an accuracy +
latency comparison — the quickest way to see the quality cost of each precision.

## Directory Structure

```
chandra/
├── LICENSE
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Olive callbacks: model loading, dummy inputs, IO configs
    ├── eval.py                    # AI2D accuracy evaluation (ONNX vs PyTorch)
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── cat.jpeg                   # Sample test image
    ├── codes/                     # Custom Qwen3-VL PyTorch model adapted for ONNX export
    └── <device>_<precision>/      # one per built target (generated)
        └── models/                # text.onnx, embedding.onnx, vision.onnx,
                                   #   genai_config.json, processor_config.json, tokenizer*
```

Targets: `cpu_int4/  cpu_fp16/  cpu_fp32/  cuda_fp16/  cuda_fp32/`. Olive + ModelBuilder
configs are generated in-Python by `optimize.py` — there are no static per-dir JSON config files.
