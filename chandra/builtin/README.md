# Chandra ONNX Runtime GenAI Example

This example demonstrates how to convert [datalab-to/chandra](https://huggingface.co/datalab-to/chandra) to ONNX format using Olive and run inference with ONNX Runtime GenAI.

Chandra is a vision-language model fine-tuned from [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), sharing the same architecture (hidden_size=4096, 36 layers, patch_size=16). The key preprocessing difference is a larger pixel range: `min_pixels=65536`, `max_pixels=16777216` (vs 3136/12845056 in the base model).

The pipeline exports three sub-models (vision encoder, text embedding, text decoder), applies graph optimizations (Cast chain elimination, Gemm→MatMul conversion), and quantizes or converts them depending on the target device:

- **CPU/Mobile:** All three sub-models are quantized to INT4.
- **CUDA:** The text decoder is INT4 (via ModelBuilder); the vision encoder and embedding model are FP16.

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

All graph transformations and quantization are declared in the JSON config files inside `cpu_and_mobile/` and `cuda/`. The top-level `optimize.py` script orchestrates the three Olive runs and generates the GenAI runtime configs.

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | Full pipeline: export, optimize, INT4 quantize (CPU) |
| `python optimize.py --config-dir cuda --device gpu` | Full pipeline: INT4 text + FP16 embedding/vision (CUDA) |
| `python optimize.py --config-dir cpu_and_mobile --skip-export` | Regenerate configs only (models already exported) |

> **Note:** The text model is exported as INT4 via ModelBuilder. The vision encoder and embedding model are quantized to INT4 (CPU) or kept at FP16 (CUDA).
>
> The vision encoder is exported for a single image using the Dynamo exporter. At runtime, ONNX Runtime GenAI handles multiple images by calling the vision encoder once per image and concatenating the results.

### 2. Run Inference

```bash
# Text-only (CPU models, default)
python inference.py --prompt "What is the capital of France?"

# With a single image
python inference.py --prompt "Describe this image" --image cat.jpeg

# CUDA models
python inference.py --model_path cuda/models --prompt "Describe this image" --image cat.jpeg

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
# ONNX only (fastest)
python eval.py --num_samples 100

# ONNX + PyTorch comparison
python eval.py --num_samples 100 --pytorch_model datalab-to/chandra

# Evaluate CUDA models
python eval.py --model_path cuda/models --num_samples 100
```

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
    ├── cpu_and_mobile/
    │   ├── embedding.json         # Olive config: export → optimize → INT4
    │   ├── vision.json            # Olive config: Dynamo export → graph surgeries → INT4
    │   ├── text.json              # Olive config: ModelBuilder INT4
    │   └── models/                # Exported ONNX models (generated)
    └── cuda/
        ├── embedding.json         # Olive config: export → optimize → FP16 + CUDA EP
        ├── vision.json            # Olive config: Dynamo export → graph surgeries → FP16 + CUDA EP
        ├── text.json              # ModelBuilder INT4 with CUDA EP
        └── models/                # Exported CUDA ONNX models (generated)
```
