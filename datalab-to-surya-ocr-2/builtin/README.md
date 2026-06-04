# surya-ocr-2 ONNX Runtime GenAI Example

This example converts [datalab-to/surya-ocr-2](https://huggingface.co/datalab-to/surya-ocr-2)
to ONNX format using Olive and runs inference with ONNX Runtime GenAI.

surya-ocr-2 is a **650M-parameter multimodal OCR and document-intelligence model** supporting
OCR, layout analysis, reading order detection, table recognition, and mathematical equation
recognition across 91 languages. It achieves 83.3% on olmOCR-bench (top under 3B parameters).

## Architecture

surya-ocr-2 uses the **Qwen3.5 architecture** (`Qwen3_5ForConditionalGeneration`) but with a
**custom OCR vocabulary and custom token IDs** — these differ entirely from standard Qwen3.5-VL:

| Property | surya-ocr-2 | Standard Qwen3.5-VL |
|----------|------------|---------------------|
| `vocab_size` | **65425** | 152064 |
| `image_token_id` | **11** | 151655 |
| `vision_start_token_id` | **9** | 151652 |
| `vision_end_token_id` | **10** | 151653 |
| `video_token_id` | **12** | 151656 |
| Vision `hidden_size` | **768** | 1152 |
| Vision `depth` | **12** | 27 |
| Vision `out_hidden_size` | **1024** | 3584 |
| `deepstack_visual_indexes` | **[]** (none) | [8, 16, 24] |
| Text `hidden_size` | 1024 | 1024 |
| Text `num_hidden_layers` | 24 | 24 |

### Preprocessing (confirmed from `preprocessor_config.json`)

| Property | Value |
|----------|-------|
| `image_mean / image_std` | [0.5, 0.5, 0.5] |
| `rescale_factor` | 1/255 |
| `patch_size` | 16 |
| `spatial_merge_size` | 2 |
| `temporal_patch_size` | 2 |
| `min_pixels` | 65536 (256×256) |
| `max_pixels` | 16777216 (4096×4096) |

### Export Strategy

Same 3-sub-model split as Qwen3.5-VL:

| Sub-model | Export path | Notes |
|-----------|-------------|-------|
| **text** | `ModelBuilder` (INT4) | Gated DeltaNet decoder handled natively |
| **vision** | Custom Dynamo export | ViT encoder (depth=12), ONNX-safe |
| **embedding** | Custom TorchScript export | `embed_tokens` + image scatter |

## Prerequisites

```bash
pip install -r requirements.txt
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| GPU (CUDA) | `pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |
| CPU | `pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |

## Steps

### 1. Export & Optimize Models

```bash
# CPU (INT4 — all three sub-models)
python optimize.py --config-dir cpu_and_mobile --device cpu

# CUDA (INT4 text + FP16 vision/embedding)
python optimize.py --config-dir cuda --device gpu

# Regenerate GenAI configs only (models already exported)
python optimize.py --config-dir cpu_and_mobile --skip-export
```

### 2. Run Inference

```bash
# OCR a document image (default prompt: "OCR the full text of this document.")
python inference.py --image document.png

# Custom OCR prompt
python inference.py --image document.png --prompt "Extract all text from this page."

# Layout analysis
python inference.py --image page.png --prompt "Identify the layout blocks in this document."

# CUDA models
python inference.py --model_path cuda/models --image document.png

# Interactive mode
python inference.py --interactive
```

## Directory Structure

```
datalab-to-surya-ocr-2/
└── builtin/
    ├── codes/
    │   ├── __init__.py
    │   └── modeling_qwen3_5_vl.py     # Shared Qwen3.5 custom model (config-driven)
    ├── cpu_and_mobile/
    │   ├── embedding.json              # Olive: export -> ORT opt -> INT4
    │   ├── vision.json                 # Olive: Dynamo export -> graph surgeries -> INT4
    │   ├── text.json                   # Olive: ModelBuilder INT4
    │   └── models/                     # Generated ONNX models
    ├── cuda/
    │   ├── embedding.json              # Olive: export -> FP16 + CUDA EP
    │   ├── vision.json                 # Olive: Dynamo export -> FP16 + CUDA EP
    │   ├── text.json                   # Olive: ModelBuilder INT4 + CUDA EP
    │   └── models/                     # Generated CUDA ONNX models
    ├── user_script.py                  # Olive callbacks (model loading, IO configs, dummy inputs)
    ├── optimize.py                     # Orchestrates Olive + writes GenAI configs
    ├── inference.py                    # ORT GenAI inference (OCR-focused)
    └── requirements.txt
```

## Checkpoint Key Remapping

| Checkpoint key | Our model key | Action |
|----------------|--------------|--------|
| `model.visual.*` | `visual.*` | ✅ Loaded |
| `model.language_model.embed_tokens.weight` | `language_model.embed_tokens.weight` | ✅ Loaded |
| `model.language_model.layers.*` (DeltaNet) | *(not in our model)* | Silently ignored |
| `lm_head.weight` | *(not in our model)* | Silently ignored |
