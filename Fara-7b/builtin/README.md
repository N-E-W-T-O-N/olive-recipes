# microsoft/Fara-7B ONNX Runtime GenAI Example

This example demonstrates how to convert [microsoft/Fara-7B](https://huggingface.co/microsoft/Fara-7B) to ONNX format using Olive and run inference with ONNX Runtime GenAI.

**Fara-7B** is Microsoft's agentic vision-language model for **computer use** — screen understanding, GUI interaction, and OS-level task automation. It is fine-tuned from [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) with specialisation in understanding screenshots, desktop UIs, web pages, and multi-step computer control tasks.

## Architecture

Fara-7B shares the Qwen2.5-VL-7B architecture with one key difference: a **larger vision window** (`window_size=112` vs 56 in the 3B variant) providing wider spatial context for screen understanding.

### Key Architecture Values (confirmed from `config.json`)

| Property | Value | Source |
|----------|-------|--------|
| Base model | Qwen2.5-VL-7B | `base_model_name_or_path` |
| Text `hidden_size` | 3584 | `text_config.hidden_size` |
| Text `num_hidden_layers` | 28 | `text_config.num_hidden_layers` |
| Text `num_attention_heads` | 28 | `text_config.num_attention_heads` |
| Vision `patch_size` | 14 | `vision_config.patch_size` |
| Vision `window_size` | **112** | `vision_config.window_size` (≠ Qwen2.5-VL-3B's 56) |
| Vision `spatial_merge_size` | 2 | `vision_config.spatial_merge_size` |
| Vision `temporal_patch_size` | 2 | `vision_config.temporal_patch_size` |
| Vision channels/patch | 1176 (3×14×14×2) | derived |
| Vision `out_hidden_size` | 3584 | `vision_config.out_hidden_size` |
| `image_token_id` | 151655 | `config.image_token_id` |
| `video_token_id` | 151656 | `config.video_token_id` |
| `vision_start_token_id` | 151652 | `config.vision_start_token_id` |

### Preprocessing (confirmed from `preprocessor_config.json`)

| Property | Value |
|----------|-------|
| `image_mean` | [0.48145466, 0.4578275, 0.40821073] (CLIP) |
| `image_std` | [0.26862954, 0.26130258, 0.27577711] (CLIP) |
| `rescale_factor` | 1/255 |
| `patch_size` | 14 |
| `merge_size` | 2 |
| `temporal_patch_size` | 2 |
| `min_pixels` | 3136 (56×56) |
| `max_pixels` | 12845056 |

### Checkpoint Key Format

Fara-7B's checkpoint uses a **flat key naming convention** — the text decoder is stored at the top level rather than nested under `language_model`:

| Checkpoint key | `Qwen2_5_VLModel` expected key | Handled by |
|----------------|-------------------------------|------------|
| `model.layers.*` | `language_model.layers.*` | `_remap_fara7b_state_dict()` |
| `model.embed_tokens.weight` | `language_model.embed_tokens.weight` | `_remap_fara7b_state_dict()` |
| `model.norm.weight` | `language_model.norm.weight` | `_remap_fara7b_state_dict()` |
| `lm_head.weight` | *(not in Qwen2_5_VLModel)* | Dropped |
| `visual.*` | `visual.*` | ✅ Direct match |

`user_script.py` remaps these keys automatically before calling `load_state_dict(strict=False)`.

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
| `python optimize.py --config-dir cuda --device gpu` | Full pipeline: FP16 vision/embedding + INT4 text (CUDA) |
| `python optimize.py --config-dir cpu_and_mobile --skip-export` | Regenerate GenAI configs only (models already exported) |

> **Note:** The text model is exported as INT4 via ModelBuilder. The vision encoder is exported
> using the Dynamo exporter with `PackedAttentionToLoopMHA` graph surgery, then quantized to INT4 (CPU)
> or FP16 (CUDA). The embedding model's Gather-based table is quantized using GatherBlockQuantized.
>
> Vision encoder uses `window_size=112` — wider than the Qwen2.5-VL-3B variant (56). This is critical
> for accurate screen understanding at high-resolution input.

### 2. Run Inference

```bash
# Text-only (CPU models, default)
python inference.py --prompt "What is the capital of France?"

# With a screenshot / screen image (primary use case)
python inference.py --prompt "Describe what you see on this screen" --image screenshot.png

# Computer use: describe UI elements
python inference.py --prompt "What buttons are visible and what do they do?" --image desktop.png

# CUDA models
python inference.py --model_path cuda/models --prompt "Click the submit button" --image browser.png

# Interactive mode
python inference.py --interactive
```

**Multi-image inference** is supported via `model-mm.py` from `onnxruntime-genai` examples:

```bash
python <onnxruntime-genai>/examples/python/model-mm.py \
    -m <path-to-builtin>/cpu_and_mobile/models \
    -up "Compare these two screenshots and describe the differences." \
    --image_paths before.png after.png \
    --non_interactive
```

## Evaluation

`eval.py` measures model quality on [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) — a multiple-choice visual QA benchmark.

```bash
# ONNX model only
python eval.py --num_samples 100

# Side-by-side with PyTorch baseline
python eval.py --num_samples 100 --pytorch_model microsoft/Fara-7B

# CUDA models
python eval.py --model_path cuda/models --num_samples 100
```

## Export Pipeline Details

```
microsoft/Fara-7B (HF Hub)
    │
    ├─► [text.json]      ModelBuilder INT4 ──────────────► text.onnx
    │
    ├─► [vision.json]    Dynamo export
    │                    → PackedAttentionToLoopMHA       ┐
    │                    → ReciprocalMulToDiv             ├─► vision.onnx
    │                    → ORT optimization               │
    │                    → Cast chain elimination         │
    │                    → GemmToMatMulAdd                │
    │                    → INT4 quantization (CPU)        ┘
    │                       FP16 conversion (CUDA)
    │
    └─► [embedding.json] TorchScript export
                         → ORT optimization               ┐
                         → Cast chain elimination         ├─► embedding.onnx
                         → GemmToMatMulAdd                │
                         → INT4 quantization (CPU)        ┘
                            FP16 conversion (CUDA)
```

## Directory Structure

```
Fara-7b/
├── LICENSE
└── builtin/
    ├── optimize.py                # End-to-end Olive pipeline + GenAI config generation
    ├── user_script.py             # Model loading (with flat→nested key remapping), IO configs, dummy inputs
    ├── eval.py                    # AI2D accuracy evaluation (ONNX vs PyTorch)
    ├── inference.py               # ONNX Runtime GenAI inference
    ├── cat.jpeg                   # Sample test image
    ├── codes/                     # Custom Qwen2.5-VL PyTorch model adapted for ONNX export
    │   └── modeling_qwen2_5_vl.py
    ├── cpu_and_mobile/
    │   ├── embedding.json         # Olive: export → ORT opt → Cast elim → INT4
    │   ├── vision.json            # Olive: Dynamo export → graph surgeries → INT4
    │   ├── text.json              # Olive: ModelBuilder INT4
    │   └── models/                # Generated ONNX models (created on export)
    └── cuda/
        ├── embedding.json         # Olive: export → ORT opt → FP16 + CUDA EP
        ├── vision.json            # Olive: Dynamo export → graph surgeries → FP16 + CUDA EP
        ├── text.json              # Olive: ModelBuilder INT4 + CUDA EP
        └── models/                # Generated CUDA ONNX models (created on export)
```

## Notes on `window_size`

The `window_size=112` in `optimize.py` is critical and **differs from the Qwen2.5-VL-3B base**:

| Model | `window_size` | Source |
|-------|--------------|--------|
| Qwen2.5-VL-3B-Instruct | 56 | `vision_config.window_size` |
| **microsoft/Fara-7B** | **112** | `vision_config.window_size` |

Using the wrong value (56) would cause incorrect positional attention masking in the vision encoder, degrading quality on high-resolution screenshots. The value 112 is confirmed directly from the HuggingFace model card `config.json`.
