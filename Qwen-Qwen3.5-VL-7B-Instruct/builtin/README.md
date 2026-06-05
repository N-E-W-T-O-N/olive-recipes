# unsloth/Qwen3.5-0.8B ONNX Runtime GenAI Example

This example converts [unsloth/Qwen3.5-0.8B](hunsloth/Qwen3.5-0.8B)
to ONNX format using Olive and runs inference with ONNX Runtime GenAI.

> **Gated model**: Qwen3.5-VL requires accepting a license on HuggingFace.
> Run `huggingface-cli login` with a valid token before use.

## Architecture Overview

Qwen3.5-VL differs from Qwen3-VL in its **text decoder**: it uses a **Hybrid Gated DeltaNet**
(3 DeltaNet linear-attention layers : 1 standard attention layer), implemented via
`torch_chunk_gated_delta_rule`. This creates very large ONNX graphs if exported naively
(see [reference analysis](https://github.com/garlic-byte/Qwen3_VL_Export_ONNX_and_TensorRT)).

This pipeline avoids that problem by splitting export into three independent sub-models:

| Sub-model | Export path | Notes |
|-----------|-------------|-------|
| **text** | `ModelBuilder` (INT4) | Full Gated DeltaNet decoder, handled natively by ModelBuilder |
| **vision** | Custom Dynamo export | ViT encoder — identical architecture to Qwen3-VL, ONNX-safe |
| **embedding** | Custom TorchScript export | `embed_tokens` + image feature scatter, no DeltaNet involved |

### Key Architecture Values

| Property | Value |
|----------|-------|
| Text hidden_size | 4096 (from `text_config`) |
| Vision patch_size | 16 |
| Vision spatial_merge_size | 2 |
| Vision temporal_patch_size | 2 |
| Vision channels/patch | 1536 (3×16×16×2) |
| Vision out_hidden_size | from `vision_config.out_hidden_size` |
| Vision num_position_embeddings | 2304 (48×48 learnable absolute pos embed) |
| image_mean / image_std | [0.5, 0.5, 0.5] |
| rescale_factor | 1/255 |
| min_pixels | 65536 (verify from your model's preprocessor_config.json) |
| max_pixels | 16777216 (verify from your model's preprocessor_config.json) |
| image_token_id | 151655 |
| video_token_id | 151656 |
| vision_start_token_id | 151652 |

> **Note on min_pixels / max_pixels**: Qwen3.5-VL expresses these as `size.shortest_edge` and
> `size.longest_edge` in `preprocessor_config.json`. Update `_PREPROCESSOR` in `optimize.py`
> once you have access to the gated model card.

## Prerequisites

```bash
pip install -r requirements.txt
huggingface-cli login      # required — model is gated
```

Install ONNX Runtime GenAI:

| Device | Install Command |
|--------|-----------------|
| GPU (CUDA) | `pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |
| CPU | `pip install onnxruntime-genai --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple` |

## Steps

### 1. Export & Optimize Models

| Command | Description |
|---------|-------------|
| `python optimize.py --config-dir cpu_and_mobile --device cpu` | Export + INT4 quantize (CPU) |
| `python optimize.py --config-dir cuda --device gpu` | Export + FP16 embedding/vision, INT4 text (CUDA) |
| `python optimize.py --config-dir cpu_and_mobile --skip-export` | Regenerate GenAI configs only |

> **Important**: The text model uses `ModelBuilder` which internally handles the Gated DeltaNet
> attention. The vision and embedding models use our custom `Qwen3_5VLModel` wrapper which
> includes only the ONNX-safe components.

### 2. Run Inference

```bash
# Text-only (CPU, default)
python inference.py --prompt "Describe Qwen3.5-VL in one sentence."

# Image + text
python inference.py --prompt "What do you see?" --image cat.jpeg

# CUDA models
python inference.py --model_path cuda/models --prompt "Describe this image" --image cat.jpeg

# Interactive
python inference.py --interactive
```

## Known Limitations

1. **Gated DeltaNet ONNX size**: The `torch_chunk_gated_delta_rule` function (text decoder)
   produces very large ONNX graphs due to Python loop unrolling. This pipeline avoids the
   problem by delegating the text decoder to `ModelBuilder`, but be aware that the exported
   `text.onnx` may be large.

2. **Gated model access**: You must accept the license on HuggingFace Hub before download.

3. **min_pixels / max_pixels**: These values in `optimize.py` are inferred from similar
   models. Verify them against the actual `preprocessor_config.json` once you have access.

## Directory Structure

```
Qwen-Qwen3.5-VL-7B-Instruct/
└── builtin/
    ├── codes/
    │   ├── __init__.py
    │   └── modeling_qwen3_5_vl.py     # Custom model: vision encoder + embed_tokens only
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
    ├── user_script.py                  # Olive callbacks: model loading, IO configs, dummy inputs
    ├── optimize.py                     # Orchestrates Olive + writes GenAI configs
    ├── inference.py                    # ORT GenAI inference
    ├── eval.py                         # AI2D accuracy evaluation
    └── requirements.txt
```

## Checkpoint Key Remapping

Qwen3.5-VL's HF checkpoint (`Qwen3_5ForConditionalGeneration`) saves weights as:

| Checkpoint key | Our model key | Action |
|----------------|--------------|--------|
| `model.visual.*` | `visual.*` | ✅ Loaded |
| `model.language_model.embed_tokens.weight` | `language_model.embed_tokens.weight` | ✅ Loaded |
| `model.language_model.layers.*` (DeltaNet) | *(not in our model)* | Silently ignored |
| `lm_head.weight` | *(not in our model)* | Silently ignored |

`user_script._load_base_model` strips the `model.` prefix and uses `strict=False`,
so the DeltaNet text-decoder weights are silently ignored — only the vision encoder
and `embed_tokens` are populated in our custom model.
