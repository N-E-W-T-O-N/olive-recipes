# unsloth/csm-1b — Olive Recipes

Olive INT4 quantization recipes for [unsloth/csm-1b](https://huggingface.co/unsloth/csm-1b),
an optimised version of the [Sesame CSM-1B](https://huggingface.co/sesame/csm-1b)
Conversational Speech Model.

---

## Model Overview

| Property | Value |
|---|---|
| Architecture | `CsmForConditionalGeneration` |
| Model type | `csm` |
| Modality | Text → Speech (TTS / voice generation) |
| Parameters | ~1B (backbone) + ~1B (depth decoder) |
| Precision | float16 |
| Trust remote code | Not required |
| Transformers support | Built-in (no custom code) |

### Architecture Details

| Component | Value |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 16 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 (GQA) |
| intermediate_size | 8192 |
| vocab_size | 128,256 text + 2,051 audio tokens |
| RoPE theta | 500,000 (LLaMA-3 style, 32× scaling) |
| Audio codec | Mimi (24 kHz, 12.5 fps, 32 RVQ codebooks) |
| Depth decoder | 4-layer transformer, hidden_size=1024 |

---

## Folder Structure

```
csm-1B/
├── cpu/    unsloth-csm-1b_cpu_int4.json    (CPUExecutionProvider,  group_size=128)
├── cuda/   unsloth-csm-1b_cuda_int4.json   (CUDAExecutionProvider, group_size=128)
└── webgpu/ unsloth-csm-1b_webgpu_int4.json (WebGpuExecutionProvider, group_size=32)
```

---

## Running a Recipe

```bash
# CPU
olive run --config cpu/unsloth-csm-1b_cpu_int4.json

# CUDA GPU
olive run --config cuda/unsloth-csm-1b_cuda_int4.json

# WebGPU
olive run --config webgpu/unsloth-csm-1b_webgpu_int4.json
```

---

## Hardware Requirements

| Target | Min VRAM / RAM | Notes |
|---|---|---|
| CPU INT4 | 4 GB RAM | ~0.5 GB model, fast on modern CPUs |
| CUDA INT4 | 4 GB VRAM | RTX 3060+ sufficient |
| WebGPU INT4 | 4 GB VRAM | Any WebGPU-capable GPU |
| fp16 (no quant) | 2 GB VRAM | Backbone only |

---

## Known Limitations

### ModelBuilder Support
`CsmForConditionalGeneration` is a **multimodal speech model** — it takes both
text tokens and audio codes as input and outputs RVQ audio codes for the Mimi codec.
The standard `ModelBuilder` pass targets text-only causal LMs; it will fail with:

```
NotImplementedError: The CsmForConditionalGeneration model is not currently supported.
```

**Workaround:** Use an ONNX conversion pipeline instead:

```json
"passes": {
    "convert": {
        "type": "OnnxConversion",
        "target_opset": 20
    },
    "quantize": {
        "type": "OnnxMatMul4Quantizer",
        "block_size": 128
    }
}
```

Or export the backbone and depth decoder as **two separate ONNX models** and call
them sequentially at inference time.

### Audio Decoding
The ONNX model outputs **RVQ audio codes**, not raw waveforms.
Post-processing with the [Mimi codec](https://huggingface.co/kyutai/mimi) is required
to convert codes → audio.

### Depth Decoder
The 4-layer depth decoder (hidden_size=1024) may need to be exported separately
from the 1B backbone for best quantization results.

---

## Quantization Strategy

| Pass | Purpose |
|---|---|
| `SelectiveMixedPrecision (k_quant_mixed)` | Identifies sensitive layers and promotes them to int8 |
| `gptq (4-bit)` | Quantises the bulk of weight matrices to int4 |
| `rtn (8-bit)` | Quantises embeddings and lm_head to int8 |
| `ModelBuilder` | Packs into ORT GenAI format (requires architecture support) |

group_size=32 for WebGPU (better accuracy on smaller tiles); 128 for CPU/CUDA.
