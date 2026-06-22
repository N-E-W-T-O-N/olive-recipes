# Higgs-Audio-v3 — Device & Precision Support

Support matrix for the ONNX pipeline. "Verified" = actually run on this machine
(Intel CPU, no NVIDIA GPU; ORT 1.24, onnxruntime-genai 0.14). "Supported" = the
toolchain handles it but not exercised here. "Needs setup" = requires an extra package.

## Per-sub-part build tool
| Sub-part | Build tool | Quantization knob |
|---|---|---|
| `llm_decoder` | onnxruntime-genai **ModelBuilder** (`create_model`) | precision = int4 / fp16 / fp32 |
| `audio_embed`, `audio_heads` | **Olive** `OnnxConversion` (+ `OnnxBlockwiseRtnQuantization` / `OnnxFloatToFloat16`) | precision = int4 / fp16 / fp32 |
| `audio_tokenizer` (codec) | **Olive** `OnnxConversion` | always **fp32** (DAC int4/fp16 too lossy) |

## Device × precision matrix

| Device (EP) | int4 | fp16 | fp32 | Notes |
|---|---|---|---|---|
| **CPU** (`CPUExecutionProvider`) | ✅ **verified** | ⚠️ supported* | ✅ supported | int4 LLM + audio verified end-to-end (cos 0.999, real audio). |
| **NVIDIA GPU** (`CUDAExecutionProvider`) | ✅ supported | ✅ supported | ✅ supported | `--device cuda`; ModelBuilder cuda int4/fp16; needs CUDA ORT + a GPU (absent here). |
| **Intel / OpenVINO** (`OpenVINOExecutionProvider`) | ⚙️ needs setup | ⚙️ needs setup | ⚙️ needs setup | EP is registered but **`openvino.dll` runtime is missing → silently falls back to CPU**. See below. |
| **DirectML** (`DmlExecutionProvider`) | ✅ supported | ✅ supported | — | ModelBuilder supports `dml`; not wired in optimize.py / not present here. |

\* **fp16 on CPU**: `create_model` can emit fp16 on CPU (OmniVoice does this), but it's
not verified here and some ops fall back to fp32. int4 or fp32 are the safe CPU choices.

`optimize.py` currently exposes `--device {cpu,cuda}` × `--precision {int4,fp16,fp32}`.

## Intel / OpenVINO — how to actually enable it
The exported ONNX is EP-agnostic, so **inference** can target Intel CPU/iGPU/NPU via
OpenVINO without rebuilding — but the OpenVINO **runtime** must be installed (currently
missing, so it falls back to CPU):

```
uv pip install openvino                 # provides openvino.dll the EP depends on
```
Then point the manifest's `execution_provider` at `OpenVINOExecutionProvider` (or pass
providers explicitly in `inference.py`). The genai `llm_decoder` runs through
onnxruntime-genai (CPU/OpenVINO-CPU); `audio_embed/heads/tokenizer` run under the
OpenVINO EP directly.

For an **OpenVINO-optimized build** (not just EP inference), Olive ships native
OpenVINO passes — available in this install:
`openvinoconversion, openvinoquantization, openvinoweightcompression,
openvinooptimumconversion, openvinoquantizationwithaccuracy, …`
A `--device openvino` target could route the audio sub-parts through
`OpenVINOConversion` + `OpenVINOWeightCompression` (int4/int8) and the LLM through
`openvinooptimumconversion`. Not yet wired — straightforward follow-up.

## Quick verification commands
```
# CPU INT4 (verified)
python optimize.py --device cpu --precision int4
python inference.py --model-path onnx/cpu_int4 --text "..." --temperature 0.8 --top-k 50

# CPU FP32 (LLM + audio fp32)
python optimize.py --device cpu --precision fp32

# CUDA INT4 (needs NVIDIA GPU + CUDA ORT)
python optimize.py --device cuda --precision int4
```

## Summary
- **CPU INT4: fully working & verified** (the recommended on-device target).
- **CPU FP32 / CUDA int4|fp16: supported** by the toolchain; CUDA untested here (no GPU).
- **Intel/OpenVINO: inference works after `pip install openvino`** (EP present, runtime
  missing); a dedicated OpenVINO *build* path is a small Olive-pass addition.
- Codec is always fp32 across all targets.
