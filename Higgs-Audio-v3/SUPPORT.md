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
`optimize.py --device openvino` builds the **same EP-agnostic ONNX as CPU** (genai LLM
on CPU int4; audio parts via Olive on CPU) and writes a manifest tagged
`OpenVINOExecutionProvider` → `onnx/openvino_int4/`. No rebuild needed to switch EPs.

**Running on OpenVINO is an environment swap, not just `pip install openvino`:**
- The OpenVINO EP ships **only** in the `onnxruntime-openvino` wheel — a *full*
  onnxruntime build. It and plain `onnxruntime` install into the same namespace and
  **cannot coexist**: a mismatched pair hard-fails every ORT session with
  `Error 127: The specified procedure could not be found` (we hit exactly this with
  `onnxruntime` 1.26 + a stale `onnxruntime-openvino` 1.24.1 dll — it broke even CPU
  builds until removed).
- Correct setup (in a dedicated env): install a **single matching set** —
  `onnxruntime-openvino==<ver>` (replacing plain `onnxruntime`) + a compatible
  `openvino` runtime — and re-check `onnxruntime-genai` still imports (it depends on
  onnxruntime). Then `OpenVINOExecutionProvider` appears in
  `ort.get_available_providers()` and the audio ONNX runs on Intel CPU/iGPU/NPU.
- This repo's default env uses plain `onnxruntime` (CPU/CUDA) + `onnxruntime-genai`;
  OpenVINO is opt-in via the swap above.

### Choosing the OpenVINO device (`--ov-device`)
`optimize.py --device openvino --ov-device {CPU,GPU,NPU,AUTO}` writes the chosen
`device_type` into `genai_config.json` (`{"OpenVINO": {"device_type": ...}}`) and the
manifest (`ov_device_type`). `inference.py` passes it through when the OpenVINO EP is
present; otherwise it cleanly falls back to CPU.

### Intel NPU (`--ov-device NPU`)
Requirements: **Intel NPU driver** + `openvino` runtime (with the NPU plugin) +
a matching `onnxruntime-openvino` (the env swap above).

Honest caveats — the NPU is great for the **small audio sub-parts**, less so for the
4B LLM:
- **Static shapes:** the NPU plugin wants fixed input shapes. Our audio_embed/heads and
  codec use dynamic axes (seq/frames); the NPU may recompile per shape or reject them.
  The codec already exports with a static time axis; the LLM/audio seq axes are dynamic.
- **Precision:** NPU targets **fp16/int8**, not int4 GQA. The 4B **int4 `llm_decoder`
  is unlikely to run natively on NPU.**
- **Recommended: `--ov-device AUTO`** (or `AUTO:NPU,CPU`) so OpenVINO runs what the NPU
  supports there and offloads the rest (the int4 LLM) to CPU/GPU automatically. Pure
  `NPU` for the whole pipeline will likely error on the LLM.
- Best realistic split: audio sub-parts on **NPU**, `llm_decoder` on **CPU/GPU**. That
  needs per-sub-part EPs (one `device_type` per session) — a small follow-up if you
  want it; today the manifest uses one `device_type` for all parts.

Build a NPU-targeted bundle:
```
python optimize.py --device openvino --precision int4 --ov-device NPU   # → onnx/openvino_int4/
# (or --ov-device AUTO for NPU-with-CPU-fallback, recommended for the 4B LLM)
```

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
