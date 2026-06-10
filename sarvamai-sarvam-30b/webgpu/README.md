# sarvamai/sarvam-30b — WebGPU optimization

This folder contains Olive recipes for optimizing sarvamai/sarvam-30b targeting the WebGPU EP.

## About the model

- **Architecture**: Custom Mixture-of-Experts (`sarvam_moe`)
- **Total parameters**: ~30B (32B)
- **Active parameters per token**: ~2.4B
- **Languages**: Multilingual, optimized for Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia, and English)
- **HuggingFace**: [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b)

## What this folder is for

- Execution Provider: WebGPU EP
- Typical precision: INT4 precision by default
- Example recipe filename: sarvamai-sarvam-30b_webgpu_int4.json

## Setup

1) Install the main branch of Olive:
   ```
   pip install git+https://github.com/microsoft/olive.git
   ```
2) Install the appropriate runtime package for this backend:
   ```
   pip install onnxruntime-web
   ```
3) Run Olive to build/optimize the model:
   ```
   olive run --config sarvamai-sarvam-30b_webgpu_int4.json
   ```

## Additional notes

- Pipeline: `SelectiveMixedPrecision` (k_quant_mixed) → `GPTQ` → `RTN` (8-bit lm_head/embeddings) → `ModelBuilder`
- Uses `k_quant_mixed` instead of `kld_gradient` because gradient-based sensitivity
  estimation exceeds available GPU memory for a ~30B parameter model.
- GPTQ group size: **32** (smaller than CPU/CUDA to respect WebGPU buffer alignment constraints)
- WebGPU enables GPU-accelerated inference in web browsers.
- Ensure your browser supports WebGPU (Chrome 113+, Edge 113+).
- **Custom architecture**: `sarvam_moe` is a proprietary MoE architecture. If `ModelBuilder`
  does not yet support this model type, remove the `m` pass from the recipe and use the
  GPTQ + RTN quantized ONNX output directly.
- Note: a 30B MoE model is very large for in-browser inference. Expect high VRAM usage
  and long load times even with INT4 quantization.

---

This README was auto-generated for the WebGPU EP of sarvamai/sarvam-30b.
