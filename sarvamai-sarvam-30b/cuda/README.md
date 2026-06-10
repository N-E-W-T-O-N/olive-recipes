# sarvamai/sarvam-30b — CUDA optimization

This folder contains Olive recipes for optimizing sarvamai/sarvam-30b targeting the CUDA EP.

## About the model

- **Architecture**: Custom Mixture-of-Experts (`sarvam_moe`)
- **Total parameters**: ~30B (32B)
- **Active parameters per token**: ~2.4B
- **Languages**: Multilingual, optimized for Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia, and English)
- **HuggingFace**: [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b)

## What this folder is for

- Execution Provider: CUDA EP
- Typical precision: INT4 precision by default
- Example recipe filename: sarvamai-sarvam-30b_cuda_int4.json

## Setup

1) Install the main branch of Olive:
   ```
   pip install git+https://github.com/microsoft/olive.git
   ```
2) Install the appropriate runtime package for this backend:
   ```
   pip install onnxruntime-genai-cuda
   ```
3) Run Olive to build/optimize the model:
   ```
   olive run --config sarvamai-sarvam-30b_cuda_int4.json
   ```

## Additional notes

- Pipeline: `SelectiveMixedPrecision` (k_quant_mixed) → `GPTQ` → `RTN` (8-bit lm_head/embeddings) → `ModelBuilder`
- Uses `k_quant_mixed` instead of `kld_gradient` because gradient-based sensitivity
  estimation exceeds available GPU memory for a ~30B parameter model.
- GPTQ group size: 128
- Requires an NVIDIA GPU with CUDA support.
- Ensure CUDA toolkit and cuDNN are properly installed.
- **Custom architecture**: `sarvam_moe` is a proprietary MoE architecture. If `ModelBuilder`
  does not yet support this model type, remove the `m` pass from the recipe and use the
  GPTQ + RTN quantized ONNX output directly.
- Due to the MoE structure, only ~2.4B parameters are active per forward pass, which
  reduces VRAM pressure significantly compared to a dense 30B model.

---

This README was auto-generated for the CUDA EP of sarvamai/sarvam-30b.
