# OmniVoice ONNX Runtime Example

This example converts [Prince-1/OmniVoice](https://huggingface.co/Prince-1/OmniVoice) — a zero-shot TTS model supporting 600+ languages — to ONNX using Olive.

OmniVoice is **not a vision-language model**. It is a text-to-speech model using a Qwen3-0.6B backbone with 8-codebook audio codec output. The Olive pipeline is structured differently from VL models.

## Architecture

```
Text input
    │
    ▼
[audio_embeddings_encoder]  ← also fuses reference audio codes for voice cloning
    │  inputs_embeds (B, S, 1024)
    ▼
[llm_decoder]               ← Qwen3-0.6B, 28 layers, all full-attention
    │  hidden_states (B, S, 1024)          (exclude_embeds + exclude_lm_head)
    ▼
[audio_heads_decoder]       ← nn.Linear(1024 → 8×1025)
    │  logits (B, 8, S, 1025)
    ▼
32-step iterative unmasking
    │  audio_codes (8, T)
    ▼
[higgs_decoder]             ← Higgs Audio V2 Tokenizer (separate export)
    │
    ▼
Waveform @ 24 kHz
```

### Sub-model Details

| Sub-model | Input | Output | Export |
|-----------|-------|--------|--------|
| `audio_embeddings_encoder` | `(B,8,S)` ids + `(B,S)` mask | `(B,S,1024)` embeds | PyTorchModel → TorchScript → INT4/FP16 |
| `llm_decoder` | `(B,S,1024)` embeds | `(B,S,1024)` hidden | ModelBuilder (Qwen3ForCausalLM, `exclude_embeds=True, exclude_lm_head=True`) |
| `audio_heads_decoder` | `(B,S,1024)` hidden | `(B,8,S,1025)` logits | PyTorchModel → TorchScript → INT4/FP16 |
| **Higgs tokenizer** (4 models) | audio waveforms | codec codes | **Separate step** — requires `boson-multimodal` |

### Key Config Values (from `config.json`)

| Property | Value |
|----------|-------|
| LLM backbone | Qwen3-0.6B (28 layers, all full-attention — NO Gated DeltaNet) |
| `hidden_size` | 1024 |
| `num_codebooks` | 8 |
| `audio_vocab_size` | 1025 (1024 codes + 1 mask) |
| `audio_mask_id` | 1024 |
| `audio_codebook_weights` | [8, 8, 6, 6, 4, 4, 2, 2] |
| Decoding steps | 32 (iterative unmasking) |
| Output sample rate | 24 kHz |

## Prerequisites

```bash
pip install -r requirements.txt
```

## Steps

### 1. Export & Optimize Backbone

```bash
# CPU (INT4 — all three sub-models)
python optimize.py --device cpu

# CUDA (FP16 audio models, FP16 LLM)
python optimize.py --device gpu

# Skip ModelBuilder (if LLM already exported or for testing)
python optimize.py --skip-llm
```

**What `optimize.py` does:**
1. Loads OmniVoice (`trust_remote_code=True`) and saves the Qwen3 LLM as a standalone HF directory (`qwen3_standalone/`) so ModelBuilder can recognise it as `Qwen3ForCausalLM`
2. Runs Olive on all three backbone JSON configs
3. Writes `omnivoice_manifest.json` describing all sub-model paths and inference pipeline

### 2. Export Higgs Audio Tokenizer (voice cloning, optional)

The Higgs Audio V2 Tokenizer converts reference audio ↔ codec codes for voice cloning. It requires the `boson-multimodal` package:

```bash
# Install dependency
pip install boson-multimodal @ git+https://github.com/boson-ai/higgs-audio.git

# Download OmniVoice model locally
huggingface-cli download Prince-1/OmniVoice --local-dir ./omnivoice_model

# Export Higgs tokenizer parts
cd ./omnivoice_model
python convert_omnivoice_to_onnx.py --only higgs --out-dir ./higgs_onnx
```

This produces:
- `higgs_acoustic_encoder.onnx` — DAC encoder (waveform_24k → acoustic_features)
- `higgs_semantic_encoder.onnx` — HuBERT encoder (waveform_16k → semantic_features)
- `higgs_quantizer_encoder.onnx` — RVQ encoder (features → codes)
- `higgs_decoder.onnx` — RVQ + DAC decoder (codes → waveform_24k)

### 3. Run Inference

```bash
# Basic TTS (auto voice mode)
python inference.py --text "Hello, how are you today?" --output speech.wav

# CUDA
python inference.py --cuda --text "Hello!" --model_dir cuda/models --output hello.wav

# Voice cloning (requires Higgs tokenizer — see higgs_decoder integration below)
python inference.py --text "Hello world" --ref_audio ref.wav --ref_text "Reference text."
```

## Iterative Decoding Loop

OmniVoice uses 32-step iterative unmasking (non-autoregressive):

```python
# Pseudocode — all three backbone sub-models are called every step
for step in range(32):
    inputs_embeds = audio_embeddings_encoder(input_ids, audio_mask)
    hidden_states = llm_decoder(inputs_embeds, attention_mask, position_ids,
                                past_key_values=empty)   # no KV cache reuse
    logits        = audio_heads_decoder(hidden_states)   # (B, 8, S, 1025)
    
    # Sample + unmask highest-confidence positions using codebook weights
    unmask_positions(input_ids, logits, codebook_weights=[8,8,6,6,4,4,2,2])
```

Note: `past_key_values` inputs to the LLM should be empty tensors `(B, num_kv_heads, 0, head_dim)` — KV cache is not reused across steps (full-sequence forward each time).

## Directory Structure

```
Prince-1-OmniVoice/
└── builtin/
    ├── codes/
    │   ├── __init__.py
    │   └── model_wrappers.py          # AudioEmbeddingsEncoderWrapper, AudioHeadsDecoderWrapper
    ├── cpu_and_mobile/
    │   ├── audio_embeddings_encoder.json  # Olive: PyTorchModel → INT4
    │   ├── llm_decoder.json               # Olive: ModelBuilder INT4 (exclude_embeds + exclude_lm_head)
    │   ├── audio_heads_decoder.json       # Olive: PyTorchModel → INT4
    │   ├── qwen3_standalone/              # Saved by optimize.py (for ModelBuilder)
    │   └── models/                        # Generated ONNX models
    ├── cuda/
    │   ├── audio_embeddings_encoder.json  # Olive: FP16 + CUDA EP
    │   ├── llm_decoder.json               # Olive: ModelBuilder FP16 + CUDA EP
    │   ├── audio_heads_decoder.json       # Olive: FP16 + CUDA EP
    │   ├── qwen3_standalone/
    │   └── models/
    ├── user_script.py                     # Olive callbacks + save_qwen3_standalone()
    ├── optimize.py                        # Orchestrates full export pipeline
    ├── inference.py                       # ONNX Runtime inference (32-step loop)
    └── requirements.txt
```
