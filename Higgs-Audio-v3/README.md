# Higgs-Audio-v3 → ONNX (CPU INT4)

Convert **`bosonai/higgs-audio-v3-tts-4b`** into sub-ONNX parts runnable on CPU at
INT4, with `optimize.py` (build), `inference.py` (run), `eval.py` (evaluate vs
original). LLM sub-part via **ModelBuilder**; audio tokenizer via **Olive**.

## Architecture (from config — not yet downloaded)

`HiggsMultimodalQwen3ForConditionalGeneration` (`model_type: higgs_multimodal_qwen3`,
`auto_map` null — custom class). Needs the `boson-multimodal` package for the audio
tokenizer.

- **text_config: qwen3** — hidden 2560, 36 layers, 32 heads, 8 kv, vocab 151936
  → Qwen3-4B backbone, ModelBuilder-friendly.

### Planned sub-model decomposition

| Sub-model | Source | Tool | Notes |
|---|---|---|---|
| **llm_decoder** | Qwen3-4B backbone | **ModelBuilder** INT4 | extract standalone Qwen3 HF dir; `exclude_embeds` (multimodal fusion outside LLM) |
| **audio_tokenizer** | Higgs Audio V2/V3 tokenizer | Olive INT4/fp16 | acoustic encoder, semantic encoder, quantizer, decoder (mirror OmniVoice `higgs/`) |
| **embeddings / audio heads** | multimodal wrapper | Olive INT4 | text+audio embed fusion / per-codebook logits |

Closely related to the OmniVoice project (same Higgs tokenizer family) — reuse its
`higgs/` recipes and `user_script.py` audio wrappers.

## Target

CPU INT4 only (this cycle). Largest of the four (~8 GB) — build one sub-model at a
time and clean the Olive cache between (disk constraint).

## Status

Scaffold only — see [`STATUS.md`](STATUS.md). Deep conversion deferred.
