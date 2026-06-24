# Qwen3-TTS-VoiceDesign → ONNX (CPU INT4)

Convert **`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`** into sub-ONNX parts runnable on
CPU at INT4, with `optimize.py` (build), `inference.py` (run), `eval.py` (evaluate
vs original). LLM sub-part via **ModelBuilder**; audio/codec via **Olive**.

## Architecture (from config — not yet downloaded)

`Qwen3TTSForConditionalGeneration` (`model_type: qwen3_tts`, `auto_map` null — custom
class, not in stock transformers). Ships a separate `speech_tokenizer/` (config +
preprocessor). Base LLM is Qwen3-1.7B (ModelBuilder-friendly).

### Planned sub-model decomposition

| Sub-model | Source | Tool | Notes |
|---|---|---|---|
| **llm_decoder** | Qwen3-1.7B backbone | **ModelBuilder** INT4 | extract standalone Qwen3 HF dir |
| **speech_tokenizer (codec)** | `speech_tokenizer/` | Olive INT4/fp32 | 12 Hz speech codec encoder/decoder |
| **embeddings / heads** | TTS wrapper | Olive INT4 | text+audio embed fusion / audio-token heads |

## Target

CPU INT4 only (this cycle).

## Status

Scaffold only — see [`STATUS.md`](STATUS.md). Deep conversion deferred.
