# VibeVoice-1.5B → ONNX (CPU INT4)

Convert **`microsoft/VibeVoice-1.5B`** into sub-ONNX parts runnable on CPU at INT4,
with `optimize.py` (build), `inference.py` (run), `eval.py` (evaluate vs original).
LLM sub-part via **ModelBuilder**; acoustic tokenizer + diffusion head via **Olive**.

## Architecture (from config — not yet downloaded)

`VibeVoiceForConditionalGeneration` (`model_type: vibevoice`, `auto_map` null — custom
class). VibeVoice pairs an LLM with a **diffusion acoustic head** and an acoustic
tokenizer.

- **decoder_config: qwen2** — hidden 1536, 28 layers, 12 heads, 2 kv, vocab 151936
  → Qwen2-1.5B backbone, ModelBuilder-friendly.

### Planned sub-model decomposition

| Sub-model | Source | Tool | Notes |
|---|---|---|---|
| **llm_decoder** | Qwen2-1.5B backbone | **ModelBuilder** INT4 | extract standalone Qwen2 HF dir |
| **acoustic_tokenizer** | VibeVoice acoustic VAE/codec | Olive INT4/fp32 | encoder + decoder |
| **diffusion_head** | diffusion acoustic head | Olive fp32 | conv/transformer denoiser — may need fixed-shape export |

The diffusion head is the main novelty/risk — expect the same data-dependent /
custom-op issues seen with the Mimi codec; likely needs static shapes + explicit masks.

## Target

CPU INT4 only (this cycle). Smallest of the four (~3 GB).

## Status

Scaffold only — see [`STATUS.md`](STATUS.md). Deep conversion deferred.
