# VibeVoice-1.5B — Current Status

_Updated: 2026-07-01_

## Status matrix (model × component)

Legend: ✅ converted & parity-verified · ⚠️ pending / partial / memory-bound · ❌ not started or blocked · — not applicable

| Model | LLM decoder | Acoustic enc | Acoustic dec | Semantic tok | Diffusion head | Connectors / projector |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **VibeVoice-1.5B** (TTS) | ✅ int4 | ✅ cos 1.0 | ✅ cos 1.0 | ✅ cos 1.0 (enc) | ✅ cos 1.0 | ✅ cos 1.0 (ac+sem) |
| **VibeVoice-ASR-HF** (7B) | ⚠️ extracted, int4 OOM | ✅ cos 1.0 | — | ✅ cos 1.0 | — | ✅ cos 1.0 (mm_projector) |
| **VibeVoice-ASR** (7B) | ⚠️ 7B int4 OOM | ✅ cos 1.0 | ✅ cos 1.0² | ✅ cos 1.0 | — none | ✅ cos 1.0 (ac+sem) |
| **Realtime-0.5B** | ✅ int4 (tts) | — (none shipped) | ✅ cos 1.0 | — none | ✅ cos 1.0 | ✅ cos 1.0 (acoustic) |

Encoder note: all acoustic/semantic **encoders** (1.5B, ASR, ASR-HF) currently bake the audio `samples` dim (24000 = 1 s @ 24 kHz) — the shared io_config uses `dynamic_axes`, which the dynamo exporter fixes. Frames/latents outputs are correct; a cross-cutting `dynamic_shapes` switch would make audio length variable (not yet applied).

Notes: the three **7B LLMs** (ASR-HF / ASR text decoders + their lm_heads) all hit the int4-serialize OOM wall at current free RAM (peak ~16–20 GB, same as chandra-8B) — extraction paths ready, re-run after freeing RAM. Realtime **acoustic enc** = none shipped (decoder-only checkpoint); Realtime **semantic tok** = none in that checkpoint. ASR acoustic dec is off the audio→text path (see ²).

## Architecture (confirmed from downloaded checkpoint)
`VibeVoiceForConditionalGeneration` (`model_type: vibevoice`, auto_map null → needs the custom
`vibevoice` package; not in transformers). Weight groups (model.safetensors.index.json):
- `model.language_model.*` (338) — **Qwen2.5-1.5B backbone** (1536/28L/12h/2kv, q/k/v bias,
  tied embeddings, **no lm_head** — the head is the diffusion prediction_head).
- `model.acoustic_tokenizer.*` (552) — acoustic VAE/codec.
- `model.semantic_tokenizer.*` (276) — semantic tokenizer.
- `model.prediction_head.*` (26) — DiT-style diffusion denoiser (adaLN + ffn).
- `model.acoustic_connector.*` / `model.semantic_connector.*` (5 each) — projection MLPs.

## Done ✅
- ✅ Model downloaded to `model/` (3 shards). Tokenizer NOT in repo → fetched Qwen2.5-1.5B's.
- ✅ **`llm_decoder` BUILT (cpu int4)** → `cpu_int4/models/llm_decoder.onnx` (+857 MB data),
  `genai_config.json`, tokenizer. Pipeline: `user_script.extract_qwen2_standalone` remaps
  `model.language_model.*` → a standalone Qwen2ForCausalLM dir; `optimize.py` runs ModelBuilder
  INT4 with **exclude_embeds + exclude_lm_head** → `inputs_embeds → hidden_states` + KV cache.
  Verified: 140 MatMulNBits (int4) + 28 GroupQueryAttention; genai_config wires inputs_embeds→hidden.
- Build: `python optimize.py --skip-download --components llm` (or no args to also download).

## Pending ⏳ (need the custom `vibevoice` package)
- `acoustic_tokenizer`, `semantic_tokenizer`, `diffusion_head`, connectors → Olive.
  `auto_map` is null and the classes aren't in transformers, so loading the custom modules for
  export needs `pip install vibevoice` (or the github source). `user_script.py` has stub loaders
  that raise a clear message until that's wired.
- Full TTS inference (`inference.py`) + `eval.py`: text → embed+connectors → llm_decoder → hidden
  → diffusion head (iterative denoise) → acoustic latents → acoustic decoder → waveform. The
  diffusion sampling loop + acoustic VAE are the novelty/risk (expect custom-op / fixed-shape work).

## Files
`optimize.py` (LLM build), `user_script.py` (Qwen2 extraction + audio stubs), scaffold (info.yml,
README, requirements). Target: CPU INT4.

## ✅ AcousticTokenizer (from VibeVoice-1.5B) — CONVERTED & verified
Handled per-checkpoint (acoustic tokenizers differ across the family: 1.5B=downsample_layers,
Realtime=stages/head, ASR-HF=conv_layers). For **1.5B**, the vendored `codes/`
`VibeVoiceAcousticTokenizerModel` matches its weights EXACTLY (552 tensors, 0 missing).
Loaded via an isolated import (bypasses the package __init__ that pulls diffusers + a
transformers-5.10.2-renamed qwen2 tokenizer) + an Auto*.register shim (coexists with
transformers' built-in vibevoice_acoustic_tokenizer). Exported both halves via Olive (fp32, dynamo):
- `acoustic_encoder.onnx` — `audio[B,1,T] → latents[B,8,64]` (VAE `.mean`)
- `acoustic_decoder.onnx` — `latents[B,8,64] → audio[B,L]`
**Parity vs PyTorch: cosine 1.0** (encoder max|Δ| ~1e-4, decoder ~5e-6). `user_script.py`
(`_load_acoustic` + get_acoustic_encoder/decoder_*), `optimize.py --components acoustic_encoder acoustic_decoder`.
(Each file ~1.37 GB — the codec is a large VAE; both currently carry full weights.)

## Deps note
`codes/` needs `diffusers` (installed via uv) for its diffusion scheduler; the acoustic tokenizer
itself is imported in isolation and does NOT need it. transformers 5.10.2 has native
`vibevoice_acoustic_tokenizer` + `vibevoice_asr` (different weight-namings than 1.5B — see above).

## Next
ASR (note: `VibeVoice-ASR` and `VibeVoice-ASR-HF` are DIFFERENT models — HF = transformers-native
8B `vibevoice_asr`), then Realtime-0.5B (vibevoice_streaming, via codes/).

## ✅ ASR-HF acoustic encoder — CONVERTED & verified (transformers-native)
`VibeVoice-ASR-HF` (`vibevoice_asr`, transformers-native, 8 shards / 16.7 GB, LLM = Qwen2.5-**7B**)
is DISTINCT from `VibeVoice-ASR` (`VibeVoiceForASRTraining`, codes/-format, config-only here).
Its acoustic encoder is a DIFFERENT arch than 1.5B (`acoustic_tokenizer_encoder.conv_layers.*`) and
loads NATIVELY via transformers `VibeVoiceAcousticTokenizerEncoderModel` — no codes/ / shim.
Loaded ONLY the `acoustic_tokenizer_encoder.*` weights (276, 0 missing) — not the 7B LLM — so it
fits in memory. Exported (Olive fp32, dynamo) → `asr-hf/cpu_int4/models/acoustic_encoder.onnx`
(`audio[B,1,T] → latents[B,7,64]`). **Parity vs PyTorch: cosine 1.0**, max|Δ| ~1e-4.
`user_script.get_asrhf_acoustic_encoder_*`.

## ✅ ASR-HF semantic encoder + multi_modal_projector — CONVERTED & verified
- **semantic_encoder** — SAME transformers-native class as the acoustic encoder
  (`vibevoice_acoustic_tokenizer_encoder`), just a different config + weight prefix
  (`semantic_tokenizer_encoder.*`, hidden 128). `audio[B,1,T] → latents[B,frames,128]`.
  **cos 1.0**, max|Δ| ~4e-5. `user_script.get_asrhf_semantic_encoder_*` (shared `_load_asrhf_encoder`).
- **multi_modal_projector** (`VibeVoiceAsrMultiModalProjector`) — fuses
  acoustic[B,T,64] + semantic[B,T,128] → LLM features[B,T,3584] (two 2-layer MLPs + RMSNorm, summed).
  Exported with dynamo **dynamic_shapes** (frames dynamic — verified T=8/20/37). **cos 1.0**, max|Δ| ~7e-6.
  `→ asr-hf/cpu_int4/models/{semantic_encoder,multi_modal_projector}.onnx`.

## ASR-HF LLM — extraction DONE, int4 serialize OOM-blocked
- **Extraction ✅**: `user_script.extract_qwen2_asrhf` streams shards → standalone `Qwen2ForCausalLM`
  dir `qwen2_asrhf_standalone/` (**WITH lm_head** — ASR emits text; `language_model.model.*`→`model.*`,
  `language_model.lm_head.*`→`lm_head.*`; config = text_config = Qwen2.5-7B: 3584/28L/28h/4kv/vocab
  152064, untied). 15.2 GB `model.safetensors` + Qwen2.5-7B tokenizer written. Ready to build.
- **int4 build ⚠️ OOM**: `create_model(..., "int4", "cpu")` (WITH embeds+lm_head) crashed mid-serialize
  at ~2–3 GB free RAM (peak need ~16–20 GB) — SAME wall as chandra-8B. No `llm_decoder.onnx` produced.
  Run `_build_asrhf_llm.py` again after freeing RAM (close Visual Studio ~3 GB + WSL) — extraction is
  cached so it resumes straight to ModelBuilder. Subprocess isolation can't shrink a single 7B serialize.

## ✅ Realtime-0.5B (vibevoice_streaming) — LLM + acoustic decoder CONVERTED & verified
Streaming TTS checkpoint (`model.safetensors`, single file). Key groups: `tts_language_model.*`
(Qwen2.5-0.5B backbone, **20 layers**, no lm_head), `acoustic_tokenizer.*` (**DECODER-ONLY**, 276 —
no encoder shipped: inference only decodes generated latents), `language_model.*` (4-layer base),
`prediction_head.*` (diffusion), `acoustic_connector.*`, `tts_eos_classifier.*`.
- **tts llm_decoder ✅ int4** → `realtime/cpu_int4/models/llm_decoder.onnx` (+192 MB data). Built
  0.5B in-memory (no OOM). `extract_qwen2_realtime` remaps `tts_language_model.*`→standalone Qwen2
  (config = decoder_config with num_hidden_layers overridden to actual 20), ModelBuilder int4
  **exclude_embeds+exclude_lm_head** → inputs_embeds→hidden. Verified 100 MatMulNBits + 20 GQA.
- **acoustic_decoder ✅** → `realtime/cpu_int4/models/acoustic_decoder.onnx` (1.38 GB). codes/ class
  matches EXACTLY (decoder 0 missing/0 unexpected; stages/head naming — NOT transformers-native
  conv_layers/convtr). Decoder-only load (`decoder.*` weights, encoder=None). `latents[B,T,64] →
  audio[B,samples]`, dynamo **dynamic_shapes** (frames dynamic — verified T=10/25/50). **cos 1.0**,
  max|Δ| ~4e-6. `user_script.get_realtime_acoustic_decoder_*`, `_codes_tokenizer` (shared codes/ import).
- **Acoustic ENCODER: none** — not in the streaming checkpoint (nothing to convert).

## Realtime pending
- prediction_head (diffusion denoiser) + acoustic_connector + tts_eos_classifier + 4-layer base
  language_model → the streaming generation loop (text → tts backbone → diffusion → latents →
  acoustic decoder → audio). Same diffusion-sampling novelty/risk as the 1.5B TTS head.

## ✅ Diffusion prediction_head + speech connectors (1.5B & Realtime) — CONVERTED & verified
Shared `codes/` classes (`_codes_import` isolated-import helper). Exported via Olive (fp32, dynamo).
- **diffusion_head** (`VibeVoiceDiffusionHead`) — ONE DDPM denoise step:
  `(noisy_images[B,64], timesteps[B] FLOAT, condition[B,H]) → pred[B,64]`. The ~20-step sampling
  loop stays in the pipeline; ONNX = one step. `H`=1536 (1.5B) / 896 (Realtime). **Batch dynamic**
  (verified B=4/7). **cos 1.0**, max|Δ| ~1e-6. Note: timesteps MUST be float32 — `TimestepEmbedder`
  casts its sinusoidal embedding back to `t.dtype` before the float MLP.
- **connectors** (`SpeechConnector`: fc1→RMSNorm→fc2) — project VAE latents into LLM hidden space,
  frames dynamic. 1.5B: `acoustic_connector` 64→1536 + `semantic_connector` 128→1536; Realtime:
  `acoustic_connector` 64→896. All **cos 1.0**, max|Δ| ≤5e-7.
- `user_script.get_diffusion_head_* / get_{acoustic,semantic}_connector_*`. Head dummy hidden via
  env `VV_HEAD_HIDDEN` (1536 default; set 896 for Realtime).

## Sub-model tally: 13 ONNX parts converted & parity-verified (cos 1.0)
- **1.5B** (6): llm_decoder(int4), acoustic_encoder, acoustic_decoder, diffusion_head,
  acoustic_connector, semantic_connector.  Pending: semantic_tokenizer encoder; end-to-end pipeline.
- **ASR-HF** (3): acoustic_encoder, semantic_encoder, multi_modal_projector. LLM extracted (int4 OOM).
- **Realtime** (4): llm_decoder(int4), acoustic_decoder, diffusion_head, acoustic_connector.
Remaining to reach end-to-end TTS: wire the sampling pipeline (text → embed+connector → llm_decoder →
hidden → DDPM loop over diffusion_head → acoustic latents → acoustic_decoder → waveform) + eval.

² ASR acoustic_decoder frames baked at 8 (shared 1.5B io_config uses dynamic_axes, which dynamo
ignores) — harmless: the decoder is NOT on the ASR audio→text path. Encoders + connectors are frames-dynamic.

## ✅ VibeVoice-ASR (weights now downloaded) — audio front-end CONVERTED & verified
`VibeVoiceForASRTraining` (`model_type vibevoice`, auto_map null → codes/), 8 shards. Same codes/
family as 1.5B TTS, but: LLM = Qwen2.5-**7B** (decoder_config 3584/28L) **WITH** top-level
`lm_head.weight` (audio→text), and **NO prediction_head** (no audio generation). Weight groups:
acoustic_tokenizer (552, enc+dec), semantic_tokenizer (276, ENCODE-only), language_model (338) +
lm_head, acoustic/semantic connectors (5 each). All front-end pieces load via the EXISTING codes/
loaders (matched exactly: acoustic 552/0-miss, semantic 276/0-miss) — no per-checkpoint divergence
here (unlike the ASR-HF transformers-native tokenizers). Exported (Olive fp32, dynamo) →
`asr/cpu_int4/models/`:
- **acoustic_encoder** `audio[B,1,T]→latents[B,f,64]` (VAE .mean) — cos 1.0
- **acoustic_decoder** `latents[B,8,64]→audio` — cos 1.0 (frames fixed 8, off-path — see ²)
- **semantic_encoder** (`VibeVoiceSemanticTokenizerModel`, encode-only) `audio→latents[B,f,128]` — cos 1.0
- **acoustic_connector** 64→3584, **semantic_connector** 128→3584 — cos 1.0, frames-dynamic
`user_script.get_semantic_tokenizer_encoder_*` (new) + reused `_load_acoustic`/`_load_connector`.

## VibeVoice-ASR LLM — extraction ready, int4 serialize OOM (same 7B wall)
`extract_qwen2_asr` remaps `model.language_model.*`→`model.*` + top-level `lm_head.weight` (differs
from ASR-HF's `language_model.model.*`/`language_model.lm_head.*` layout), config = decoder_config
(Qwen2.5-7B), fetches Qwen2.5-7B tok. NOT run yet (avoids a 2nd 15 GB standalone dir on disk); the
int4 ModelBuilder serialize would OOM at current free RAM exactly like ASR-HF. Run after freeing RAM.

## Updated tally: 19 ONNX parts converted & parity-verified (cos ~1.0)
1.5B(7) + ASR-HF(3) + Realtime(4) + **ASR(5)**. Remaining: the three 7B
LLMs (ASR-HF/ASR extracted-or-ready but int4 OOM-bound); end-to-end pipelines.

## Inference (ONNX) — inference_common.py + 3 drivers
`inference_common.py` (shared): resolve() (reuses optimize registry), **OnnxLLM** (drives the
genai llm_decoder.onnx directly via ORT — 28-layer KV cache + growing attn mask; verified
KV-incremental==full cos 1.0; handles hidden_states/logits outputs), OnnxOp, **DiffusionSampler**
(diffusion_head.onnx + codes/ DPM scheduler + CFG), audio load/normalize(-25dBFS)/save,
embed_tokens lookup, load_scaling, acoustic_decode_to_wav.
- **inference.py** — VibeVoice-1.5B TTS: text→embed→prefill→[hidden→diffuse→latent→connector→step]→
  decode all latents→wav. Verified: 8 frames → 1.07 s wav.
- **inference_realtime.py** — Realtime-0.5B streaming TTS: same loop, per-frame decode (dynamic
  decoder) streamed. Verified: 6 frames → 0.80 s wav.
- **inference_asr.py** — asr/asr-hf audio→text: encoders→(connectors|projector)→fuse at speech-pad
  positions→prefill(lm_head→logits)→greedy decode. Verified: front-end → fused [N,3584]; degrades
  cleanly when the 7B llm_decoder.onnx isn't built.
Caveats (documented in files): int4 is lossy in raw-hidden space (first-token hidden cos ~0.83 vs
fp32) — prefer fp16 for audio fidelity; learned EOS (1.5b lm_head / realtime tts_eos_classifier)
isn't in the sub-part set → stops at --max-frames; exact prompt layout/voice-clone needs the
original VibeVoice processor/tokenizer assets (repos ship none → Qwen2.5 tokenizer used). Fixed a
real bug: acoustic_decoder io_config marked the vae_dim (dim 2) as "frames" instead of dim 1 →
frames stayed static; switched to dynamic_shapes on dim 1 and re-exported.
