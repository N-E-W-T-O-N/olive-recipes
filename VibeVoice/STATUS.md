# VibeVoice-1.5B — Current Status

> **Handoff:** read [HANDOFF.md](HANDOFF.md) — mental model, per-checkpoint table, the 25 traps, the new-checkpoint checklist, and the 5-minute operator card. Skill: `.claude/skills/vibevoice-onnx/`.

_Updated: 2026-09-21_

## 2026-09-21 — VibeVoice-ASR-Streaming-7B added (recon + registry + driver; LLM build needs a retarget)

New checkpoint `microsoft/VibeVoice-ASR-Streaming-7B` (`VibeVoiceForASRStreamingTraining`), local
at `VibeVoice/asr-streaming/`. Recon (HANDOFF §A): 28-layer Qwen2 GQA backbone (28 heads/4 kv,
hidden 3584) matching config exactly (no layer-count lie), top-level untied `lm_head.weight`,
acoustic tokenizer with BOTH encoder+decoder (decoder dead for ASR), semantic tokenizer
encoder-only, `diffusion_head_config` in `config.json` is vestigial (0 tensors named "diffusion"
in the index — pure ASR checkpoint, no audio generation). Weight layout turned out **identical to
`asr`** — verified by strict-loading (`strict=False`, assert 0/0) the acoustic tokenizer, semantic
tokenizer, and both connectors against this checkpoint using `asr`'s EXISTING loader functions,
unmodified: 0 missing / 0 unexpected on all four.

- **Registry**: added `"asr-streaming"` to `optimize.py`'s `MODELS` (reuses `extract_qwen2_asr` +
  `asr`'s Olive component specs), `HF_REPO`, `detect_model_type`, `EMBED_WEIGHT_KEY`; mirrored
  `MODEL_IDS`/`EMBED_KEY` in `common.py`. `optimize.py --list` and path auto-detection both confirm.
- **Protocol**: pulled the real inference code from GitHub (`vibevoice/modular/
  modeling_vibevoice_asr.py::streaming_generate`, commit `1541f59` — postdates our vendored
  303b283 pin, so this class isn't in the vendored tree) rather than guessing from
  `added_tokens.json`. Two findings that would have produced a silently-wrong driver otherwise —
  now documented as HANDOFF trap #25:
  1. The checkpoint's new-looking tokens (`<|AUDIO|>`/`<|audio_bos|>`/`<|audio_eos|>`) are **not
     used** — real speech markers are still the reused `<|object_ref_start|>`/`<|object_ref_end|>`
     grounding tokens (same as `asr`/`asr-hf`).
  2. `<|text_chunk_end|>` (id 151665) IS genuinely new and IS used, but the stock Qwen2.5-7B
     tokenizer `extract_qwen2_asr` fetches (and that ModelBuilder copies into the ONNX dir) does
     NOT have it — verified directly (`None` from the ONNX-dir tokenizer vs `151665` from the
     checkpoint dir's own tokenizer). `inference_asr_streaming.py` loads from the checkpoint dir
     specifically to avoid this.
- **New driver**: `inference_asr_streaming.py` — one persistent KV-cache session for the whole
  file; audio windowed by `chunk_frames`/`lookahead_frames` from `preprocessor_config.json`
  (22+4 frames = 2.933s chunk / 0.533s lookahead, re-encoded fresh each window, no chat template);
  per chunk feeds `[speech_start, fused_features, speech_end]`, greedy/temperature-decodes until
  `<|text_chunk_end|>`/EOS, then unconditionally steps the `<|text_chunk_end|>` embed once more
  (matches upstream) before the next chunk.
- **Build**: `--device cpu --precision fp16 --model asr-streaming asr-streaming` completed (all 6
  components + embeddings.onnx); `eval.py asr-streaming` → **7/7 PASS** (component parity cos
  ~1.0, codec round-trip corr +0.996/SNR +19dB). BUT: `llm_decoder.onnx` showed **0
  GroupQueryAttention / 28 MultiHeadAttention** — the known trap #20 broken op-selection for
  `(cpu, fp16)` on a GQA-shaped model (prefills fine, decode step can fail on a KV-buffer shape
  mismatch). `eval.py`'s llm check is structural (op counts) and doesn't catch this — matches the
  documented lesson exactly. **This specific cpu_fp16 build is NOT verified end-to-end** (a
  single-token decode smoke test was started but not confirmed complete before the build dir was
  removed). Next: rebuild with `cpu_int4`, `cpu_fp32`, `cuda_int4`, or `cuda_fp16` and run
  `inference_asr_streaming.py` for a real smoke test before shipping this checkpoint.

## 2026-07-27 (later) — VibeVoice-ASR eval 7/7 + WORKING inference (all 5 variants assembled)

User rebuilt `llm/` correctly (all 5 arch_ok, head 128/hidden 3584/28L; fp32≈27G, fp16≈14G, int4≈4.2G).
Formats match the precision×EP menu: **fused** = cpu_int4, cpu_fp32, **cuda_fp16** (GQA=28); **unfused**
(off-menu) = cpu_fp16, cuda_fp32. Moved all 5 decoders into `onnx/asr/<variant>/`, built fp32 + fp16
component sets, distributed so every variant is self-contained (6 onnx each).

**eval** `--device cpu --precision int4 asr` → **7/7 PASS** (llm structural MatMulNBits=141/GQA=28;
acoustic/semantic enc+dec+connectors parity; codec round-trip corr +0.996 / SNR 19 dB).

**inference** `inference_asr.py --audio en-Alice_woman.wav onnx/asr/cpu_int4` → **working transcript**:
`[{"Start":0,"End":9.27,"Speaker":0,"Content":"So, just to clarify, we've had 19 to 20 year olds…"}]`.
Two bugs fixed to get there:
1. **semantic_encoder baked static `[1,1,24000]`** (dynamo ignored `dynamic_axes`, trap #2) → any non-1s
   clip rejected. Fixed `get_semantic_tokenizer_encoder_io_config` to `dynamic_shapes` + rebuilt/redistributed
   to all 5 variants. (The asr-hf semantic io at ~line 358 still has the same `dynamic_axes` bug — fix when
   that model is next built.)
2. **ASR prompt protocol** — the driver used a bare "please transcribe" prompt → degenerate ", at, at…".
   VibeVoice-ASR needs the chat-templated JSON protocol (vendored `VibeVoiceASRProcessor`): system prompt
   "You are a helpful assistant that transcribes audio input into text output in JSON format." + user turn
   `<sp_start> <sp_pad>×N <sp_end>\nThis is a {dur}s audio, please transcribe it with these keys: Start
   time, End time, Speaker ID, Content` + generation prompt; inject `acoustic+semantic` fused features at
   the pad slots. **Reused tokens** (ASR ships no tokenizer): speech_start/end/pad = Qwen grounding tokens
   `<|object_ref_start|>`=151646 / `<|object_ref_end|>`=151647 / `<|box_start|>`=151648 — all in plain
   Qwen2.5, no vocab surgery. Also: `apply_chat_template(tokenize=True)` mangles those reused special
   tokens → render `tokenize=False` then `tok.encode(add_special_tokens=False)` (preserves single IDs).

## 2026-07-27 — VibeVoice-ASR assembled at onnx/asr (cpu variants) + cuda_fp16 diagnosis

Assembled the ASR pipeline from the pre-built `llm/` decoders + freshly built components. See
[onnx/asr/ASSEMBLY_NOTES.md](onnx/asr/ASSEMBLY_NOTES.md) for the full table/commands.
- **`onnx/asr/cpu_int4`** — ✅ complete & CPU-valid: int4 fused-GQA LLM (4.2 GB) + 5 fp32 components;
  all 6 ONNX load in an ORT CPU session. This is the one runnable-on-CPU ASR pipeline.
- **`onnx/asr/cpu_fp16`** — built 5 **fp16** components (Olive dynamo + fp16 pass, all load on CPU) +
  moved the fp16 LLM in. ⚠️ The fp16 LLM is the off-menu **fp16-CPU unfused** graph (GQA=0, 14 GB) →
  GPU asset only, not CPU-runnable.
- **`onnx/asr/cpu_fp32`** — fp32 components staged (copied from int4's fp32 set); fp32 7B LLM NOT built
  (needs ~30 GB; impractical on CPU). Command to finish is in ASSEMBLY_NOTES.
- LLM decoders **moved** (not copied) into their variant dirs to save disk.

**`llm/` review:** `cpu_int4` ✅ valid; `cpu_fp16` = full fp16 but UNFUSED (off-menu); `cpu_fp32` =
**wrong model** (hidden 896/20L ≈ 0.5B-class — junk); `cuda_fp32` = correct 7B arch but actually
**int4** (MatMulNBits+GQA), mislabeled.

**cuda_fp16 `create_model` "bug" — diagnosis:** NOT a code bug. `set_io_dtype` maps cuda+fp16 →
FLOAT16 fused GQA correctly, the combo is officially supported, and it **builds successfully here**
on the CPU-only genai 0.14.1 for a tiny Qwen2 repro (no GPU needed at build time). The 7B failure is
**resource-bound** (this box had ~14 GB free disk; a 7B build needs ~15 GB fp32 extraction + ~15 GB
fp16 output + RAM). Fix = build on a box with ~35 GB free disk / ~16–20 GB RAM via
`uv run optimize.py --device cuda --precision fp16 asr`. (Not convertible from `cuda_fp32`, which is
int4.) Refs: onnxruntime-genai builder.py `set_io_dtype`; supported-combos list; issues #881/#1137.

## 2026-07-17 — 1.5b rebuilt & re-verified on CPU (after checkpoints re-downloaded)

Checkpoints re-downloaded: **1.5B** (`model/`, 3/3 shards ✅) and **realtime** (`realtime/`, ✅)
complete; **asr** (3/8) and **asr-hf** (4/8) still downloading; `acoustic/` config-only. This box
is **CPU-only (no NVIDIA)**, so builds use `--device cpu` (LLM int4 — genai has no fp16-CPU).

Rebuilt all 7 **1.5b** sub-parts → `onnx/1.5b/cpu_int4/` (5.2 GB: llm_decoder int4 + acoustic
enc/dec + semantic enc + diffusion_head + acoustic/semantic connectors). `eval.py 1.5b` = **9/9
PASS** (llm: 140 MatMulNBits / 28 GQA / inputs_embeds; encoders/head/connectors cos 1.0;
acoustic_decoder cos 0.935 but max|Δ| 3.5e-10 = near-silent case; codec round-trip corr +0.996 /
SNR +19 dB; tts chain smoke OK). Driver smoke: `inference.py … onnx/1.5b/cpu_int4` → 2.13 s WAV.

Rebuilt all 4 **realtime** sub-parts → `onnx/realtime/cpu_int4/` (1.7 GB: llm_decoder int4 [20
layers, decoder-only] + acoustic_decoder + diffusion_head + acoustic_connector). `eval.py realtime`
= **5/5 PASS** (llm 100 MatMulNBits / 20 GQA / inputs_embeds; diffusion_head + connector cos 1.0;
acoustic_decoder cos 0.893 but max|Δ| 3.07e-10 near-silent; tts chain smoke H=896 OK). Driver smoke:
`inference_realtime.py … onnx/realtime/cpu_int4` → 2.13 s WAV. Scratch `qwen2_*_standalone/` removed.

Built + verified both **7B front-ends** on CPU (`--exclude-llm`; the 7B LLM int4 serialize is still
RAM-bound → needs a ≥32 GB-free box):
- **asr-hf** → `onnx/asr-hf/cpu_int4/` (acoustic_encoder + semantic_encoder + multi_modal_projector).
  `eval.py asr-hf` = **4/4 PASS** (encoders + projector cos 1.0; asr fusion chain fused=(1,7,3584);
  LLM/codec/tts-chain SKIP as expected without the LLM).
- **asr** → `onnx/asr/cpu_int4/` (acoustic enc/dec + semantic enc + acoustic/semantic connectors).
  `eval.py asr` = **6/6 PASS** (encoders/connectors cos 1.0; acoustic_decoder cos 0.935 maxd 3.9e-10
  near-silent; codec round-trip corr +0.996 / SNR +19 dB).

Added **`acoustic`** registry key — the standalone `vibevoice_acoustic_tokenizer` checkpoint
(`acoustic/`, 1.3 GB), transformers-native (`AutoModel`, no `codes/`). New loaders
`get_acoustic_std_{encoder,decoder}_model` (encode→`.latents`, decode→`.sample`); `all_components`
handles the no-LLM case; `detect_model_type` maps `vibevoice_acoustic_tokenizer`→`acoustic`. Built
→ `onnx/acoustic/cpu_fp32/`; `eval.py --precision fp32 acoustic` = **3/3 PASS** (encoder cos 0.999,
decoder cos 0.939 near-silent, codec round-trip corr +0.999 / SNR +22.9 dB).

All checkpoints downloaded: 1.5b, realtime, asr (17 GB), asr-hf (16 GB), standalone `acoustic` (1.3 GB).

`.gitmodules` fixed: `codes` submodule URL corrected `huggingface/vibevoice` → `microsoft/VibeVoice`
(pin 303b283) — the old URL left `codes/` empty on fresh clones → `ModuleNotFoundError:
vibevoice.modular.modular_vibevoice_tokenizer` in every codes/-backed loader.

Audio-quality caveat stands for the TTS builds — int4 hidden-state conditioning is lossy; fp16 would
need a GPU (genai has no fp16-on-CPU). Remaining: the three **7B ASR LLM decoders** (asr, asr-hf +
lm_heads) — build on a big-RAM machine; front-ends are done and `inference_asr.py` degrades cleanly.

### Later 2026-07-17 — fp16 eval fix, acoustic fp16/int4, git-dependency option

- **eval.py made dtype-aware** (`parity_component` + `whole_pipeline._feed`): feed each ONNX its
  DECLARED input dtype (fp16 graphs want float16; timesteps stay fp32). Fixes false "failures" on fp16
  builds. `1.5b/cpu_fp16` now evals **9/9** (it was fine all along — the harness fed fp32). HANDOFF trap 13.
- **1.5b CPU variants:** `cpu_int4` 9/9, `cpu_fp32` 9/9, `cpu_fp16` 9/9. `gpu_*` can't be built/run here
  (CPU-only box; `torch +cpu`, no CUDAExecutionProvider). The user-supplied fp16-on-CPU **LLM** is genai's
  unfused GQA×0 build needing `position_ids` (not driver-compatible) — a GPU fp16 build avoids that.
- **acoustic:** `fp32` 3/3 ✅; `fp16` rebuilt with `op_block_list=[ConstantOfShape,ConvTranspose,Resize,
  Range]` → 3/3 ✅ (ConstantOfShape had emitted fp16 into a float32 consumer — HANDOFF trap 11); `int4`
  removed — it's a no-op on a conv VAE, so `build_model` now **warns + downgrades int4→fp32** for
  LLM-less keys (HANDOFF trap 12).
- **`optimize.py` acoustic key** finalized: `HF_REPO["acoustic"]=microsoft/VibeVoice-AcousticTokenizer`,
  listed in `--help`/`--list`, transformers-native loaders (no `codes/`).
- **2026-07-22 — VibeVoice source VENDORED, submodule + git-dep removed (uploadable).** The required
  upstream subset (modular/, processor/, schedule/, scripts/, configs/ — ~591 KB, MIT/`VIBEVOICE_LICENSE`)
  now ships as `VibeVoice/vibevoice/` and (copied) `onnx/1.5b/vibevoice/`. `_vibevoice_dir()` returns that
  tree and raises if absent — no pip/git fallback. Removed: `codes/` submodule, `vibevoice-repo/` clone
  (~263 MB each), `.gitmodules`, the `vibevoice @ git+…` line in all three onnx pyprojects, and the
  root-pyproject `vibevoice` dep + `vibevoice-repo` workspace. Imports still use the isolated shim
  (trap 1). Verified: modular/schedule/processor all resolve from the vendored tree with codes/ deleted;
  the `onnx/1.5b/` package resolves its own copy. HANDOFF trap 14.

## 2026-07-17 — faithful 1.5B TTS voice-cloning inference built (from source)

No upstream 1.5B TTS `generate` exists (not in transformers — only `vibevoice_asr`/`vibevoice_acoustic_tokenizer`; not in the HF repo; `codes/` has only the training `forward`). Reconstructed it on the ONNX sub-parts, replicating from source:
- **Dynamic acoustic encoder** — fixed `dynamic_axes`→`dynamic_shapes` (trap #2); now emits `samples/3200` frames (was baked 24000 → 7 vs 7.5 drift), aligning with the processor's `speech_tok_compress_ratio=3200`. Also fixes the codec round-trip length drift.
- **Voice+prompt prefill** (`common.voice_prompt_embeds`) — runs `codes/` `VibeVoiceProcessor` (with trap-#1 shims: qwen2-fast alias + empty namespaces) → `input_ids` + `speech_input_mask` (70) + reference `speech_tensors`; acoustic-encodes the (24 kHz) voice, applies checkpoint `speech_scaling_factor`/`bias` (0.196/−0.049), connects, scatters the voice embeds into the masked positions. Replicates `forward_speech_features` (acoustic-only; TTS has no semantic tensors).
- **CFG negative** — parallel `<|image_pad|>` (id 151655) LLM context, prefilled + stepped alongside the positive one (from the streaming `generate`; NOT the old zero-hidden). `inference.py --voice` (default `samples/voices/en-Alice_woman.wav`).

Signal-level result (can't audition here): output went unconditioned→voiced→sharper as each piece landed — RMS 0.004→0.029, ZCR 0.110→0.052, centroid 1721→1065 Hz, sub-4 kHz energy 0.89→0.97. Intelligibility not verified (needs listening); remaining is quality/tuning, not missing machinery.

Notes: the voice path needs the **dynamic** acoustic_encoder (re-exported into `cpu_fp32` + `cpu_int4`). Inference now pulls `codes/` (processor + scheduler) — no longer onnxruntime-only, inherent to VibeVoice's prompt format.

### Working + tuned (later 2026-07-17)
Confirmed by listening — the pipeline produces **intelligible human speech** in the reference voice. Tuning applied:
- **fp32, not int4** — int4's raw-hidden quantization adds audible "hiss/background" (trap #5); fp32 is clear. (The `*_bgm.wav` sample voices carry real background music; `en-Alice_woman.wav` is clean.)
- **CFG negative = static `<|image_pad|>`** (prefill once, reuse) — cfg 1.3 preferred over 1.0; keeps voice while staying **single-session** (fp32 ~5 GB, not ~10 GB).
- **Auto frame-length** (`--max-frames 0`): ~5 frames/word (no EOS classifier is exported → fixed budget, not auto-stop). Prevents truncation of long text.
- **Sentence chunking**: split on `[.!?]`, generate each chunk short, concat (150 ms gaps) → keeps each generation in the stable short regime.

**Resolved — the generate works** (single-shot; earlier "chunking" was a regression and is removed). Confirmed intelligible on prose AND a Shakespeare sonnet with the reference voice. Final recipe:
- **Single-shot** generation (whole prompt in one context) — chunking split phrasing and made some inputs noise; reverted.
- **Learned EOS (the key fix)** — the 1.5B `lm_head` is **tied to `embed_tokens`**, so `logits = hidden @ embedᵀ` needs no lm_head export. VibeVoice reuses vision tokens for speech; generation **stops when lm_head predicts `<|vision_end|>` (speech-end) or `<|endoftext|>` (EOS)**. This removed the "jargon tail" at the source (e.g. sonnet stopped at frame 101 of a 172 budget → 13.5 s not 22.9 s). `--max-frames` is now just a safety cap. **So "no EOS exported" is NOT a limitation** — it's recovered from the tied weights.
- **fp32, not int4** for clarity (int4 hidden-quant → hiss). **static `<|image_pad|>` CFG negative**, cfg 1.3. Voice-conditioned prefill (processor + acoustic splice + scale/bias).

**Remaining minor** (1.5B model quality, not pipeline): approximate timbre cloning (gender not always captured); occasional pronunciation quirks ("boy"→"bow") and weak digit reading; fp32 CPU ~2–3 s/frame (GPU `gpu_fp16` far faster). The 7B model would improve fidelity.

## Status matrix (model × component)

Legend: ✅ converted & parity-verified · ⚠️ pending / partial / memory-bound · ❌ not started or blocked · — not applicable

| Model | LLM decoder | Acoustic enc | Acoustic dec | Semantic tok | Diffusion head | Connectors / projector |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **VibeVoice-1.5B** (TTS) | ✅ int4 | ✅ cos 1.0 | ✅ cos 1.0 | ✅ cos 1.0 (enc) | ✅ cos 1.0 | ✅ cos 1.0 (ac+sem) |
| **VibeVoice-ASR-HF** (7B) | ⚠️ extracted, int4 OOM | ✅ cos 1.0 | — | ✅ cos 1.0 | — | ✅ cos 1.0 (mm_projector) |
| **VibeVoice-ASR** (7B) | ⚠️ 7B int4 OOM | ✅ cos 1.0 | ✅ cos 1.0² | ✅ cos 1.0 | — none | ✅ cos 1.0 (ac+sem) |
| **VibeVoice-ASR-Streaming** (7B) | ⚠️ fp16 built but trap #20 (MHA not GQA) — needs int4/fp32/cuda | ✅ cos 1.0 | ✅ cos 1.0² | ✅ cos 1.0 | — none | ✅ cos 1.0 (ac+sem) |
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
