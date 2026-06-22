# Higgs-Audio-v3 — Current Status

_Updated: 2026-06-10_

## Done ✅
- Project scaffold: `info.yml`, `requirements.txt`, `.gitignore`, `README.md`.
- **LLM sub-part built and verified** (the priority):
  - `optimize.py` downloads the checkpoint → `model/`, extracts a standalone
    Qwen3-4B-Base (`body.* / tied.embedding.text_embedding` remap, **no
    boson-multimodal needed**), and runs **ModelBuilder INT4**.
  - Output: `cpu_int4/models/llm_decoder/` — `model.onnx` + `model.onnx.data`
    (2.2 GB INT4), `genai_config.json`, tokenizer. Built with
    `exclude_embeds=True, exclude_lm_head=True` (decoder takes `inputs_embeds`
    → `hidden_states`; this also avoids the onnx_ir serialization crash that the
    151936×2560 embedding/lm_head tensors trigger on a 4B model).
  - `inference.py` — drives the decoder + tied embedding for a text demo.
  - `eval.py` — parity + greedy-prefix agreement vs the original PyTorch Qwen3.
- **Verification (INT4 ONNX vs PyTorch Qwen3):**
  - hidden-state cosine **0.998–0.999**, next-token argmax **3/3**
  - greedy continuation **48/48 tokens identical (100%)**

## Project layout (current)
```
model/                       # downloaded checkpoint (gitignored)
qwen3_standalone/            # extracted Qwen3-4B LLM (top-level; reference + tied embeds)
onnx/{device}_{precision}/   # ALL sub-parts, flat, unique names:
    llm_decoder.onnx (+.data) + genai_config.json + tokenizer*
    audio_embed.onnx
    audio_heads.onnx
    audio_tokenizer.onnx
    manifest.json
```

## Can onnxruntime-genai run the whole model? — NO
og.Generator can drive a *text* decoder defined by `genai_config.json`, but it cannot
run this TTS model end-to-end:
- our `llm_decoder` is built with `exclude_lm_head` → outputs `hidden_states`, not
  logits (og's generate loop needs a logits/lm-head);
- the audio path uses a **fused multi-codebook head** + **delay-pattern sampling** +
  a **neural codec** — none are og-supported ops/architectures.
So og is used only to *build* the decoder; the audio AR loop, delay sampler, and
**codes→waveform are implemented by us** (`inference.py`, `audio_tokenizer.onnx`).

## Framework refactor ✅ (device / precision / layout)
- `optimize.py --device {cpu,cuda} --precision {int4,fp16,fp32}` → outputs to
  **`onnx/{device}_{precision}/`** (per-target folders), writes `manifest.json`.
- `inference.py --model-path onnx/{device}_{precision}` and
  `eval.py --model-path onnx/{device}_{precision}` are **manifest-driven** — a single
  path holds all sub-parts (`llm_decoder/`, `qwen3_standalone/`, audio_* when built).
- Current built target: `onnx/cpu_int4/` (LLM verified: cosine 0.998, 36/36 prefix).

## Build / run / eval
```
python optimize.py --device cpu --precision int4      # → onnx/cpu_int4/
python optimize.py --device cpu --precision int4 --skip-download
python inference.py --model-path onnx/cpu_int4 --prompt "..."
python eval.py      --model-path onnx/cpu_int4
```

## Audio research conclusion (decided)
- `boson-multimodal` is **v2/v2.5 only** — the boson-ai repo states *"Higgs Audio v3 is a
  standalone release and does not depend on the code here."* → not usable for v3.
- Authoritative v3 modeling = **sgl-project/sglang-omni** `sglang_omni/models/higgs_tts/`
  (`modeling.py`, `model.py`, `audio_codec.py`). Its `HiggsAudioCodec` reads the codec
  from the same prefix `tied.embedding.modality_embeddings.0.model.*` we found, and it
  **vendors the v2 tokenizer** (`_vendored/higgs_audio_v2_tokenizer_hf.py`) — confirming
  our OmniVoice/HF v2 work is the right codec.
- **Decision: reimplement the wrappers from the sglang-omni v3 source** (reference saved
  in `_ref/`), reusing OmniVoice DAC know-how for the codec.

## Audio sub-parts — status
- ✅ **audio_embed** (`HiggsFusedMultiTextEmbedding`): codes `[B,L,8]` → `[B,L,2560]`
  (`codes + arange(8)*1026`, `F.embedding(...).sum(-2)`). Olive INT4.
  Verified vs PyTorch: cos **0.998**, max|d| 0.06.
- ✅ **audio_heads** (`HiggsFusedMultiTextHead`, tied to audio_embed): `[B,L,2560]` →
  `[B,L,8,1026]`. Olive INT4. Verified: cos **0.998**, argmax-agree 0.87.
  Delay pattern is applied in the generation loop (sglang `sampler.py`), not in this
  ONNX module.
- ⏳ **audio_tokenizer (waveform codec / code2wav)** — last remaining part. Weights at
  `tied.embedding.modality_embeddings.0.model.*` (DAC `acoustic_decoder` + `semantic_model`).
  Plan: load into the vendored/HF `higgs_audio_v2_tokenizer`, export with Olive reusing
  OmniVoice wrappers (`OmniVoice/codes/model_wrappers.py`, `OmniVoice/higgs/*.json`);
  expect DAC pre-tracing + static-shape handling as in OmniVoice/CSM.

## ✅ Self-contained fix — text_embed.onnx (no original model at runtime)
The LLM is built with `exclude_embeds`+`exclude_lm_head`, so the embedding lookup must
live outside the decoder. Previously that meant inference depended on
`qwen3_standalone/model.safetensors` (the original 9 GB checkpoint extract) — defeating the
point of ONNX. **Now the embedding is a first-class ONNX sub-part:**
- `user_script.py`: `TextEmbed` wrapper (`input_ids → inputs_embeds`, a Gather) +
  `get_text_embed_{model,io_config,dummy_inputs}` (weight = `tied.embedding.text_embedding.weight`).
- `optimize.py`: `text_embed` added to `ALL_COMPONENTS`/build loop/manifest; forced **fp16**
  at int4 (RTN int4 only quantizes MatMul, not a Gather → fp16 halves it to ~778 MB).
- `inference.py`: `embed_ids()` calls `text_embed.onnx`; falls back to the numpy table only
  if a standalone is present (eval-only `logits_last`). Standalone no longer required for TTS.
- Quick rebuild into a specific dir: `python _build_text_embed.py <model_dir>` (direct
  torch.onnx, no Olive); or `python optimize.py --components text_embed`.
Verified: `text_embed.onnx` runs under onnxruntime alone (`[batch,seq]→[batch,seq,2560]` fp16),
producing inputs_embeds with **no torch / original model / standalone**.

## ✅ Voice cloning (reference audio + transcript) — supported & wired
The model has the clone tokens (`<|ref_audio|>`=151679, `<|ref_text|>`=151680). Cloning needs
the codec **encoder** (waveform→codes), which the original export lacked (only the decoder was
built). Added:
- `user_script.py`: `CodecEncoderWrapper` (`input_values[B,1,T] → audio_codes[B,8,frames]`,
  `codec.encode`) + `get_audio_encoder_{model,io_config,dummy_inputs}` (fp32, dynamic time).
- `optimize.py`: `audio_encoder` added to `ALL_COMPONENTS`/`AUDIO_FUNCS`/build loop/manifest
  (forced fp32 like the decoder).  Build: `python optimize.py --components audio_encoder`.
- `inference.py`: `--ref-audio` + `--ref-text` → `generate_clone()`. Clone prompt (sglang ref):
  `<|tts|> <|ref_text|> tok(ref_text) <|ref_audio|> [ref-code embeds] <|text|> tok(text) <|audio|>`.
  Reference audio → `audio_encoder` codes → delay pattern → `audio_embed` fused embeds, spliced
  at `<|ref_audio|>`. Shares the AR loop with zero-shot (`_run_ar`).
Verified end-to-end: ref hello.wav → 153 code frames → clone prefill 182 → natural EOC at 3.36 s
(80640 samples). NOTE: clone *fidelity* (does it match the ref voice) still needs a human listen;
the mechanics/prompt format are confirmed against the sglang reference.

Encoder length behavior (export TracerWarnings investigated): `audio_encoder.onnx` folded the
acoustic/semantic length-alignment branch (`modeling:548`), so it only runs when both encoder
streams come out equal length — which holds iff the input is a whole number of codec frames
(hop = SR/fps = **960 samples**). Arbitrary lengths fail with a Concat mismatch (e.g. 1112 vs
1111). **Fix:** `encode_audio` pads the reference up to the next multiple of 960 with silence.
Verified bit-exact vs PyTorch `codec.encode` on the padded input across 30000/50000/33333/17003
(all → match 1.000). The few padded frames are trailing silence — harmless for cloning. (A fully
length-robust encoder would need a dynamo re-export that keeps the alignment branch; not needed
given the pad.)

  Usage: `python inference.py --model-path onnx/cpu_int4 --ref-audio ref.wav \
            --ref-text "transcript of ref.wav" --text "New text" --out  clone.wav`

## Built target: onnx/cpu_int4/  (flat, unique names, self-contained)
  text_embed.onnx · audio_embed.onnx · audio_heads.onnx · audio_tokenizer.onnx ·
  audio_encoder.onnx · llm_decoder.onnx (+.data) + genai_config.json + tokenizer* · manifest.json
  NO qwen3_standalone needed at runtime (only eval.py uses it for the PyTorch reference).
  manifest no longer carries `standalone_dir`.

## TTS prompt format (critical — was the silent-audio bug)
Zero-shot generation REQUIRES the Higgs TTS prompt, not a chat template:
  `[<|tts|>=151667, <|text|>=151672, *tok(text), <|audio|>=151670]`
The trailing `<|audio|>` token is what makes the model emit audio codes. With a
chat-template prompt the audio head ran on text hidden-states and collapsed to a
constant code → silent waveform. Fixed in `generate_speech._tts_prompt_ids`.
(Voice-clone prompts add `<|ref_audio|>`/`<|ref_text|>` + delayed ref codes — see
sglang text_tokenizer.py; zero-shot implemented.) Result now: real audio, natural
EOC stop (e.g. "Hello, this is a test." → 3.72 s, rms 0.032, peak 0.25).

## Sampling (fixes "speaks then long buzz")
Pure greedy/argmax degenerates: codebook-0 sticks on one code (e.g. 244) → monotone
buzz, EOC never fires, runs to max_frames. Fixed with per-codebook temperature+top-k
sampling (matches sglang `_sample_independent`) + a repeat guard. Defaults
`--temperature 0.8 --top-k 50`; tune per input. (`--temperature 0` = greedy.)

## inference.py — generates audio ✅
- `python inference.py --model-path onnx/cpu_int4 --selftest --out x.wav`
  → codec decodes codes → **real 24 kHz wav** (verified: 50 frames → 48000 samples).
- `python inference.py --model-path onnx/cpu_int4 --text "..." --out x.wav`
  → full pipeline: text → LLM (KV-cache AR) → audio_heads → **delay-pattern sampler**
  (BOC=1024/EOC=1025) → reverse delay → codec → wav. Mechanics implemented per the
  sglang-omni reference; speech intelligibility depends on the exact voice-design
  prompt format and should be validated against the reference runtime.

## All FIVE sub-parts BUILT (onnx/cpu_int4/) ✅
- llm_decoder.onnx (+2.29 GB data) + genai_config + tokenizer — verified cos 0.9987, 36/36 prefix
- text_embed.onnx (fp16, ~778 MB) — input_ids→inputs_embeds Gather; self-contained, verified
- audio_embed.onnx · audio_heads.onnx — cos 0.998
- audio_tokenizer.onnx — codec, cos 1.0
- Full `inference.py --text` runs end-to-end → wav; `--selftest` codec → wav.

## ✅ Full generation loop validated end-to-end (self-contained)
`inference.py --model-path onnx/cpu_int4 --text "Hello, this is a test." --max-frames 80`
→ loads all 5 sub-parts (incl. text_embed; **no standalone / no torch model**), runs
text→LLM(KV-cache AR)→audio_heads→delay sampler→codec → **49920 samples / 2.08 s @ 24 kHz**.
Signal is speech-like, NOT degenerate buzz: rms 0.067, peak 0.40, rich spectrum
(entropy ≈ 8.5), frame-energy std/mean ≈ 0.76 (temporal dynamics). Prompt format matches
the sglang reference (`[<|tts|>, <|text|>, *tok(text), <|audio|>]`).
REMAINING UNKNOWN (only one left): exact transcript intelligibility can't be auto-verified
without listening or a PyTorch v3 reference runtime (sglang-omni not installed) — needs a
human listen or a reference-codes parity run to fully confirm word accuracy.

## ✅ RESOLVED: cuda_int4 on ARM64/Jetson = prebuilt-ORT runtime bug, NOT an export bug
Reporter (Jetson Orin Nano 8 GB, aarch64, CUDA 13.2, cuDNN 9.20) confirmed end-to-end:
- **Prebuilt `onnxruntime-gpu 1.26.0` wheel**: `llm_decoder` all-zero on CUDA (GQA + MatMulNBits
  both PLACED on CUDA, so not a fallback — the kernel computes zeros) → NaN in sampling. Our
  diagnostic localized it: audio_heads (MatMulNBits) is correct on CUDA, only the decoder's
  **GroupQueryAttention** fails → GQA is the culprit.
- **Source-built ORT 1.28.0** (`CMAKE_CUDA_ARCHITECTURES=87`, `onnxruntime_USE_FLASH_ATTENTION=ON`):
  all-zero is GONE; `llm_decoder` CUDA vs CPU cos ≈ 0.95 (fp16 rounding, not zero), and
  **audio generation works — valid 24 kHz WAV, sounds correct** (German sample). Needed a manual
  workaround for a **CUDA 13.2 CCCL header bug** (`proclaims_copyable_arguments` specialization in
  `cub/device/device_transform.cuh`) unrelated to ORT/model.
**Conclusion: the ONNX export is correct; the failure was the prebuilt aarch64 ORT GQA CUDA kernel.**
Fix for Jetson users: build ORT from source for SM 8.7 (flash attn on), or use the CPU EP.

Two code fixes this surfaced (now in `inference.py` + `diagnose_cuda.py`):
- **Input dtype**: CUDA/fp16 ModelBuilder decoders expect **float16** `inputs_embeds`/KV cache
  (CPU int4 expects float32). Code now reads the expected dtype from the graph and casts
  (was hardcoded float32 → "Unexpected input data type … expected float16" on CUDA). A generic
  `_run()`/`cast_feeds()` casts every float feed to each session's expected dtype.
- **NaN guard** in `_sample` (+ the existing all-zero guard after prefill): a broken contrib
  kernel now raises a clear error instead of a numpy "Probabilities contain NaN" stacktrace.

### (historical) original analysis — cuda_int4 silent-zeros on ARM64/Jetson
Reported on Jetson Orin Nano (ARM64, SM 8.7, CUDA 13, onnxruntime-gpu 1.25–1.27): the
CUDAExecutionProvider registers without error, but `llm_decoder.onnx` outputs **all zeros**
→ silent/garbage audio. CPU EP works (hidden_abs_mean ≈ 1.49).

Local op-inventory analysis (`onnx/cpu_int4/*.onnx`):
- `llm_decoder`: 180× **MatMulNBits**, 36× **GroupQueryAttention**, Skip/SimplifiedLayerNorm.
- `audio_heads`: 1× MatMulNBits (reporter says it works on their CUDA).
- `audio_embed`: 1× GatherBlockQuantized (reporter says it works).
- `text_embed`: plain Gather.
→ Since `audio_heads` ALSO uses MatMulNBits and reportedly computes fine, **GroupQueryAttention
is the stronger suspect** (unique to the decoder), possibly together with MatMulNBits at scale.
Root cause class: onnxruntime CUDA **contrib-op kernels** (GQA flash/mem-efficient attention,
MatMulNBits) are validated mainly on **x86_64 + desktop NVIDIA**. ARM64/Jetson JetPack ORT builds
may register the op but lack a working CUDA kernel for SM 8.7 / CUDA 13 → silent zeros (no error).
**This is a portability limitation of the ModelBuilder INT4 export, not a conversion error** —
validation here was CPU-only; "works on CPU/x86 CUDA" does NOT imply "works on all CUDA targets".

Mitigations:
- **Now:** run the CPU provider (correct, slower) — `inference.py` adds a silent-zeros guard that
  raises a clear error instead of emitting dead audio.
- **fp16 build** drops MatMulNBits (regular fp16 MatMul) but ModelBuilder still emits GQA, so it
  only helps if MatMulNBits (not GQA) is the culprit — needs a Jetson test to confirm.
- **Portable fix (if GQA is the cause):** export the decoder via Olive with eager attention
  (no GroupQueryAttention contrib op), like the Qwen3-TTS talker — heavier but EP-agnostic. TODO.
- **Or** build onnxruntime from source on the Jetson with CUDA contrib kernels for SM 8.7.
- Model-card note recommended so Jetson users aren't surprised.
To pin GQA vs MatMulNBits, ask the reporter for: `onnxruntime.get_build_info()`, an ORT verbose
log (`log_severity_level=0`, shows kernel placement/fallbacks), a rigorous numeric check of
`audio_heads` on CUDA, and an isolated GQA-only vs MatMulNBits-only probe.

## Known issue: LLM int4 build needs RAM headroom
`create_model` int4 serialization (onnx_ir `to_proto`, materializing LazyTensors into
one proto) is memory-heavy for this 4B model. It **succeeded with ~20 GB+ free**
(verified: cosine 0.998, 36/36 prefix) but **fails (SerdeError / segfault) at ~16 GB
free**. Not a code bug. To (re)build the LLM, free RAM (close VS/browser) and run:
  `python optimize.py --device cpu --precision int4 --skip-download --skip-extract --components llm_decoder`
The other 3 sub-parts + codec audio generation work regardless.

## Notes / artifacts on disk (gitignored)
- `model/` (~12 GB) downloaded checkpoint — keep for audio-tokenizer work, or delete.
- `cpu_int4/qwen3_standalone/` (7.6 GB) — needed by `inference.py`/`eval.py`
  (embedding source + PyTorch reference). Keep.

## Expected ONNX shape (do not forget)
- Sub-ONNX parts; **LLM → ModelBuilder INT4**; audio/codec/heads → Olive INT4
  (fp32/fp16 where INT4 is too lossy). Outputs under `cpu_int4/models/` + manifest.
  Three scripts: `optimize.py`, `inference.py`, `eval.py`. Template: OmniVoice.
