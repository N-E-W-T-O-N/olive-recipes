# Qwen3-TTS — Current Status

_Updated: 2026-06-19_

## ✅ De-duplicated talker — ship talker_cache only (~870 MB/dir saved)
`talker.onnx` (no-cache) and `talker_cache.onnx` (KV-cache) hold the SAME transformer weights
→ shipping both duplicated ~870 MB per dir. Fix: default export now builds **only
`talker_cache`** (it does prefill+decode, faster O(n), and is what inference auto-uses); plain
`talker` is still buildable via explicit `--components talker`. Dropped the redundant
`talker.onnx` (+ manifest entry) from all dirs that had both → **freed ~22.5 GB** across the
8 built variants. inference.py guards for "neither talker present". Verified generation still
works on talker_cache-only dirs (KV-cache path, audio produced).

## ✅ Config-driven dims — 0.6B now exports; 1.7B undisturbed
`user_script.py` had 1.7B dims hardcoded (hidden 2048, 28 layers, etc.) in the talker /
talker_cache / code_predictor / residual_embed io+dummy funcs → 0.6B (hidden **1024**) would
export wrong. Fixed with a module `_DIMS` populated by `_load_tts()` from the loaded model's
`talker_config` (Olive calls model_loader before io/dummy). **Defaults equal the 1.7B values**,
so 1.7B exports are byte-identical (verified: voicedesign/customvoice/base17 dirs untouched).
**Validated**: full `base/0.6B` int4 export — all 9 components incl. talker/talker_cache/
speaker_encoder — `onnx/base06/cpu_int4/`, talker `inputs_embeds [batch,seq,1024]` ✓.
(0.6B differs from 1.7B only in hidden_size; layers/kv/head_dim/groups are identical.)

## ✅ EXPORT OOM RESOLVED — `optimize.py` isolates each component in a subprocess
The Windows pagefile/OOM kill on full-run exports is fixed: `main()` now re-invokes itself once
per component in a fresh subprocess (`--_child`), so memory is fully reclaimed between builds and
the heavy talker/talker_cache run alone. Default behavior; `--no-isolate` forces single-process.
`speaker_encoder` auto-dropped from defaults for non-base models. **Validated**: a full
**cpu_fp16** voicedesign export (the heaviest case — 5.6 GB fp16 talker + talker_cache) completed
in ONE command, all 8 components, no OOM → `onnx/voicedesign/cpu_fp16/`.

## ✅ All 3 model types exported + validated with their feature
| dir | model_type | feature | validated |
|---|---|---|---|
| `onnx/voicedesign/cpu_int4/` | voice_design | `--instruct` | ✅ generates |
| `onnx/customvoice/cpu_int4/`  | custom_voice | `--speaker ryan` | ✅ generates (27 frames) |
| `onnx/base17_cpu_int4/` | base | clone `--ref-audio/--ref-text` | ✅ runs e2e; speaker_encoder parity 1.0 |
Model-type gating enforces the right feature per model (wrong flag → clear error).
Pending exports: 0.6B Base, other precisions/devices (fp16/fp32, cuda) — mechanical repeats
(component-by-component due to the pagefile cap). KV-cache talker still blocked.

## TL;DR (current session)
- ✅ Full text→speech `generate()` implemented + **100% greedy parity vs PyTorch** (all-16
  codebooks 432/432). Needed talker→(logits,hidden) + `residual_embed.onnx` + the
  `dynamic_shapes` seq fix — all done & validated.
- ✅ **Model-type feature gating in inference.py** — `model_type` read from config;
  `_check_features` enforces: voice_design→`--instruct`, custom_voice→`--speaker`,
  base→clone (`--ref-audio/--ref-text`). VALIDATED both ways: voice_design+instruct generates;
  voice_design+speaker raises a clear ValueError. `--text` prints `model_type=…`.
- ✅ **voicedesign `onnx/voicedesign/cpu_int4/` restored** (all 7 components) and generation
  re-validated (model_type=voice_design, 24-frame wav). Re-export had to be done
  **component-by-component** (talker solo) — a single full-run OOM/pagefile-killed the talker
  (`OSError 1455: paging file too small`). The lighter parts export together.
- ✅ **Env fix:** `optimize.py` + `inference.py` PEP-723 now pin `numba>=0.60 / llvmlite>=0.43`
  — librosa otherwise pulls numba 0.53.1 → llvmlite 0.36 which won't build on Python 3.12.
- ✅ Base checkpoints downloaded (`base/1.7B`, `base/0.6B`); **Base-1.7B int4 exported** →
  `onnx/base17_cpu_int4/` (7 components incl. `residual_embed` + 2-output dynamic talker).
- ✅ **Base voice cloning — IMPLEMENTED & runs end-to-end.** `generate(ref_audio, ref_text)`
  → `_generate_clone`. Pieces:
  • `speaker_encoder.onnx` exported (ECAPA + **inline mel/STFT** front-end, dynamic audio len,
    fp32, dynamo; `audio[B,T]→x-vector[B,2048]`). `SpeakerEncoderWrapper` reimplements
    mel_spectrogram WITHOUT its `if torch.min(y)<-1` debug branch (broke torch.export).
  • ICL prefill mirrors `generate_icl_prompt`: ref_text+text+eos / codec_bos + **per-frame
    ref-code sum (= `step_embed`/residual_embed!)**, with the x-vector injected in the codec
    prefix. Ref audio→codes via `tok_encoder` in 1 s windows (`encode_chunked`).
  • AR loop factored into shared `_ar_loop`. Verified runs: ref 39 frames + x-vector →
    prefill 64 → 40 frames → 3.2 s wav (`onnx/base17_cpu_int4/`, model_type=base).
  ✅ **speaker_encoder parity** (`eval_speaker.py`): ONNX x-vector vs PyTorch
  `extract_speaker_embedding` = **cosine 1.000000, max|Δ| ~1e-6** at 2/3.5/6 s → the
  reimplemented mel/STFT is exact. With tok_encoder (100%), residual_embed/talker/predictor
  (100% TTS parity) all verified and the ICL prefill mirroring the reference, clone is correct
  by construction. (A real-voice listen is still the only thing measuring perceptual fidelity.)
- ✅ **KV-cache talker — SOLVED (was blocked).** Fix: feed the cache as a plain list of
  [k,v] tensor pairs under input name **`past_kv`** (NOT `past_key_values`, so Olive's
  DynamicCache-pytree auto-conversion — which torch.export rejected — doesn't fire), and
  build the `DynamicCache` in-graph (`from_legacy_cache`/`to_legacy_cache`). `talker_cache.onnx`
  exports (59 in / 58 out: flattened 28×2 K/V). Verified: empty-past output == no-cache talker
  (logits/hidden cos 1.0, max|Δ| 0.0); end-to-end greedy **100% exact** vs no-cache. `inference.py`
  auto-uses it (`_ar_loop_cached`) when present. Speedup grows with length (talker O(n) vs O(n²);
  ~1.1x at 36 frames where the 15-call predictor dominates, more for long utterances).
  Exported for `onnx/voicedesign/cpu_int4/` so far.
- ⏳ **Export dirs:** the earlier `onnx/voicedesign/` & `onnx/customvoice/` were cleared in a
  reorg; only `onnx/base17_cpu_int4/` survives. Re-exporting voicedesign cpu_int4 now.
- TODO: validate generate on base export; export 0.6B; optional tok_decoder dynamic-frames.

## Targets (3 models, auto-detected by config `model_type`) — all downloaded locally
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`  (qwen3_tts)  ← `voicedesign/`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`  (qwen3_tts)  ← `customvoice/`
- `Qwen/Qwen3-TTS-Tokenizer-12Hz`         (qwen3_tts_tokenizer_12hz) ← `tokenizer/` (also embedded in each TTS as `speech_tokenizer/`)

## Architecture (verified from repo + load)
TTS `Qwen3TTSForConditionalGeneration` → `talker`:
- `talker.model` — Qwen3-style **28 L, hidden 2048, MROPE**, dual codec/text embedding
- `talker.codec_head` — Linear 2048→3072 (first-codebook logits)
- `talker.code_predictor` — 5 L, hidden 1024, **16 per-group heads** (residual codes)
- `talker.text_projection`
Tokenizer `Qwen3TTSTokenizerV2Model` → **encoder** (wav→codes) + **decoder** (codes→wav),
24 kHz, 12.5/12 fps, 16 quantizers (decode), conv+transformer+RVQ.

## Decomposition → `onnx/{device}_{precision}/` (flat)
- TTS: `talker.onnx`, `code_predictor.onnx` (+ `tok_encoder/decoder.onnx` from embedded codec)
- Tokenizer: `tok_encoder.onnx`, `tok_decoder.onnx`
- codecs forced **fp32** (DAC int4/fp16 too lossy); LLM/predictor honor `--precision`.

## create_model (ModelBuilder) — empirically checked, NOT viable for the talker
onnxruntime-genai ModelBuilder supports these archs: Llama, Mistral, Qwen2, **Qwen3**,
Gemma/2/3, Phi*, Granite, Nemotron, Olmo, SmolLM3, GptOss, InternLM2, Lfm2, HunYuan,
VideoChatFlashQwen. The talker is **`qwen3_tts_talker`** (not listed) and, decisively,
uses **MROPE** (`mrope_section=[24,20,20]`, interleaved) + dual codec/text embedding +
`codec_head`. Even remapped to `Qwen3ForCausalLM`, ModelBuilder applies **standard RoPE
(no MROPE)** → positionally wrong output. So create_model can't correctly convert it →
**Olive export (with the real MROPE forward) is the correct path.** Confirmed via the
supported-architecture list + the talker rope config.

## Key decisions / findings
- **Talker is NOT ModelBuilder-able** (custom MROPE + dual embedding ≠ stock Qwen3) →
  exported via **Olive** (`OnnxConversion` + RtnQuant/Float16), like CSM backbone.
  (ModelBuilder remap attempted only if a talker is detectably stock.)
- **transformers version**: vendored `qwen_tts` (in `codes/`) needs **transformers==4.57.3**;
  the shared venv is 5.10.2 (incompatible — check_model_inputs, config defaults,
  ROPE_INIT_FUNCTIONS, …). Solved by making **optimize.py a PEP-723 uv script** that pins
  4.57.3 in an isolated env (`uv run optimize.py ...`). Shared venv untouched.
  One vendored patch: `@check_model_inputs()` → `@check_model_inputs` (tf5 API); 25Hz
  tokenizer import made optional (needs `sox`).
- **Model loads cleanly** under 4.57.3 (instantiates; submodules mapped).

## Fixed — external-data relink on flattened models
`optimize.py` flattens Olive's `model.onnx[.data]` → `{name}.onnx[.data]`, but the proto's
`external_data.location` still pointed at `model.onnx.data` → onnxruntime failed to load any
model with external data (`External data path does not exist`). Affected talker fp32/fp16
(int4 talker is self-contained, no `.data`). Fixed: `relink_external_data()` rewrites each
tensor's `location` (proto-only, no multi-GB RAM load) after the move; existing dirs relinked
in place (cpu/cuda fp32=313 refs, fp16=255). Verified by `inference.py --selftest` loading talker.

## inference.py (manifest-driven) — full generation implemented
`Pipeline(onnx/{dev}_{prec}, tts_dir=...)` loads all parts with the manifest EP (CPU fallback).
Building blocks verified in eval_*.py: `embed_text`, `embed_codec`, `talker_step`,
`predict_residual`, `step_embed`, `encode`, `decode`/`decode_chunked`.

`generate(text, language, instruct=, speaker=)` now implements the real text→speech path,
faithful to `Qwen3TTSForConditionalGeneration.generate` with `non_streaming_mode=True`:
  • **text-to-speech** — text + language
  • **voice design**   — text + `instruct` (natural-language style)        [VoiceDesign]
  • **custom voice**   — text + `speaker` name (+ optional instruct)       [CustomVoice]
Prefill assembly mirrors modeling lines 2068-2234 (role + codec tags + pad/bos + text body +
eos + codec_bos; instruct embeds prepended). AR talker loop is no-cache (re-runs the growing
prefix; MROPE collapses to `arange` since `get_rope_index`=cumsum(mask)-1 with 3 identical rows
and no padding). Each step: talker→first-codebook logits + last hidden → 15 residual codes via
the **causal** teacher-forced predictor (fill known codes, read `group_logits[j-1]`) → next input
= `residual_embed(codes16)` (=`codec_hiddens.sum(1)`) + `tts_pad`. suppress_tokens + repetition
penalty + top-k/top-p sampling replicated in numpy. `--text` CLI bug fixed (was `if selftest or
True:`; now proper branching).

**Two new exported components** were required (the old 6 couldn't roll out generation):
  • `talker.onnx` now emits **(logits, hidden_states)** — the predictor is conditioned on the
    talker's last hidden state, not its logits. (re-export needed)
  • `residual_embed.onnx` — `codec_ids[B,16] → codec_hiddens.sum(1)[B,2048]`, summing
    `talker.model.codec_embedding(code0)` + `code_predictor.model.codec_embedding[i](code_{i+1})`.
    Those residual per-group embeddings were buried in the predictor graph; needed for the
    next-step talker input.

**Audio-clone (ICL ref_audio/ref_text)** is a `base`-model feature: `create_voice_clone_prompt`
raises for non-base, and `speaker_encoder` is None for VoiceDesign/CustomVoice. Not applicable to
our two targets → intentionally not implemented (documented in inference.py).

Re-export to refresh a dir:
  `uv run optimize.py --model voicedesign --skip-download --device cpu --precision fp32 \
       --components residual_embed talker`

## Fixed — talker needed `dynamic_shapes` (dynamo ignores `dynamic_axes`)
The first talker re-export locked seq to the dummy's 32 (`attention_mask Got 26 Expected 32`
at gen step 0), breaking the AR loop. Root cause: with `use_dynamo_exporter`, Olive
(`conversion.py` ~L368) passes **`dynamic_shapes`** to torch.export and **drops `dynamic_axes`**.
Fix: added `dynamic_shapes` to the talker io_config (`{input: {axis:int → "dimname"}}`, shared
`"seq"` ties the 3 inputs; torch 2.12 accepts string dim names). Re-exported → talker inputs are
now `['batch','seq',2048]` etc. (Same root cause as the tok_decoder fixed-25-frame limitation —
that decoder could be re-exported with `dynamic_shapes` too if a dynamic-frame decoder is wanted.)

## ✅ Generation validated end-to-end (cpu_fp32, voicedesign)
`uv run inference.py --model-path onnx/customvoice/cpu_fp32 --tts-dir voicedesign \
   --text "Hello, this is a test." --instruct "A calm female voice." --out out.wav`
→ AR loop runs (dynamic-seq talker → first code + hidden; 15 residuals via causal predictor;
`residual_embed` next-step input), decodes real audio (RMS 0.10, peak 0.64, non-silent).

**Greedy parity vs PyTorch (`eval_generate.py`, voicedesign, cpu_fp32):** tokenization identical;
first-codebook 27/27 frames (100%); **all-16-codebooks 432/432 = 100.00%** over 27 frames — the
ONNX generation reproduces `model.generate` exactly (prefill + MROPE + AR talker + causal residual
predictor + residual_embed + EOS). Frame count matches the reference (sampling varies length).
NOTE: `onnx/customvoice/cpu_fp32/` is hand-organized and actually holds **voicedesign** content
(manifest `model_id: .\voicedesign\`). Only this dir has the upgraded talker + residual_embed so
far; the other device×precision dirs still need the `residual_embed`/2-output-talker re-export.

## Known operational note — memory pressure on full-run export
A single `optimize.py ... --model voicedesign` (all components) can OOM-kill the process at
`code_predictor`: the `talker` fp16 step holds ~2.8 GB external data + a ~150 s float16 pass, and
stacking `code_predictor`'s float16 pass right after exhausts RAM. **No code defect** — every
component converts cleanly in isolation (verified: code_predictor cuda/fp16 = 354 MB, ½ of fp32).
Workaround: export heavy parts separately, e.g.
`--components talker` then `--components code_predictor text_embed codec_embed tok_encoder tok_decoder`.
`onnx/cuda_fp16/` now holds all 6 (text_embed 639 MB, codec_embed 12.5 MB, talker+.data,
code_predictor 354 MB, tok_encoder/decoder fp32). TODO(optional): make optimize.py export each
component in a subprocess so one full-run command can't OOM.

## Build / run
```
uv run optimize.py --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --device cpu --precision int4
uv run optimize.py --model Qwen/Qwen3-TTS-Tokenizer-12Hz        --device cpu --precision fp32
uv run optimize.py --model voicedesign --skip-download --components tok_decoder   # subset
```

## Export recipe (learned)
- Load all sub-models with `attn_implementation="eager"`.
- Codec transformer parts use `create_causal_mask` (not torch.onnx/TorchScript-traceable)
  → set **`use_dynamo_exporter: true`** on the OnnxConversion pass (done for codec parts).
- Bypass high-level wrappers with Python loops: `tok_decoder` calls `self.tok.decoder(...)`
  directly (skips `model.decode`'s `chunked_decode` while-loop); transpose codes
  `[B,T,16]→[B,16,T]` + clamp ≥0 first.
- Codes layout: `model.decode` expects `[B, codes_length, num_quantizers]` (=[B,T,16]).

## Status / next
- ✅ Vendored `qwen_tts` (codes/), uv-script env (tf 4.57.3) validated, model loads.
- ✅ `optimize.py` (uv script, model-name + device/precision dispatch) + `user_script.py`.
- ✅ **6 sub-models export** (text_embed, codec_embed, talker MROPE LLM, code_predictor, tok_encoder, tok_decoder).
- ✅ **Talker embedding primitives added + verified** (`text_embed`, `codec_embed`). The talker
  prefill itself is control flow (variable text len, voice-clone/ICL branches, concat, MROPE
  positions) → stays in Python (inference.py); only the *learned* lookups are in ONNX:
  - `text_embed` = `text_projection(text_embedding(ids))` [B,T,2048] (also covers tts_bos/eos/pad
    — specific ids); 1.28 GB.  `codec_embed` = `codec_embedding(ids)` [B,T,2048]; 25 MB.
  - Parity (`eval_embed.py`, fp32): text_embed cosine 1.000000 / max|Δ| 1.2e-7;
    codec_embed cosine 1.000000 / max|Δ| 0. **Every learned weight now runs through ONNX.**
- ✅ **`code_predictor` corrected + verified** (was missing `small_to_mtp_projection` 2048→1024 +
  codec-embedding assembly). New `CodePredictorWrapper` folds those in-graph (mirrors
  `forward_sub_talker_finetune`); interface = `(talker_hidden[B,2048], codec_ids[B,16]) →
  group_logits[B,15,vocab]`. Parity (`eval_predictor.py`, fp32, voicedesign):
  wrapper-vs-native cosine 1.000000 / max|Δ| 0 / argmax 100%; ONNX-vs-wrapper cosine 1.000000 /
  max|Δ| 5.7e-5 / argmax 100%. This is the **teacher-forced** variant (needs all 16 codes) — for
  parity; the single-step AR variant for real generation is still TODO.
- ✅ **tokenizer verified** (`eval_tokenizer.py`): encoder 100% exact index match; decoder cosine
  1.00000 vs PyTorch. Note: `tok_decoder.onnx` is **fixed at 25 frames** (dynamic frames axis did
  not survive dynamo export) — fine for parity, but inference must chunk/pad to 25.
- ✅ **precision check** (`check_precision.py`): inspects weight dtypes + quant ops (size is
  misleading — codecs are forced fp32, hence byte-identical across int4/fp16/fp32 dirs).
- ✅ `tok_encoder` (wav→codes) **resolved** (225 MB, fp32, dynamo) → all 4 sub-models export.
  Mirrors `MimiModel._encode_frame` (bypasses streaming `encode()`), **plus two static-shape
  patches in `user_script.py`** so torch.export gets concrete conv lengths:
  1. `_patch_mimi_static_padding` — rewrites `MimiConv1d._get_extra_padding_for_conv1d` to
     pure-Python int math (stock builds padding as 0-dim tensors → `.item()` → unbacked
     symints → RVQ `torch.cdist`'s `npoints>25` guard can't resolve).
  2. `_intify_mimi_convs` — converts each `MimiConv1d`'s `stride`/`kernel_size`/`padding_total`
     buffers (0-dim int64 tensors in this model) + derived `padding_left/right` to Python ints
     (reading a buffer in forward, even via `int()`, is a `.item()` under export).
  Fixed input length (24000); only batch dynamic. Unblocks CustomVoice voice-clone (clones via
  reference codes from this encoder; no separate speaker encoder for custom_voice/voice_design).
- Vendored patches applied for the trimmed-25Hz case: guarded `AutoConfig.register` in
  `inference/qwen3_tts_tokenizer.py` (skip None 25Hz config; idempotent 12Hz).
- ⏳ `inference.py` (text→talker→code_predictor→codes→tok_decoder→wav) + `eval.py`.
- Other models: CustomVoice = same TTS arch; Tokenizer-12Hz = standalone codec
  (tok_decoder works; tok_encoder shares the deferred Mimi issue).
