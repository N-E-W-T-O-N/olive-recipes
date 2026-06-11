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
  in `codes/`), reusing OmniVoice DAC know-how for the codec.

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

## Built target: onnx/cpu_int4/  (flat, unique names)
  audio_embed.onnx · audio_heads.onnx · audio_tokenizer.onnx · manifest.json
  llm_decoder.onnx (+.data) + genai_config.json + tokenizer*   ← see RAM note below
  qwen3_standalone/ at project top-level (reference/embeds; gitignored)

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

## All four sub-parts BUILT (onnx/cpu_int4/) ✅
- llm_decoder.onnx (+2.29 GB data) + genai_config + tokenizer — verified cos 0.9987, 36/36 prefix
- audio_embed.onnx · audio_heads.onnx — cos 0.998
- audio_tokenizer.onnx — codec, cos 1.0
- Full `inference.py --text` runs end-to-end → wav; `--selftest` codec → wav.

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
