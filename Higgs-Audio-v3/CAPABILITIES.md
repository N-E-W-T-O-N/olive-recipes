# Higgs-Audio-v3 — Audio, Voice & Expression Capabilities

Capabilities of `bosonai/higgs-audio-v3-tts-4b` and how they map onto this ONNX
pipeline. Source: the HF model card + the sglang-omni reference. Control tokens are
plain text tokens, so they flow through our zero-shot prompt
(`<|tts|> <|text|> tok(text) <|audio|>`) unchanged — no pipeline change needed to use them.

## Audio format
| Property | Value |
|---|---|
| Output | 24 kHz mono waveform (`audio_tokenizer.onnx`) |
| Frame rate | 25 fps (40 ms/frame); samples ≈ frames × 960 |
| Codec | Higgs v2 tokenizer, **8 codebooks × 1026 vocab**, delay pattern (BOC=1024, EOC=1025) |
| Backbone | Qwen3-4B, interleaved text+audio tokens |

## Languages
- **102 languages** total: **85 at production quality** (WER/CER < 5%) and 17 at usable
  quality (5–10%). Multilingual text is tokenized the same way; no flag needed.

## Voice / speaker
- **Zero-shot voice cloning** from a reference clip + reference transcript. ✅ IMPLEMENTED:
  `audio_encoder.onnx` (waveform→codes) + delay pattern + `audio_embed` splicing →
  `<|tts|> <|ref_text|> tok(ref) <|ref_audio|> [ref-code embeds] <|text|> tok(text) <|audio|>`.
  Use `--ref-audio ref.wav --ref-text "<exact transcript>"`. (Ref audio is padded to a
  multiple of 960 samples internally; clone *fidelity* needs a listen.)
- No fixed/predefined speaker slots — voice identity comes from the reference clip.

## Expression & emotion — inline control tokens
Syntax: **`<|category:value|>`** inserted in the text. Verified: these are real single
special tokens in the tokenizer (not split into subwords), so they're embedded and fed to
the LLM exactly as the original model expects. Example (NOTE the category is `sfx`, not
`sound_effect`): `Hello <|emotion:amusement|> that's funny <|sfx:laughter|>.`

- **Emotions (21):** elation, amusement, enthusiasm, determination, pride, contentment,
  affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger,
  fear, disgust, bitterness, sadness, shame, helplessness
- **Styles (3):** singing, shouting, whispering
- **Sound effects (9), category `sfx`:** `<|sfx:cough|>`, laughter, crying, screaming,
  burping, humming, sigh, sniff, sneeze (each pairs with matching onomatopoeia)
- **Environment (1):** `<|env:music|>`
- **Prosody:**
  - Speed: very_slow (~0.65×), slow (~0.85×), fast (~1.2×), very_fast (~1.4×)
  - Pitch: pitch_low (−3 semitones), pitch_high (+2.5 semitones)
  - Pauses: pause (~400–700 ms), long_pause (~700–1500 ms)
  - Delivery: expressive_high, expressive_low

## Using control tokens with this pipeline
They're regular text — just include them in `--text`:
```
uv run inference.py --model-path onnx/cpu_int4 \
  --text "I can't believe it <|emotion:surprise|> <|sfx:laughter|> amazing!" \
  --temperature 0.8 --top-k 50 --out expressive.wav
```
Use the EXACT token names (`emotion:`, `style:`, `sfx:`, `prosody:`, `env:music`). A
mistyped tag (e.g. `sound_effect:`) is encoded as ordinary subwords (harmless, just no
effect). Sampling (`--temperature`/`--top-k`) is recommended — pure greedy degenerates.

## Streaming
- **Audio-OUTPUT streaming** ✅ `stream_inference.py` — generates from a fully-known text
  prompt (standard `<|tts|>`) but decodes + emits audio in rolling chunks (left-context
  windows for clean seams), writing the wav incrementally and reporting time-to-first-chunk.
  Works for zero-shot and voice-clone. Verified end-to-end.
  `uv run stream_inference.py --model-path onnx/cpu_int4 --text "..." --chunk-frames 50 --out s.wav`
- The model's **`<|streaming_tts|>` interleaved-text-INPUT mode is NOT implemented** — the
  authoritative sglang-omni `higgs_tts` reference implements TTS only and documents NO prompt
  format for `<|streaming_tts|>` / `<|audio_cont_txt|>` / `<|await_audio|>`, so we don't
  fabricate one. (Output-streaming above covers the practical "start playback early" need.)

## ASR (speech→text) — NOT available, and no reference exists
- `<|asr|>` (151665) / `<|streaming_asr|>` (151666) tokens exist, but the sglang-omni
  reference does **not** implement ASR and documents no prompt format for it. Implementing it
  would be reverse-engineering (risking wrong output), AND it needs a text-logit readout
  (the tied **lm_head**, `hidden @ text_embedᵀ`) which the self-contained export doesn't ship
  as a matmul — only `text_embed` as a Gather. Wiring ASR would require both an `lm_head`/logits
  ONNX and a verified ASR prompt format. Not done.

## Supported summary
Zero-shot TTS, voice clone (ref audio+text), 102-language text, all inline
expression/emotion/style/sfx/prosody/env tokens, and audio-output streaming — all supported.
Not exposed: ASR, and the model's interleaved streaming-input modes.
