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
- **Zero-shot voice cloning** from a reference clip (optionally + reference transcript
  for higher fidelity). Reference path uses the codec **encoder** + delay pattern:
  `<|tts|> [<|ref_text|> tok(ref)] <|ref_audio|> [-100]×N <|text|> tok(text) <|audio|>`.
  - In this repo: zero-shot is implemented; voice-clone needs the codec *encoder*
    sub-part + ref-code splicing (see STATUS "deferred"). The codec decoder is exported;
    the encoder is the remaining piece for cloning.
- No fixed/predefined speaker slots — voice identity comes from the reference clip.

## Expression & emotion — inline control tokens
Syntax: **`<|category:value|>`** inserted in the text. Examples:
`Hello <|emotion:amusement|> that's funny <|sound_effect:laughter|>.`

- **Emotions (21):** elation, amusement, enthusiasm, determination, pride, contentment,
  affection, relief, contemplation, confusion, surprise, awe, longing, arousal, anger,
  fear, disgust, bitterness, sadness, shame, helplessness
- **Styles (3):** singing, shouting, whispering
- **Sound effects (9):** cough, laughter, crying, screaming, burping, humming, sigh,
  sniff, sneeze (each pairs with matching onomatopoeia)
- **Prosody:**
  - Speed: very_slow (~0.65×), slow (~0.85×), fast (~1.2×), very_fast (~1.4×)
  - Pitch: pitch_low (−3 semitones), pitch_high (+2.5 semitones)
  - Pauses: pause (~400–700 ms), long_pause (~700–1500 ms)
  - Delivery: expressive_high, expressive_low

## Using control tokens with this pipeline
They're regular text — just include them in `--text`:
```
uv run inference.py --model-path onnx/cpu_int4 \
  --text "I can't believe it <|emotion:surprise|> <|sound_effect:laughter|> amazing!" \
  --temperature 0.8 --top-k 50 --out expressive.wav
```
If a tag isn't in the tokenizer's added vocab it is encoded as ordinary subwords
(harmless, just ignored as an effect). Sampling (`--temperature`/`--top-k`) is
recommended — pure greedy degenerates (see STATUS).
