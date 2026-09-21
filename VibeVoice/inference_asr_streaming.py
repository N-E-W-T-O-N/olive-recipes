"""VibeVoice ASR-Streaming inference (audio -> text, chunked) over exported ONNX sub-parts.

Protocol reverse-engineered from upstream `vibevoice/modular/modeling_vibevoice_asr.py`
(`VibeVoiceASRForConditionalGeneration.streaming_generate`, GitHub commit 1541f59 — this
checkpoint postdates the vendored `vibevoice/` source pinned at 303b283, so its streaming
class isn't in our vendored tree; we drive the ONNX sub-parts directly instead):

  - NO chat template, unlike `asr`/`asr-hf` (trap #16(c)). Plain instruction string, encoded with
    `tokenizer.encode(text, add_special_tokens=False)`.
  - Speech markers REUSE the Qwen-VL grounding tokens, same trick as `asr`/`asr-hf` (trap #16/#18):
    speech_start = <|object_ref_start|>, speech_end = <|object_ref_end|>. The checkpoint's own
    NEW-looking tokens <|AUDIO|>/<|audio_bos|>/<|audio_eos|> exist in added_tokens.json but the
    ASR tokenizer class (`VibeVoiceASRTextTokenizerFast._add_vibevoice_special_tokens`) does NOT
    use them for markers — verified against upstream source. Do not "fix" this to the new-looking
    tokens; that would be wrong. There is no pad-token/slot-counting step here (unlike asr/asr-hf):
    audio features are concatenated directly as embeddings between the start/end marker embeddings.
  - <|text_chunk_end|> (id 151665, genuinely new — NOT in stock Qwen2.5 vocab) marks a chunk
    boundary. It only exists in THIS checkpoint's own tokenizer files, so unlike every other
    VibeVoice checkpoint (trap #10: "ships no tokenizer, fetch stock Qwen2.5") we load the
    tokenizer from `src` (the checkpoint dir itself, which does ship one here) rather than
    whatever ModelBuilder copied next to the ONNX — see `load_tokenizer_strict` below.
  - Audio is windowed into (chunk_frames + lookahead_frames) frames from preprocessor_config.json
    (default 22 + 4 => 2.933s chunk, 0.533s lookahead), advanced by chunk_frames*hop samples each
    step (the lookahead re-overlaps into the next window and is re-encoded from scratch — the
    front-end tokenizers are stateless per call here, matching the exported ONNX encoders).
  - ONE persistent KV cache for the whole session (prompt + every chunk + every generated token).
    Per chunk: feed [speech_start, fused_features..., speech_end] as one step, then decode tokens
    one at a time (greedy / temperature) until <|text_chunk_end|> or EOS, THEN unconditionally feed
    the <|text_chunk_end|> embedding once more (chunk-boundary marker in the cache — upstream does
    this even when the model already emitted it as the stop token) before the next chunk.

Usage:
  uv run inference_asr_streaming.py --audio speech.wav onnx/asr-streaming/cpu_fp16
  uv run inference_asr_streaming.py --audio speech.wav --hotwords "Microsoft,VibeVoice" onnx/asr-streaming/cuda_fp16
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C

SPEECH_START_TOKEN = "<|object_ref_start|>"
SPEECH_END_TOKEN = "<|object_ref_end|>"
TEXT_CHUNK_END_TOKEN = "<|text_chunk_end|>"

# Upstream strips these 7 literal special-token strings from each decoded chunk (belt-and-suspenders
# on top of skip_special_tokens=True) — includes both this checkpoint's real markers (object_ref_*)
# and the alias spellings (<|speech_*|>) some other VibeVoice tokenizer variants use for the same
# concept, in case a future checkpoint or tokenizer swap reintroduces them.
STRIP_TOKENS = (TEXT_CHUNK_END_TOKEN, "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
                "<|speech_start|>", "<|speech_end|>", "<|speech_pad|>")


def load_tokenizer_strict(src):
    """Load the tokenizer from the CHECKPOINT dir (not the ONNX dir / fetched-stock fallback).

    Every other VibeVoice checkpoint ships no tokenizer (trap #10), so `common.load_tokenizer`
    prefers whatever generic Qwen2.5 tokenizer ModelBuilder copied next to the ONNX. This
    checkpoint is different: <|text_chunk_end|> is genuinely new (id 151665, one past the last
    id in stock Qwen2.5's vocab) and only exists in the tokenizer files shipped in `src`. Loading
    the wrong one silently returns unk for it and no chunk would ever end.
    """
    from transformers import AutoTokenizer
    if not (Path(src) / "tokenizer_config.json").exists():
        sys.exit(f"[error] {src} has no tokenizer_config.json — not a valid ASR-Streaming checkpoint dir")
    tok = AutoTokenizer.from_pretrained(str(src))
    for name in (SPEECH_START_TOKEN, SPEECH_END_TOKEN, TEXT_CHUNK_END_TOKEN):
        tid = tok.convert_tokens_to_ids(name)
        if tid is None or tid == tok.unk_token_id:
            sys.exit(f"[error] {name} not found in {src}'s tokenizer — this checkpoint is not a "
                     f"streaming ASR model, or the tokenizer files are incomplete.")
    return tok


def load_frame_config(src):
    """chunk_frames / lookahead_frames -> (chunk_samples, lookahead_samples, hop)."""
    pc = json.loads((Path(src) / "preprocessor_config.json").read_text(encoding="utf-8"))
    sr = pc.get("target_sample_rate", C.SR)
    hop = pc.get("speech_tok_compress_ratio", 3200)
    chunk_frames = pc["chunk_frames"]
    lookahead_frames = pc["lookahead_frames"]
    return chunk_frames * hop, lookahead_frames * hop, sr


def encode_audio_features(ops, segment):
    """segment: 1-D float32 samples -> fused [n, H] speech features (sum of acoustic+semantic).
    `ops`: dict of pre-built C.OnnxOp sessions (built once outside the chunk loop — one per chunk
    would otherwise reopen 4 onnxruntime sessions every ~3 seconds of audio)."""
    x = segment[None, None, :].astype(np.float32)             # [1,1,T]
    al = ops["acoustic_encoder"].run(audio=x)                  # [1,fa,64]
    sl = ops["semantic_encoder"].run(audio=x)                  # [1,fs,128]
    n = min(al.shape[1], sl.shape[1])
    al, sl = al[:, :n].astype(np.float32), sl[:, :n].astype(np.float32)
    ac = ops["acoustic_connector"].run(features=al)
    sc = ops["semantic_connector"].run(features=sl)
    fused = ac + sc
    return fused[0].astype(np.float32)                         # [n,H]


def build_prompt(context_info=None):
    keys_str = "speaker, content"
    if context_info:
        return ("You are a helpful assistant that transcribes audio input into text output. "
                f"Please transcribe the following audios streamingly with these keys: {keys_str} "
                f"and extra info: {context_info}\n")
    return ("You are a helpful assistant that transcribes audio input into text output. "
            f"Please transcribe the following audios streamingly with these keys: {keys_str}\n")


def main():
    ap = argparse.ArgumentParser(description="VibeVoice ASR-Streaming (ONNX, chunked)")
    ap.add_argument("model_path", help="built ONNX dir, e.g. onnx/asr-streaming/cpu_fp16")
    ap.add_argument("--audio", required=True, help="input wav")
    ap.add_argument("--hotwords", default=None, help="comma-separated hotwords/context, e.g. 'Microsoft,VibeVoice'")
    ap.add_argument("--max-new-tokens-per-chunk", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-pad-last-chunk", action="store_true",
                    help="don't zero-pad a short final chunk to full chunk+lookahead length")
    args = ap.parse_args()

    key, src, onnx_dir, device, precision, model_id = C.resolve_from_path(args.model_path)
    print(f"=== ASR-Streaming | model_id={model_id}  device={device}  precision={precision} ===")
    if key != "asr-streaming":
        sys.exit(f"[error] {model_id} ({key}) is not the ASR-Streaming model — use inference_asr.py")
    for need in ("acoustic_encoder", "semantic_encoder", "acoustic_connector", "semantic_connector"):
        if not (onnx_dir / f"{need}.onnx").exists():
            sys.exit(f"missing {need}.onnx in {onnx_dir} — build: uv run optimize.py {key}")
    llm_path = onnx_dir / "llm_decoder.onnx"
    if not llm_path.exists():
        sys.exit(f"missing llm_decoder.onnx in {onnx_dir} (7B is RAM-bound — see HANDOFF.md trap #6). "
                 f"Build it: uv run optimize.py --components llm {key}")

    tok = load_tokenizer_strict(src)
    speech_start_id = tok.convert_tokens_to_ids(SPEECH_START_TOKEN)
    speech_end_id = tok.convert_tokens_to_ids(SPEECH_END_TOKEN)
    text_chunk_end_id = tok.convert_tokens_to_ids(TEXT_CHUNK_END_TOKEN)
    eos_id = tok.eos_token_id

    chunk_samples, lookahead_samples, sr = load_frame_config(src)
    print(f"  chunk={chunk_samples/sr:.3f}s  lookahead={lookahead_samples/sr:.3f}s")

    wav = C.load_audio(args.audio, sr)
    total_samples = len(wav)
    duration = total_samples / sr

    llm = C.OnnxLLM(llm_path, device)

    def embed(ids):
        return C.embed_tokens(src, ids, key, onnx_dir=onnx_dir)

    prompt_ids = tok.encode(build_prompt(args.hotwords), add_special_tokens=False)
    llm.prefill(embed(prompt_ids))
    speech_start_embed = embed([speech_start_id])
    speech_end_embed = embed([speech_end_id])
    tce_embed = embed([text_chunk_end_id])

    window = chunk_samples + lookahead_samples
    chunk_bounds = []
    t = 0
    while t < total_samples:
        end = min(t + window, total_samples)
        chunk_bounds.append((t, end))
        t = min(t + chunk_samples, total_samples)
    total_chunks = len(chunk_bounds)

    ops = {name: C.OnnxOp(onnx_dir / f"{name}.onnx", device)
           for name in ("acoustic_encoder", "semantic_encoder", "acoustic_connector", "semantic_connector")}

    print(f"  audio {duration:.2f}s -> {total_chunks} chunks, decoding…")
    transcript_parts = []
    for idx, (a, b) in enumerate(chunk_bounds):
        seg = wav[a:b]
        if not args.no_pad_last_chunk and len(seg) < window:
            seg = np.pad(seg, (0, window - len(seg)))
        feats = encode_audio_features(ops, seg)                        # [n,H]
        audio_embeds = np.concatenate([speech_start_embed, feats[None], speech_end_embed], axis=1)
        hidden = llm.step(audio_embeds)
        logits = hidden[:, -1:, :]

        chunk_tokens = []
        for _ in range(args.max_new_tokens_per_chunk):
            last = np.asarray(logits)[0, -1].copy()
            if args.repetition_penalty != 1.0 and chunk_tokens:
                prev = np.asarray(chunk_tokens)
                pv = last[prev]
                last[prev] = np.where(pv > 0, pv / args.repetition_penalty, pv * args.repetition_penalty)
            if args.temperature <= 0:
                nxt = int(last.argmax())
            else:
                p = np.exp((last - last.max()) / args.temperature)
                p /= p.sum()
                nxt = int(np.random.choice(len(p), p=p))
            if nxt == text_chunk_end_id or nxt == eos_id:
                break
            chunk_tokens.append(nxt)
            logits = llm.step(embed([nxt]))

        llm.step(tce_embed)   # chunk-boundary marker, always injected (matches upstream)

        chunk_text = tok.decode(chunk_tokens, skip_special_tokens=True)
        for st in STRIP_TOKENS:
            chunk_text = chunk_text.replace(st, "")
        transcript_parts.append(chunk_text)
        print(f"  [{idx + 1}/{total_chunks}] {chunk_text}", flush=True)

    print(f"\nTRANSCRIPT:\n{''.join(transcript_parts)}")


if __name__ == "__main__":
    main()
