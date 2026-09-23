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
  uv run inference_asr_streaming.py --mic onnx/asr-streaming/cpu_int4       # live microphone, Ctrl+C to stop
  uv run inference_asr_streaming.py --list-devices                          # list INPUT-capable devices only

Live mic note: prefer a FAST target (cpu_int4 / cuda_int4 / cuda_fp16), not cpu_fp32 — token
generation must keep up with audio arriving in real time (chunk_duration seconds of speech
every chunk_duration seconds), and a 7B fp32-on-CPU decode is far slower than that. int4 barely
loses quality here since ASR only needs clean text generation through lm_head, not fidelity of
raw hidden states (unlike the TTS checkpoints — see HANDOFF trap #5).
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

    Loads `Qwen2TokenizerFast` DIRECTLY rather than via `AutoTokenizer` — the latter resolves
    `AutoConfig.from_pretrained(src)` internally, which instantiates the full transformers-native
    `VibeVoiceConfig` from the checkpoint's composite config.json. That class (as of transformers
    >=5.15ish) validates `diffusion_head_config.hidden_size == text_config.hidden_size`; this
    checkpoint uses the codes-family `decoder_config` key (not `text_config`), so `text_config`
    silently defaults to a generic Qwen2 config (hidden_size=4096) that never matches the real,
    vestigial `diffusion_head_config` (hidden_size=3584, unused — no diffusion weights exist in
    this checkpoint, see HANDOFF trap #25) — crashing tokenizer loading with a config validation
    error that has nothing to do with the tokenizer itself. `tokenizer_config.json` declares
    `tokenizer_class: Qwen2Tokenizer` (standard across the VibeVoice family), so loading that
    class directly skips AutoConfig/model-config resolution entirely.
    """
    from transformers import Qwen2TokenizerFast
    if not (Path(src) / "tokenizer_config.json").exists():
        sys.exit(f"[error] {src} has no tokenizer_config.json — not a valid ASR-Streaming checkpoint dir")
    tok = Qwen2TokenizerFast.from_pretrained(str(src))
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


class ChunkDecoder:
    """Owns the persistent LLM/KV-cache session and per-chunk decode loop — the part shared
    identically between file mode (chunks sliced from a loaded wav) and --mic mode (chunks sliced
    from a live rolling audio buffer). Construct once per session; call decode(seg) per chunk."""

    def __init__(self, key, src, onnx_dir, device, tok, hotwords, max_new_tokens, temperature,
                repetition_penalty):
        self.tok = tok
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.speech_start_id = tok.convert_tokens_to_ids(SPEECH_START_TOKEN)
        self.speech_end_id = tok.convert_tokens_to_ids(SPEECH_END_TOKEN)
        self.text_chunk_end_id = tok.convert_tokens_to_ids(TEXT_CHUNK_END_TOKEN)
        self.eos_id = tok.eos_token_id

        self.ops = {name: C.OnnxOp(onnx_dir / f"{name}.onnx", device)
                    for name in ("acoustic_encoder", "semantic_encoder", "acoustic_connector", "semantic_connector")}
        self.llm = C.OnnxLLM(onnx_dir / "llm_decoder.onnx", device)

        def embed(ids):
            return C.embed_tokens(src, ids, key, onnx_dir=onnx_dir)
        self.embed = embed

        prompt_ids = tok.encode(build_prompt(hotwords), add_special_tokens=False)
        self.llm.prefill(embed(prompt_ids))
        self.speech_start_embed = embed([self.speech_start_id])
        self.speech_end_embed = embed([self.speech_end_id])
        self.tce_embed = embed([self.text_chunk_end_id])

    def decode(self, seg):
        """One audio chunk (1-D float32 samples, already windowed to chunk+lookahead) -> (text, n_tokens)."""
        feats = encode_audio_features(self.ops, seg)                       # [n,H]
        audio_embeds = np.concatenate([self.speech_start_embed, feats[None], self.speech_end_embed], axis=1)
        hidden = self.llm.step(audio_embeds)
        logits = hidden[:, -1:, :]

        chunk_tokens = []
        for _ in range(self.max_new_tokens):
            last = np.asarray(logits)[0, -1].copy()
            if self.repetition_penalty != 1.0 and chunk_tokens:
                prev = np.asarray(chunk_tokens)
                pv = last[prev]
                last[prev] = np.where(pv > 0, pv / self.repetition_penalty, pv * self.repetition_penalty)
            if self.temperature <= 0:
                nxt = int(last.argmax())
            else:
                p = np.exp((last - last.max()) / self.temperature)
                p /= p.sum()
                nxt = int(np.random.choice(len(p), p=p))
            if nxt == self.text_chunk_end_id or nxt == self.eos_id:
                break
            chunk_tokens.append(nxt)
            logits = self.llm.step(self.embed([nxt]))

        self.llm.step(self.tce_embed)   # chunk-boundary marker, always injected (matches upstream)

        chunk_text = self.tok.decode(chunk_tokens, skip_special_tokens=True)
        for st in STRIP_TOKENS:
            chunk_text = chunk_text.replace(st, "")
        return chunk_text, len(chunk_tokens)


def level_dbfs(seg):
    """RMS level of a chunk in dBFS (0 = full-scale). -inf shown as -99.0 for digital silence."""
    rms = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 1e-6 else -99.0


def level_bar(dbfs, width=20, floor=-60.0):
    """A [####........] style meter — dbfs clamped to [floor, 0] and scaled to `width` cells."""
    frac = max(0.0, min(1.0, (dbfs - floor) / -floor))
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def print_chunk_status(idx, total, seg_sr_pair, text, n_tokens, elapsed):
    """One verbose status line + the transcribed text, for both file and mic modes."""
    seg, sr = seg_sr_pair
    audio_s = len(seg) / sr
    rtf = elapsed / audio_s if audio_s > 0 else float("inf")
    dbfs = level_dbfs(seg)
    tag = f"{idx}/{total}" if total else str(idx)
    speed = f"{1/rtf:.1f}x realtime" if rtf > 0 else "n/a"
    pace = "OK" if rtf <= 1.0 else "FALLING BEHIND"
    print(f"  [{tag}] audio={audio_s:.2f}s decode={elapsed:.2f}s (RTF={rtf:.2f}, {speed}, {pace})  "
          f"tokens={n_tokens}  level={dbfs:6.1f}dBFS {level_bar(dbfs)}")
    shown = text.strip()
    print(f"        \"{shown}\"" if shown else "        (silence / no speech detected)", flush=True)


def list_devices():
    """Print only INPUT-capable ('listening') devices — sd.query_devices() also lists every
    output-only device (speakers, HDMI, SPDIF...), which just adds noise when picking a mic."""
    import sounddevice as sd
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    print("Input (listening) devices:")
    found = False
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) <= 0:
            continue
        found = True
        mark = ">" if idx == default_in else " "
        print(f" {mark} {idx:3d}  {d['name']}  (host={sd.query_hostapis(d['hostapi'])['name']}, "
              f"in_ch={d['max_input_channels']}, default_sr={d['default_samplerate']:.0f}Hz)")
    if not found:
        print("  (none found)")
    print(f"\nUse --mic-device <index> to pick one. '>' marks the system default input.")


def build_prompt(context_info=None):
    keys_str = "speaker, content"
    if context_info:
        return ("You are a helpful assistant that transcribes audio input into text output. "
                f"Please transcribe the following audios streamingly with these keys: {keys_str} "
                f"and extra info: {context_info}\n")
    return ("You are a helpful assistant that transcribes audio input into text output. "
            f"Please transcribe the following audios streamingly with these keys: {keys_str}\n")


def run_file(decoder, wav, chunk_samples, lookahead_samples, sr, pad_last_chunk):
    """Offline mode: slide a fixed window over an already-loaded wav array."""
    import time
    window = chunk_samples + lookahead_samples
    total_samples = len(wav)
    chunk_bounds = []
    t = 0
    while t < total_samples:
        end = min(t + window, total_samples)
        chunk_bounds.append((t, end))
        t = min(t + chunk_samples, total_samples)
    total_chunks = len(chunk_bounds)

    print(f"  audio {total_samples/sr:.2f}s -> {total_chunks} chunks, decoding…")
    print("  " + "-" * 68)
    transcript_parts = []
    session_start = time.perf_counter()
    for idx, (a, b) in enumerate(chunk_bounds, start=1):
        seg = wav[a:b]
        if pad_last_chunk and len(seg) < window:
            seg = np.pad(seg, (0, window - len(seg)))
        t0 = time.perf_counter()
        chunk_text, n_tokens = decoder.decode(seg)
        elapsed = time.perf_counter() - t0
        transcript_parts.append(chunk_text)
        print_chunk_status(idx, total_chunks, (seg, sr), chunk_text, n_tokens, elapsed)
    total_elapsed = time.perf_counter() - session_start
    audio_s = total_samples / sr
    print("  " + "-" * 68)
    print(f"  done: {total_chunks} chunks, {audio_s:.2f}s audio in {total_elapsed:.2f}s "
          f"(overall RTF={total_elapsed/audio_s:.2f})")
    print(f"\nTRANSCRIPT:\n{''.join(transcript_parts)}")


def run_mic(decoder, chunk_samples, lookahead_samples, sr, device_arg, block_ms):
    """Live mode: capture from the microphone and decode chunks as they fill, until Ctrl+C.

    Uses a rolling buffer: once >= window samples are available, decode the first `window` of
    them, then drop the leading `chunk_samples` (NOT the whole window) — the trailing
    `lookahead_samples` are kept and re-used as the start of the next window, exactly mirroring
    how run_file() re-slices an overlapping window from a fully-loaded array. The front-end
    encoders are stateless per call (see the module docstring), so re-encoding the overlap from
    scratch each chunk is correct, not wasted work.
    """
    import queue
    import time
    import sounddevice as sd

    window = chunk_samples + lookahead_samples
    block_frames = max(1, int(sr * block_ms / 1000))
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  [mic][warn] {status}", file=sys.stderr)
        q.put(indata[:, 0].copy())

    dev_desc = device_arg
    try:
        info = sd.query_devices(device_arg if device_arg is not None else sd.default.device[0])
        dev_desc = f"{device_arg!r} = '{info['name']}'"
    except Exception:
        pass
    print(f"  listening on device={dev_desc} (block={block_ms}ms, chunk_window={window/sr:.2f}s) "
          f"— Ctrl+C to stop")
    print("  " + "-" * 68)
    transcript_parts = []
    buf = np.zeros(0, dtype=np.float32)
    idx = 0
    session_start = time.perf_counter()
    total_audio_s = 0.0
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block_frames,
                            device=device_arg, callback=callback):
            while True:
                while len(buf) < window:
                    buf = np.concatenate([buf, q.get()])
                seg = buf[:window]
                t0 = time.perf_counter()
                chunk_text, n_tokens = decoder.decode(seg)
                elapsed = time.perf_counter() - t0
                buf = buf[chunk_samples:]
                idx += 1
                total_audio_s += chunk_samples / sr
                print_chunk_status(idx, 0, (seg, sr), chunk_text, n_tokens, elapsed)
                if chunk_text.strip():
                    transcript_parts.append(chunk_text)
    except KeyboardInterrupt:
        pass
    total_elapsed = time.perf_counter() - session_start
    print("  " + "-" * 68)
    print(f"  stopped: {idx} chunks, ~{total_audio_s:.2f}s audio over {total_elapsed:.2f}s session")
    print(f"\nTRANSCRIPT:\n{''.join(transcript_parts)}")


def main():
    ap = argparse.ArgumentParser(description="VibeVoice ASR-Streaming (ONNX, chunked)")
    ap.add_argument("model_path", nargs="?", help="built ONNX dir, e.g. onnx/asr-streaming/cpu_int4")
    src_group = ap.add_mutually_exclusive_group()
    src_group.add_argument("--audio", help="input wav (offline mode)")
    src_group.add_argument("--mic", action="store_true", help="live microphone (Ctrl+C to stop)")
    ap.add_argument("--mic-device", default=None,
                    help="input device index or name substring (default: system default input). "
                         "See --list-devices.")
    ap.add_argument("--mic-block-ms", type=int, default=100,
                    help="microphone capture block size in ms (default 100)")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    ap.add_argument("--hotwords", default=None, help="comma-separated hotwords/context, e.g. 'Microsoft,VibeVoice'")
    ap.add_argument("--max-new-tokens-per-chunk", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-pad-last-chunk", action="store_true",
                    help="don't zero-pad a short final chunk to full chunk+lookahead length (file mode only)")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return
    if not args.model_path:
        ap.error("model_path is required (unless --list-devices)")
    if not args.mic and not args.audio:
        ap.error("pass --audio FILE (offline) or --mic (live)")

    key, src, onnx_dir, device, precision, model_id = C.resolve_from_path(args.model_path)
    print(f"=== ASR-Streaming | model_id={model_id}  device={device}  precision={precision} ===")
    if key != "asr-streaming":
        sys.exit(f"[error] {model_id} ({key}) is not the ASR-Streaming model — use inference_asr.py")
    for need in ("acoustic_encoder", "semantic_encoder", "acoustic_connector", "semantic_connector"):
        if not (onnx_dir / f"{need}.onnx").exists():
            sys.exit(f"missing {need}.onnx in {onnx_dir} — build: uv run optimize.py {key}")
    if not (onnx_dir / "llm_decoder.onnx").exists():
        sys.exit(f"missing llm_decoder.onnx in {onnx_dir} (7B is RAM-bound — see HANDOFF.md trap #6). "
                 f"Build it: uv run optimize.py --components llm {key}")
    mic_device = args.mic_device
    if mic_device is not None and mic_device.lstrip("-").isdigit():
        mic_device = int(mic_device)   # --list-devices prints numeric indices; sounddevice wants
        args.mic_device = mic_device   # an int for those, not the digit string (which it'd treat
                                        # as a name substring to match, not an index)

    if args.mic and precision == "fp32" and device == "cpu":
        print("  [warn] cpu_fp32 decode is likely too slow to keep up with live audio — "
              "prefer cpu_int4 / cuda_int4 / cuda_fp16 for --mic (see module docstring).")

    tok = load_tokenizer_strict(src)
    chunk_samples, lookahead_samples, sr = load_frame_config(src)
    print(f"  chunk={chunk_samples/sr:.3f}s  lookahead={lookahead_samples/sr:.3f}s")

    decoder = ChunkDecoder(key, src, onnx_dir, device, tok, args.hotwords,
                           args.max_new_tokens_per_chunk, args.temperature, args.repetition_penalty)

    if args.mic:
        run_mic(decoder, chunk_samples, lookahead_samples, sr, args.mic_device, args.mic_block_ms)
    else:
        wav = C.load_audio(args.audio, sr)
        run_file(decoder, wav, chunk_samples, lookahead_samples, sr, not args.no_pad_last_chunk)


if __name__ == "__main__":
    main()
