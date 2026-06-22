# /// script
# requires-python = ">=3.10"
# dependencies = ["onnxruntime", "numpy", "soundfile", "transformers"]
# # librosa is imported lazily in inference._load_wav only when a ref wav needs resampling
# ///
"""Streaming (low-latency) audio OUTPUT for Higgs-Audio-v3 ONNX.

Streams the AUDIO as it is generated: decode + emit in rolling chunks so playback / writing
can start after the first chunk (~1 s) instead of waiting for the whole utterance. Reuses the
exact same model + prompt as inference.py (zero-shot or voice-clone) — only the *delivery* is
incremental. Works for both `--text` and `--text --ref-audio/--ref-text`.

IMPORTANT — what "streaming" means here:
  This streams audio OUTPUT from a fully-known text prompt (standard `<|tts|>` path). It is NOT
  the model's `<|streaming_tts|>` interleaved-text-INPUT mode: the authoritative sglang-omni
  reference implements TTS only and documents NO prompt format for `<|streaming_tts|>` /
  `<|streaming_asr|>` / `<|audio_cont_txt|>` / `<|await_audio|>`, so we don't fabricate one.

Each flush decodes a rolling window of [ctx_frames left-context + new frames] through the codec
and emits only the new samples — the left context keeps the conv-decoder boundaries clean, so
chunk seams match the one-shot decode.

Usage:
  uv run stream_inference.py --model-path onnx/cpu_int4 --text "..." --out out.wav
  uv run stream_inference.py --model-path onnx/cpu_int4 --text "..." \
      --ref-audio ref.wav --ref-text "transcript" --chunk-frames 50 --out clone.wav
"""
import argparse
import sys
import time

import numpy as np

from inference import (Pipeline, reverse_delay_pattern, N_CODEBOOKS, SR, BOC_ID, EOC_ID)

HOP = 960          # samples per codec frame (SR / 25 fps)


class StreamPipeline(Pipeline):
    def generate_stream(self, text, ref_audio=None, ref_text=None, chunk_frames=50,
                        ctx_frames=8, max_frames=2000, temperature=0.8, top_k=50,
                        top_p=1.0, seed=0, max_repeat=32):
        """Generator: yields mono float32 waveform chunks as they are produced."""
        prefill = self.build_prefill(text, ref_audio, ref_text)
        rng = np.random.default_rng(seed)
        past = self._empty_past()
        hidden, past = self._llm_step(prefill, prefill.shape[1], past)
        if float(np.abs(hidden).mean()) < 1e-8:
            raise RuntimeError(
                f"llm_decoder produced all-zero hidden states on provider '{self.provider}' "
                "— INT4 contrib kernels not computing (known on prebuilt ARM64/Jetson CUDA). "
                "Use CPU EP or a source-built ORT. See STATUS.md.")
        total = prefill.shape[1]

        delayed = []; delay_count = 0; eoc_countdown = None
        last_cb0, repeat = None, 0; stop = "cap"
        emitted_T = 0          # how many post-delay frames already decoded+emitted

        def flush(final=False):
            """Decode the newly-finished post-delay frames with left context; emit new samples."""
            nonlocal emitted_T
            if len(delayed) < N_CODEBOOKS:
                return
            codes_TN = np.clip(reverse_delay_pattern(np.stack(delayed)), 0, 1023)  # [T,8]
            T = codes_TN.shape[0]
            if T <= emitted_T and not final:
                return
            a = max(0, emitted_T - ctx_frames)                 # left-context start
            wav = self.decode_codes(codes_TN[a:T])             # decode window
            drop = (emitted_T - a) * HOP                       # samples belonging to context
            emitted_T = T
            new = wav[drop:]
            if new.size:
                yield_chunk.append(new)

        yield_chunk = []
        t0 = time.time()
        for _ in range(max_frames):
            logits = self._run(self.audio_heads, {"hidden_states": hidden[:, -1:, :]})[0]
            codes = self._sample(logits[0, 0], temperature, top_k, top_p, rng)
            if codes[0] == last_cb0:
                repeat += 1
                if repeat >= max_repeat:
                    stop = "repeat"; break
            else:
                repeat = 0
            last_cb0 = int(codes[0])
            if delay_count < N_CODEBOOKS:
                nxt = delay_count + 1
                if nxt < N_CODEBOOKS:
                    codes[nxt:] = BOC_ID
                delay_count += 1
            elif eoc_countdown is not None:
                eoc_countdown -= 1
                if eoc_countdown <= 0:
                    delayed.append(codes); stop = "eoc"; break
            elif int(codes[0]) == EOC_ID:
                eoc_countdown = N_CODEBOOKS - 2
            delayed.append(codes)
            emb = self._run(self.audio_embed, {"codes": codes[None, None]})[0]
            total += 1
            hidden, past = self._llm_step(emb, total, past)

            if len(delayed) - (emitted_T + N_CODEBOOKS - 1) >= chunk_frames:
                flush()
                for c in yield_chunk:
                    yield c
                yield_chunk.clear()

        flush(final=True)                                      # tail
        for c in yield_chunk:
            yield c
        secs = len(delayed) * HOP / SR
        msg = {"cap": f"hit max-frames ({secs:.1f}s) — likely TRUNCATED",
               "repeat": f"repeat-guard stop at {secs:.1f}s",
               "eoc": f"natural end-of-speech at {secs:.1f}s"}[stop]
        print(f"  [{stop}] {msg}; total wall {time.time()-t0:.1f}s", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Higgs-Audio-v3 streaming (incremental audio output).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--ref-audio", default=None, help="reference wav for voice cloning")
    ap.add_argument("--ref-text", default=None, help="transcript of --ref-audio")
    ap.add_argument("--out", default="stream.wav", help="wav written incrementally as chunks arrive")
    ap.add_argument("--chunk-frames", type=int, default=50, help="frames per emitted chunk (~2s)")
    ap.add_argument("--ctx-frames", type=int, default=8, help="left-context frames for clean seams")
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0, help="nucleus sampling (1.0=off; ~0.95 helps)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import soundfile as sf
    pipe = StreamPipeline(args.model_path)
    print(f"Loaded {args.model_path} (provider={pipe.provider})")

    t0 = time.time()
    first = None
    total_samples = 0
    with sf.SoundFile(args.out, mode="w", samplerate=SR, channels=1, subtype="PCM_16") as f:
        for i, chunk in enumerate(pipe.generate_stream(
                args.text, ref_audio=args.ref_audio, ref_text=args.ref_text,
                chunk_frames=args.chunk_frames, ctx_frames=args.ctx_frames,
                max_frames=args.max_frames, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, seed=args.seed)):
            if first is None:
                first = time.time() - t0
                print(f"  first audio chunk after {first:.2f}s (latency to first sound)")
            f.write(chunk.astype(np.float32))
            total_samples += chunk.shape[0]
            print(f"    chunk {i}: +{chunk.shape[0]/SR:.2f}s  (total {total_samples/SR:.2f}s)",
                  file=sys.stderr)
    print(f"Wrote {args.out}  ({total_samples/SR:.2f}s, first-chunk {first:.2f}s)")


if __name__ == "__main__":
    main()
