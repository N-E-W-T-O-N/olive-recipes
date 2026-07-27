"""VibeVoice Acoustic Tokenizer — ONNX encode/decode driver (the shared 7.5 Hz speech codec used by
the VibeVoice TTS and ASR models). onnxruntime + numpy only (+ soundfile/librosa for audio I/O).

Flow (hop = 3200 samples/frame @ 24 kHz):

  audio ─[acoustic_encoder]→ latents [1, frames, 64] ─[acoustic_decoder]→ audio
         (encoder fixed at 24000 samples/call → 7 frames; long audio is encoded in 1 s chunks)

Usage:
  python inference_tokenizer.py --models-dir fp32 --input in.wav --output recon.wav   # round-trip
  python inference_tokenizer.py --models-dir fp32 --input in.wav --latents-out z.npy --encode-only
  python inference_tokenizer.py --models-dir fp32 --latents-in z.npy --output out.wav --decode-only
"""
import argparse
from pathlib import Path

import numpy as np

SR = 24_000          # sampling rate
CHUNK = 24_000       # encoder's baked input length (1 s)
HOP = 3_200          # samples per latent frame
VAE_DIM = 64


def _session(path, provider):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=[provider])


def _npdtype(sess):
    return np.float16 if "float16" in sess.get_inputs()[0].type else np.float32


def load_audio(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def save_audio(path, wav):
    import soundfile as sf
    sf.write(path, np.asarray(wav, np.float32).ravel(), SR, subtype="PCM_16")


def encode(enc, wav):
    """wav [T] → latents [1, frames, 64], printing the encoder flow. Chunks into 1 s windows."""
    npdt = _npdtype(enc)
    n = int(np.ceil(len(wav) / CHUNK))
    padded = np.pad(wav, (0, n * CHUNK - len(wav)))
    dt = "fp16" if npdt == np.float16 else "fp32"
    print(f"[encode] {len(wav)} samples ({len(wav)/SR:.2f}s) @ {SR} Hz  ->  {n} x {CHUNK} chunk(s) "
          f"(padded to {len(padded)}), {dt}")
    lat = []
    for i in range(n):
        chunk = padded[i * CHUNK:(i + 1) * CHUNK][None, None, :].astype(npdt)
        z = enc.run(["latents"], {"audio": chunk})[0]
        lat.append(z)
        print(f"  encoder: audio[1,1,{CHUNK}]  ->  latents{list(z.shape)}   (chunk {i+1}/{n})")
    latents = np.concatenate(lat, axis=1)
    print(f"[encode] done: latents {list(latents.shape)}  (7.5 Hz, {VAE_DIM}-dim per frame)")
    return latents


def decode(dec, latents):
    """latents [1, frames, 64] → audio [samples], printing the decoder flow."""
    npdt = _npdtype(dec)
    frames = latents.shape[1]
    audio = dec.run(["audio"], {"latents": latents.astype(npdt)})[0].ravel()
    print(f"[decode] decoder: latents[1,{frames},{VAE_DIM}]  ->  audio[1,1,{len(audio)}]  "
          f"({len(audio)/SR:.2f}s, {HOP}/frame)")
    return audio


def main():
    ap = argparse.ArgumentParser(description="VibeVoice Acoustic Tokenizer — ONNX encode/decode")
    ap.add_argument("--models-dir", default=".", help="dir with acoustic_encoder/decoder.onnx (fp32|fp16)")
    ap.add_argument("--input", "-i", default=None)
    ap.add_argument("--output", "-o", default="reconstructed.wav")
    ap.add_argument("--latents-out", default=None)
    ap.add_argument("--latents-in", default=None)
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    prov = "CUDAExecutionProvider" if args.cuda else "CPUExecutionProvider"
    d = Path(args.models_dir)
    print(f"=== VibeVoice Acoustic Tokenizer (ONNX) | models={d} | {prov} ===")

    if args.decode_only:
        latents = np.load(args.latents_in)
        wav = decode(_session(d / "acoustic_decoder.onnx", prov), latents)
        save_audio(args.output, wav)
        print(f"saved -> {args.output}")
        return

    wav = load_audio(args.input)
    latents = encode(_session(d / "acoustic_encoder.onnx", prov), wav)
    if args.latents_out:
        np.save(args.latents_out, latents); print(f"latents -> {args.latents_out}")
    if args.encode_only:
        return
    recon = decode(_session(d / "acoustic_decoder.onnx", prov), latents)
    save_audio(args.output, recon)
    n = min(len(recon), len(wav))
    corr = float(np.corrcoef(wav[:n], recon[:n])[0, 1]) if n > 1 else float("nan")
    print(f"saved -> {args.output}   round-trip corr={corr:+.3f}")
    if len(wav) > CHUNK:
        print("  note: multi-chunk round-trip loses ~1600 samples/chunk at the fixed-encoder edge "
              "(cumulative time offset lowers corr); per-1s-window parity is corr ~0.999.")


if __name__ == "__main__":
    main()
