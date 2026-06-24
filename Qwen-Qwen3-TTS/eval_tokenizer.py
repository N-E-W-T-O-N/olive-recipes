# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3",
#   "torch",
#   "torchvision",
#   "torchaudio",
#   "onnx",
#   "onnxruntime>=1.20",
#   "numpy",
#   "safetensors",
#   "huggingface_hub",
#   "accelerate",
#   "librosa",
#   "soundfile",
# ]
# ///
"""Evaluate the exported Qwen3-TTS tokenizer ONNX vs the original PyTorch.

Reference = the exact `TokEncoderWrapper` / `TokDecoderWrapper` forwards from
user_script.py (what optimize.py exported), loaded in fp32 from the tokenizer
checkpoint. Pins transformers==4.57.3 via PEP-723, like optimize.py.

Checks
------
(a) Encoder parity : audio[1,1,24000] → codes[1,frames,16]; exact index match %
    (ONNX vs PyTorch) + per-codebook agreement.
(b) Decoder parity : codes → waveform; cosine + max|Δ| (ONNX vs PyTorch).
(c) Round-trip     : encode→decode entirely in ONNX; reconstruction vs input
    (and vs the PyTorch round-trip) — cosine / max|Δ|.

Usage:
  uv run eval_tokenizer.py --model-path onnx/cpu_fp16
  uv run eval_tokenizer.py --model-path onnx/cpu_fp16 --tok-path tokenizer --save-wav
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "codes"))

SR = 24000
N = 24000  # encoder is exported with a static 1-second input


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_signal(seed=0):
    """Deterministic 1 s @ 24 kHz mix of sweeps + harmonics (speech-band-ish)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / SR
    sweep = np.sin(2 * np.pi * (120 + 400 * t) * t)
    harm = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    env = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, 1)          # syllable-like envelope
    sig = (sweep + harm) * (0.5 + 0.5 * env) + 0.01 * rng.standard_normal(N)
    sig = sig / (np.abs(sig).max() + 1e-6) * 0.95
    return sig.astype(np.float32)[None, None, :]               # [1, 1, N]


def main():
    ap = argparse.ArgumentParser(description="Qwen3-TTS tokenizer ONNX vs PyTorch")
    ap.add_argument("--model-path", required=True, help="onnx/{device}_{precision} dir")
    ap.add_argument("--tok-path", default="tokenizer", help="PyTorch tokenizer checkpoint dir")
    ap.add_argument("--save-wav", action="store_true", help="dump input + reconstructions")
    args = ap.parse_args()

    import torch
    import onnxruntime as ort
    from user_script import get_tok_encoder_model, get_tok_decoder_model

    mdir = Path(args.model_path)
    manifest = json.loads((mdir / "manifest.json").read_text())
    enc_onnx = mdir / manifest["sub_models"]["tok_encoder"]["filename"]
    dec_onnx = mdir / manifest["sub_models"]["tok_decoder"]["filename"]
    print(f"ONNX dir   : {mdir}")
    print(f"  encoder  : {enc_onnx.name}")
    print(f"  decoder  : {dec_onnx.name}")
    print(f"PyTorch ref: {args.tok_path}\n")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    enc_sess = ort.InferenceSession(str(enc_onnx), so, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(dec_onnx), so, providers=["CPUExecutionProvider"])

    dec_shape = dec_sess.get_inputs()[0].shape          # [batch, frames, 16]
    dec_frames = dec_shape[1] if isinstance(dec_shape[1], int) else None
    print(f"  decoder input shape: {dec_shape}"
          + ("  (frames FIXED — see note)" if dec_frames else "  (frames dynamic)"))

    def fit_frames(codes, F):
        """Tile/trim codes [1,T,16] to F frames so a fixed-length decoder accepts them."""
        if F is None or codes.shape[1] == F:
            return codes
        T = codes.shape[1]
        idx = np.arange(F) % T
        return codes[:, idx, :]

    print("Loading PyTorch reference wrappers (fp32) ...")
    enc_pt = get_tok_encoder_model(args.tok_path)
    dec_pt = get_tok_decoder_model(args.tok_path)

    audio = test_signal()

    # ── (a) Encoder parity ──────────────────────────────────────────────────
    print("\n=== (a) Encoder parity  audio[1,1,24000] → codes[1,T,16] ===")
    codes_onnx = enc_sess.run(None, {"audio": audio})[0]
    with torch.no_grad():
        codes_pt = enc_pt(torch.from_numpy(audio)).cpu().numpy()
    codes_onnx = np.asarray(codes_onnx).astype(np.int64)
    codes_pt = codes_pt.astype(np.int64)
    print(f"  shapes   onnx={codes_onnx.shape}  pytorch={codes_pt.shape}")
    if codes_onnx.shape == codes_pt.shape:
        match = float((codes_onnx == codes_pt).mean())
        print(f"  exact index match : {match:.4%}")
        per_cb = (codes_onnx == codes_pt).mean(axis=(0, 1))   # [16]
        worst = int(np.argmin(per_cb))
        print(f"  per-codebook match: min={per_cb.min():.3f} (cb{worst})  "
              f"mean={per_cb.mean():.3f}  max={per_cb.max():.3f}")
        enc_ok = match > 0.99
    else:
        print("  SHAPE MISMATCH — cannot compare indices")
        enc_ok = False

    # ── (b) Decoder parity ──────────────────────────────────────────────────
    print("\n=== (b) Decoder parity  codes → waveform ===")
    codes_in = fit_frames(codes_pt, dec_frames)  # same codes into both decoders
    if dec_frames and codes_in.shape[1] != codes_pt.shape[1]:
        print(f"  (decoder is fixed at {dec_frames} frames; tiled {codes_pt.shape[1]}→{dec_frames})")
    wav_onnx = dec_sess.run(None, {"audio_codes": codes_in})[0]
    with torch.no_grad():
        wav_pt = dec_pt(torch.from_numpy(codes_in)).cpu().numpy()
    n = min(wav_onnx.shape[-1], wav_pt.shape[-1])
    c = cosine(wav_onnx[..., :n], wav_pt[..., :n])
    d = float(np.abs(wav_onnx[..., :n] - wav_pt[..., :n]).max())
    print(f"  shapes   onnx={wav_onnx.shape}  pytorch={wav_pt.shape}")
    print(f"  cosine={c:.5f}  max|Δ|={d:.4e}")
    dec_ok = c > 0.999

    # ── (c) Round-trip (ONNX encode→decode) ─────────────────────────────────
    print("\n=== (c) Round-trip  ONNX encode→decode vs input / PyTorch ===")
    codes_rt = fit_frames(codes_onnx, dec_frames)
    wav_rt_onnx = dec_sess.run(None, {"audio_codes": codes_rt})[0]
    with torch.no_grad():
        wav_rt_pt = dec_pt(torch.from_numpy(codes_rt)).cpu().numpy()
    m = min(wav_rt_onnx.shape[-1], audio.shape[-1])
    print(f"  recon vs input    : cosine={cosine(wav_rt_onnx[..., :m], audio[..., :m]):.4f}")
    k = min(wav_rt_onnx.shape[-1], wav_rt_pt.shape[-1])
    print(f"  ONNX vs PyTorch RT: cosine={cosine(wav_rt_onnx[..., :k], wav_rt_pt[..., :k]):.5f}  "
          f"max|Δ|={float(np.abs(wav_rt_onnx[..., :k] - wav_rt_pt[..., :k]).max()):.4e}")

    if args.save_wav:
        import soundfile as sf
        sf.write(mdir / "eval_input.wav", audio[0, 0], SR)
        sf.write(mdir / "eval_recon_onnx.wav", wav_rt_onnx.reshape(-1), SR)
        sf.write(mdir / "eval_recon_pytorch.wav", wav_rt_pt.reshape(-1), SR)
        print(f"\n  wrote eval_input.wav / eval_recon_onnx.wav / eval_recon_pytorch.wav → {mdir}")

    print("\n=== verdict ===")
    print(f"  encoder index parity : {'PASS' if enc_ok else 'CHECK'}")
    print(f"  decoder waveform     : {'PASS' if dec_ok else 'CHECK'}")
    sys.exit(0 if (enc_ok and dec_ok) else 1)


if __name__ == "__main__":
    main()
