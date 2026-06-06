"""
Standalone Higgs Audio V2 Tokenizer — ONNX inference.

Encodes audio to discrete codec codes and decodes codes back to audio,
using the four exported ONNX sub-models:

  acoustic_encoder.onnx   — DAC encoder:   (1,1,T_24k) → (1,256,T_frames)
  semantic_encoder.onnx   — HuBERT encoder: (1,T_16k)  → (1,768,T_frames)
  quantizer_encoder.onnx  — RVQ encode:    (1,256,T)+(1,768,T) → (8,1,T)
  higgs_decoder.onnx      — DAC decode:    (8,1,T) → (1,1,T_24k)

No PyTorch, no transformers required at runtime — only onnxruntime.

Usage:
  # Round-trip encode → decode (sanity check)
  python higgs_inference.py --input speech.wav --output reconstructed.wav

  # Encode only → save codes as .npy
  python higgs_inference.py --input speech.wav --encode-only --codes-out codes.npy

  # Decode only from saved codes
  python higgs_inference.py --decode-only --codes-in codes.npy --output decoded.wav

  # Use a different models directory
  python higgs_inference.py --models-dir path/to/higgs/models --input speech.wav --output out.wav
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (from audio_tokenizer/config.json)
# ---------------------------------------------------------------------------
SR_24K        = 24_000          # acoustic DAC sample rate
SR_16K        = 16_000          # HuBERT semantic sample rate
HOP_LENGTH    = 960             # DAC downsampling factor: product of [8,5,4,2,3]
D_ACOUSTIC    = 256             # acoustic encoder output channels
D_SEMANTIC    = 768             # semantic encoder output channels
N_CODEBOOKS   = 8               # RVQ codebooks
CODEBOOK_SIZE = 1024            # entries per codebook
DEFAULT_MODELS_DIR = Path(__file__).parent / "higgs" / "models"


# =============================================================================
# Audio I/O
# =============================================================================

def load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file, resample to target_sr, mix to mono.

    Returns float32 array in [-1, 1], shape (T,).
    Tries torchaudio first (accurate), falls back to scipy.
    """
    path = str(path)

    # --- torchaudio (preferred) ---
    try:
        import torchaudio, torch
        wav, sr = torchaudio.load(path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav.squeeze(0).numpy().astype(np.float32)
    except ImportError:
        pass

    # --- scipy fallback ---
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data


def save_wav(path: str, waveform: np.ndarray, sr: int = SR_24K):
    """Save float32 numpy waveform as 16-bit PCM WAV."""
    path = str(path)
    try:
        import soundfile as sf
        sf.write(path, waveform, sr, subtype="PCM_16")
        return
    except ImportError:
        pass
    from scipy.io import wavfile
    pcm = np.clip(waveform, -1.0, 1.0)
    wavfile.write(path, sr, (pcm * 32767).astype(np.int16))


# =============================================================================
# ONNX session loader
# =============================================================================

class HiggsOnnxSessions:
    """Holds the four Higgs ONNX InferenceSession objects."""

    MODEL_FILES = {
        "acoustic_encoder":  "acoustic_encoder.onnx",
        "semantic_encoder":  "semantic_encoder.onnx",
        "quantizer_encoder": "quantizer_encoder.onnx",
        "higgs_decoder":     "higgs_decoder.onnx",
    }

    def __init__(self, models_dir: str, provider: str = "CPUExecutionProvider"):
        import onnxruntime as ort
        models_dir = Path(models_dir)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3   # suppress INFO / WARNING noise

        self.sessions = {}
        for key, filename in self.MODEL_FILES.items():
            p = models_dir / filename
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    f"Export Higgs models first:\n"
                    f"  cd OmniVoice && python optimize.py --higgs-only"
                )
            self.sessions[key] = ort.InferenceSession(
                str(p), sess_options=opts, providers=[provider]
            )
        print(f"Loaded {len(self.sessions)} Higgs ONNX sessions from {models_dir}")

    def __getitem__(self, key):
        return self.sessions[key]


# =============================================================================
# dtype helper — cast float32 inputs to whatever the ONNX session expects
# =============================================================================

_ORT_TO_NP = {
    "tensor(float16)": np.float16,
    "tensor(float)":   np.float32,
    "tensor(double)":  np.float64,
    "tensor(int64)":   np.int64,
    "tensor(int32)":   np.int32,
}


def _cast_feed(feed: dict, sess) -> dict:
    """Auto-cast numpy arrays to the dtype expected by the ONNX session.

    OnnxFloatToFloat16 converts all float inputs/outputs to float16. Feeding
    float32 to such a model raises INVALID_ARGUMENT. This inspects the session
    input metadata and casts each value to the declared type.
    """
    type_map = {inp.name: _ORT_TO_NP.get(inp.type) for inp in sess.get_inputs()}
    out = {}
    for k, v in feed.items():
        tgt = type_map.get(k)
        if tgt is not None and isinstance(v, np.ndarray) and v.dtype != tgt:
            v = v.astype(tgt)
        out[k] = v
    return out


# =============================================================================
# Encode: audio → codec codes
# =============================================================================

def encode(sessions: HiggsOnnxSessions, wav_path: str) -> np.ndarray:
    """Encode an audio file to RVQ codec codes.

    Pipeline:
      wav_path (24 kHz) → acoustic_encoder → (1, 256, T_a)
      wav_path (16 kHz) → semantic_encoder → (1, 768, T_s)
      align T_s → T_a   (linear interpolation along time axis)
      concat channels   → quantizer_encoder → (8, 1, T_a)  int64

    Returns:
      codes: np.ndarray  shape (8, 1, T_a)  dtype int64
      T_a is the number of codec frames (25 per second of 24 kHz audio).
    """
    # 1. Load audio at both sample rates
    wav24 = load_wav(wav_path, SR_24K)   # (T_24k,)
    wav16 = load_wav(wav_path, SR_16K)   # (T_16k,)

    dur = len(wav24) / SR_24K
    print(f"  Audio: {dur:.2f}s  "
          f"({len(wav24)} samples @ {SR_24K} Hz, "
          f"{len(wav16)} samples @ {SR_16K} Hz)")

    # 2. Acoustic encoder  → (1, 256, T_a)
    waveform_24k = wav24[None, None, :]          # (1, 1, T_24k)
    _ae_sess = sessions["acoustic_encoder"]
    acoustic_feat = _ae_sess.run(
        ["acoustic_features"],
        _cast_feed({"waveform_24k": waveform_24k}, _ae_sess)
    )[0]                                          # (1, 256, T_a)
    T_a = acoustic_feat.shape[2]
    print(f"  acoustic_encoder → {acoustic_feat.shape}  ({T_a} frames, {T_a/dur:.1f} fps)")

    # 3. Semantic encoder  → (1, 768, T_s)
    waveform_16k = wav16[None, :]                # (1, T_16k)
    _se_sess = sessions["semantic_encoder"]
    semantic_feat = _se_sess.run(
        ["semantic_features"],
        _cast_feed({"waveform_16k": waveform_16k}, _se_sess)
    )[0]                                          # (1, 768, T_s)
    T_s = semantic_feat.shape[2]
    print(f"  semantic_encoder → {semantic_feat.shape}  ({T_s} frames, {T_s/dur:.1f} fps)")

    # 4. Frame alignment check
    #    The fixed semantic_encoder wrapper applies semantic_downsample_factor=2 and
    #    the (160,160) padding internally, matching _extract_semantic_features() exactly.
    #    T_s == T_a naturally for typical audio. Trim on rare off-by-one edge cases.
    if T_s != T_a:
        T = min(T_a, T_s)
        acoustic_feat = acoustic_feat[:, :, :T]
        semantic_feat = semantic_feat[:, :, :T]
        print(f"  [warn] T_a={T_a} != T_s={T_s}, trimmed both to T={T}")

    # 5. Quantizer encoder  → (8, 1, T_a)  int64
    _qe_sess = sessions["quantizer_encoder"]
    codes = _qe_sess.run(
        ["codes"],
        _cast_feed({
            "acoustic_features": acoustic_feat,    # (1, 256, T_a)
            "semantic_features":  semantic_feat,   # (1, 768, T_a)
        }, _qe_sess)
    )[0]                                           # (8, 1, T_a)
    print(f"  quantizer_encoder → {codes.shape}  "
          f"({N_CODEBOOKS} codebooks × {T_a} frames)  "
          f"value range [{codes.min()}, {codes.max()}]")

    return codes


# =============================================================================
# Decode: codec codes → audio
# =============================================================================

def decode(sessions: HiggsOnnxSessions, codes: np.ndarray) -> np.ndarray:
    """Decode RVQ codec codes back to a waveform.

    Args:
      codes: np.ndarray  shape (8, 1, T_frames)  dtype int64

    Returns:
      waveform: np.ndarray  shape (T_samples,)  float32  at 24 kHz
    """
    if codes.ndim != 3 or codes.shape[0] != N_CODEBOOKS:
        raise ValueError(
            f"codes must have shape (8, 1, T_frames), got {codes.shape}"
        )
    if codes.dtype != np.int64:
        codes = codes.astype(np.int64)

    T_frames = codes.shape[2]
    expected_dur = T_frames * HOP_LENGTH / SR_24K

    _hd_sess = sessions["higgs_decoder"]
    waveform = _hd_sess.run(
        ["waveform_24k"],
        _cast_feed({"codes": codes}, _hd_sess)
    )[0]                          # (1, 1, T_samples)

    waveform = waveform.squeeze()  # (T_samples,)
    actual_dur = len(waveform) / SR_24K
    print(f"  higgs_decoder → {waveform.shape}  "
          f"({actual_dur:.2f}s  expected {expected_dur:.2f}s)")

    return waveform


# =============================================================================
# Convenience: round-trip encode + decode
# =============================================================================

def encode_decode(sessions: HiggsOnnxSessions, wav_path: str) -> tuple:
    """Encode audio to codes then decode back to waveform.

    Returns:
      codes:    np.ndarray (8, 1, T_frames) int64
      waveform: np.ndarray (T_samples,)     float32
    """
    print("[Encode]")
    codes = encode(sessions, wav_path)
    print("[Decode]")
    waveform = decode(sessions, codes)
    return codes, waveform


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Higgs Audio V2 Tokenizer — standalone ONNX encode/decode"
    )
    parser.add_argument(
        "--models-dir", default=str(DEFAULT_MODELS_DIR),
        help=f"Higgs ONNX models directory (default: {DEFAULT_MODELS_DIR})"
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input audio file to encode (.wav or any format torchaudio/scipy supports)"
    )
    parser.add_argument(
        "--output", "-o", default="output.wav",
        help="Output WAV file for decoded audio (default: output.wav)"
    )
    parser.add_argument(
        "--encode-only", action="store_true",
        help="Only encode — save codes to --codes-out, do not decode"
    )
    parser.add_argument(
        "--decode-only", action="store_true",
        help="Only decode — load codes from --codes-in, skip encoding"
    )
    parser.add_argument(
        "--codes-out", default="codes.npy",
        help="Path to save encoded codes as .npy (default: codes.npy)"
    )
    parser.add_argument(
        "--codes-in", default=None,
        help="Path to load codes .npy for --decode-only mode"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Use CUDAExecutionProvider instead of CPU"
    )
    args = parser.parse_args()

    provider = "CUDAExecutionProvider" if args.cuda else "CPUExecutionProvider"

    # Validate argument combinations
    if args.encode_only and args.decode_only:
        parser.error("--encode-only and --decode-only are mutually exclusive")
    if not args.decode_only and args.input is None:
        parser.error("--input is required unless --decode-only is used")
    if args.decode_only and args.codes_in is None:
        parser.error("--codes-in is required when using --decode-only")

    # Load ONNX sessions
    sessions = HiggsOnnxSessions(args.models_dir, provider)

    # -------------------------------------------------------------------------
    if args.encode_only:
        print(f"\nEncoding: {args.input}")
        codes = encode(sessions, args.input)
        np.save(args.codes_out, codes)
        print(f"\nSaved codes → {args.codes_out}  shape={codes.shape}  dtype={codes.dtype}")

    # -------------------------------------------------------------------------
    elif args.decode_only:
        print(f"\nLoading codes from: {args.codes_in}")
        codes = np.load(args.codes_in)
        print(f"  codes shape={codes.shape}  dtype={codes.dtype}")
        print(f"\nDecoding...")
        waveform = decode(sessions, codes)
        save_wav(args.output, waveform, SR_24K)
        print(f"Saved WAV → {args.output}  ({len(waveform)/SR_24K:.2f}s @ {SR_24K} Hz)")

    # -------------------------------------------------------------------------
    else:
        # Full round-trip
        print(f"\nRound-trip encode → decode")
        print(f"  Input : {args.input}")
        print(f"  Output: {args.output}")
        codes, waveform = encode_decode(sessions, args.input)

        # Optionally save codes
        np.save(args.codes_out, codes)
        print(f"\nSaved codes → {args.codes_out}")

        save_wav(args.output, waveform, SR_24K)
        dur_in  = len(load_wav(args.input, SR_24K)) / SR_24K
        dur_out = len(waveform) / SR_24K
        print(f"Saved WAV → {args.output}  ({dur_out:.2f}s, input was {dur_in:.2f}s)")
        print(f"\nReconstruction note: codec compression (RVQ, 8 codebooks × 1024 entries)")
        print(f"at 25 fps introduces mild quality loss — this is expected.")


if __name__ == "__main__":
    main()
