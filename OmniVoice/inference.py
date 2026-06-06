"""
ONNX Runtime inference for Prince-1/OmniVoice.

Implements OmniVoice's 32-step iterative unmasking decoding loop using the three
exported backbone ONNX sub-models, and optionally the four Higgs Audio Tokenizer
ONNX sub-models for voice cloning from a reference audio clip.

Pipeline (auto-voice, no ref_audio):
  text → tokenize → iterative unmasking (32 steps) → audio codes → higgs_decoder → WAV

Pipeline (voice cloning, with ref_audio):
  ref_audio (24 kHz) → acoustic_encoder → acoustic_features
  ref_audio (16 kHz) → semantic_encoder → semantic_features
  acoustic_features + semantic_features → quantizer_encoder → ref_codes
  text + ref_codes prefix → iterative unmasking → new codes → higgs_decoder → WAV

Usage:
  python inference.py --text "Hello, how are you today?" --output speech.wav
  python inference.py --text "Say this in my voice" --ref_audio ref.wav --ref_text "Hello." --output cloned.wav
  python inference.py --model_dir cuda/models --text "Hello" --output hello.wav
"""
import argparse
import json
import numpy as np
from pathlib import Path


# =============================================================================
# ONNX session helpers
# =============================================================================

def load_sessions(model_dir: str, provider: str = "CPUExecutionProvider"):
    """Load backbone ONNX sessions (embeddings + LLM + heads)."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def _sess(name):
        p = Path(model_dir) / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run optimize.py first.")
        return ort.InferenceSession(str(p), sess_options=opts, providers=[provider])

    return {
        "audio_embeddings": _sess("audio_embeddings_encoder.onnx"),
        "llm_decoder":      _sess("llm_decoder.onnx"),
        "audio_heads":      _sess("audio_heads_decoder.onnx"),
    }


def load_higgs_sessions(higgs_dir: str, provider: str = "CPUExecutionProvider"):
    """Load Higgs Audio Tokenizer ONNX sessions (all four sub-models).

    higgs_dir should point to the higgs/models/ folder produced by
    ``python optimize.py --higgs-only``.
    """
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3

    def _sess(name):
        p = Path(higgs_dir) / name
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found.\n"
                "Export Higgs models first: python optimize.py --higgs-only"
            )
        return ort.InferenceSession(str(p), sess_options=opts, providers=[provider])

    return {
        "acoustic_encoder":  _sess("acoustic_encoder.onnx"),
        "semantic_encoder":  _sess("semantic_encoder.onnx"),
        "quantizer_encoder": _sess("quantizer_encoder.onnx"),
        "higgs_decoder":     _sess("higgs_decoder.onnx"),
    }


# =============================================================================
# Audio loading helpers
# =============================================================================

def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file and resample to target_sr.  Returns (T,) float32 in [-1, 1]."""
    try:
        import torchaudio
        import torch
        wav, sr = torchaudio.load(path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        # Mono-mix if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy().astype(np.float32)
    except ImportError:
        pass

    # Fallback: scipy + built-in resampling
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    sr, data = wavfile.read(path)
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data


# =============================================================================
# Higgs tokenizer ONNX pipeline
# =============================================================================

SR_24K = 24_000
SR_16K = 16_000


def higgs_encode(higgs_sessions: dict, wav_path: str) -> np.ndarray:
    """Encode a reference audio file to Higgs codec codes via the ONNX pipeline.

    Steps:
      1. Load audio at 24 kHz → (1, 1, T)  → acoustic_encoder  → (1, 256, T_a)
      2. Load audio at 16 kHz → (1, T)     → semantic_encoder  → (1, 768, T_a)
         (semantic_encoder internally pads (160,160) and applies downsample_factor=2,
          matching _extract_semantic_features() — T_s == T_a naturally)
      3. (1, 256, T_a) + (1, 768, T_a)    → quantizer_encoder → (8, 1, T_a)

    Returns codes: np.ndarray shape (8, 1, T_a) int64
    """
    # Load at both sample rates
    wav24 = load_audio(wav_path, SR_24K)
    wav16 = load_audio(wav_path, SR_16K)

    # acoustic_encoder: input (B=1, 1, T_24k)
    waveform_24k = wav24[None, None, :].astype(np.float32)  # (1, 1, T)
    acoustic_feat = higgs_sessions["acoustic_encoder"].run(
        ["acoustic_features"], {"waveform_24k": waveform_24k}
    )[0]  # (1, 256, T_a)

    # semantic_encoder: input (B=1, T_16k)
    # The ONNX model internally applies pad=(160,160) + average all hidden states
    # + downsample_factor=2, matching _extract_semantic_features() exactly.
    waveform_16k = wav16[None, :].astype(np.float32)  # (1, T)
    semantic_feat = higgs_sessions["semantic_encoder"].run(
        ["semantic_features"], {"waveform_16k": waveform_16k}
    )[0]  # (1, 768, T_a)  — naturally same T_a as acoustic

    # Safety trim for rare off-by-one edge cases (e.g. non-multiple audio lengths)
    T_a = acoustic_feat.shape[2]
    T_s = semantic_feat.shape[2]
    if T_s != T_a:
        T = min(T_a, T_s)
        acoustic_feat = acoustic_feat[:, :, :T]
        semantic_feat = semantic_feat[:, :, :T]

    # quantizer_encoder: inputs (1, 256, T_a) + (1, 768, T_a) → codes (8, 1, T_a)
    codes = higgs_sessions["quantizer_encoder"].run(
        ["codes"],
        {"acoustic_features": acoustic_feat, "semantic_features": semantic_feat}
    )[0]  # (8, 1, T_a)

    return codes


def higgs_decode(higgs_sessions: dict, codes: np.ndarray) -> np.ndarray:
    """Decode codec codes to a waveform via the ONNX higgs_decoder.

    codes: (8, 1, T_frames) int64
    Returns waveform: (T_samples,) float32 at 24 kHz
    """
    waveform = higgs_sessions["higgs_decoder"].run(
        ["waveform_24k"], {"codes": codes}
    )[0]  # (1, 1, T_samples)
    return waveform.squeeze()   # (T_samples,)


def save_wav(path: str, waveform: np.ndarray, sr: int = SR_24K):
    """Save a float32 numpy waveform as a 16-bit WAV file."""
    try:
        import soundfile as sf
        sf.write(path, waveform, sr, subtype="PCM_16")
        return
    except ImportError:
        pass
    from scipy.io import wavfile
    pcm = np.clip(waveform, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    wavfile.write(path, sr, pcm16)


# =============================================================================
# Backbone forward (one iterative decoding step)
# =============================================================================

def run_backbone_step(sessions: dict, input_ids: np.ndarray, audio_mask: np.ndarray):
    """Execute one full backbone forward pass (one unmasking step).

    input_ids  : (B, 8, S)  int64
    audio_mask : (B, S)     bool
    Returns logits: (B, 8, S, 1025)
    """
    # 1. audio_embeddings_encoder
    embeds = sessions["audio_embeddings"].run(
        ["inputs_embeds"],
        {"input_ids": input_ids, "audio_mask": audio_mask}
    )[0]   # (B, S, 1024)

    B, S, _ = embeds.shape

    # 2. llm_decoder
    llm_sess   = sessions["llm_decoder"]
    llm_inputs = llm_sess.get_inputs()
    attn_mask  = np.ones((B, S), dtype=np.int64)
    pos_ids    = np.arange(S, dtype=np.int64)[None, :]
    feed = {
        "inputs_embeds":  embeds,
        "attention_mask": attn_mask,
        "position_ids":   pos_ids,
    }
    for inp in llm_inputs:
        if "past" in inp.name:
            feed[inp.name] = np.zeros((B, 8, 0, 128), dtype=np.float32)

    hidden_states = llm_sess.run(["hidden_states"], feed)[0]   # (B, S, 1024)

    # 3. audio_heads_decoder
    logits = sessions["audio_heads"].run(
        ["logits"], {"hidden_states": hidden_states}
    )[0]   # (B, 8, S, 1025)

    return logits


# =============================================================================
# Iterative unmasking decoding
# =============================================================================

def iterative_unmask(
    sessions: dict,
    text_tokens: list,
    num_audio_tokens: int = 256,
    num_steps: int = 32,
    audio_mask_id: int = 1024,
    num_codebooks: int = 8,
    prefix_codes: np.ndarray = None,
) -> np.ndarray:
    """OmniVoice iterative unmasking decoding (confidence-based greedy).

    prefix_codes: optional (8, 1, T_ref) int64 codes from a reference audio clip.
        When provided, they are prepended to the audio portion of the sequence so
        the model hears the voice style while generating the new audio tokens.

    Returns audio_codes: (8, num_audio_tokens) int64
    """
    T_text = len(text_tokens)
    T_ref  = prefix_codes.shape[2] if prefix_codes is not None else 0
    T_gen  = num_audio_tokens
    S      = T_text + T_ref + T_gen
    B      = 1

    # Build input_ids: (B, 8, S)
    input_ids = np.zeros((B, num_codebooks, S), dtype=np.int64)

    # Fill text positions (all 8 rows use the same text token IDs)
    for cb in range(num_codebooks):
        input_ids[0, cb, :T_text] = text_tokens

    # Fill ref audio positions (if voice cloning)
    if T_ref > 0:
        ref = prefix_codes[:, 0, :]   # (8, T_ref)
        for cb in range(num_codebooks):
            input_ids[0, cb, T_text : T_text + T_ref] = ref[cb]

    # Fill generation positions with MASK token
    for cb in range(num_codebooks):
        input_ids[0, cb, T_text + T_ref :] = audio_mask_id

    # audio_mask: True only at the positions we want to generate (not the ref prefix)
    audio_mask = np.zeros((B, S), dtype=bool)
    audio_mask[0, T_text + T_ref :] = True

    # Codebook unmasking weights (OmniVoice uses [8,8,6,6,4,4,2,2] priority per CB)
    cb_weights = np.array([8, 8, 6, 6, 4, 4, 2, 2], dtype=np.float32)
    cb_weights = cb_weights / cb_weights.sum()

    num_masked = T_gen
    gen_start  = T_text + T_ref   # index into S where generation region begins

    for step in range(num_steps):
        if num_masked == 0:
            break

        logits = run_backbone_step(sessions, input_ids, audio_mask)
        # logits: (1, 8, S, 1025) — take only the generation region
        gen_logits = logits[0, :, gen_start:, :]   # (8, T_gen, 1025)

        # Confidence = max probability over real tokens (exclude MASK token at 1024)
        real_logits = gen_logits[:, :, :1024]   # (8, T_gen, 1024)
        prob = np.exp(real_logits - real_logits.max(axis=-1, keepdims=True))
        prob = prob / prob.sum(axis=-1, keepdims=True)
        max_prob = prob.max(axis=-1)             # (8, T_gen)

        # Weighted confidence across codebooks
        confidence = (max_prob * cb_weights[:, None]).sum(axis=0)  # (T_gen,)

        # Find still-masked positions in the generation region
        masked_pos = np.where(input_ids[0, 0, gen_start:] == audio_mask_id)[0]
        if len(masked_pos) == 0:
            break

        # Unmask the most-confident position
        best = masked_pos[confidence[masked_pos].argmax()]
        for cb in range(num_codebooks):
            input_ids[0, cb, gen_start + best] = gen_logits[cb, best, :1024].argmax()
        num_masked -= 1

        if (step + 1) % 8 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{num_steps}: {num_masked:4d} positions remain masked")

    return input_ids[0, :, gen_start:]   # (8, T_gen)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OmniVoice ONNX inference")
    parser.add_argument("--model_dir", default="cpu_and_mobile/models",
                        help="Directory containing backbone ONNX models (default: cpu_and_mobile/models)")
    parser.add_argument("--higgs_dir", default="higgs/models",
                        help="Directory containing Higgs tokenizer ONNX models (default: higgs/models)")
    parser.add_argument("--text", required=True, help="Text to synthesise")
    parser.add_argument("--output", default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument("--ref_audio", default=None,
                        help="Reference audio file for voice cloning (.wav)")
    parser.add_argument("--ref_text", default=None,
                        help="Transcription of the reference audio (optional, used for logging)")
    parser.add_argument("--num_audio_tokens", type=int, default=256,
                        help="Number of audio frames to generate (default: 256, ~10 s)")
    parser.add_argument("--num_steps", type=int, default=32,
                        help="Iterative unmasking steps (default: 32)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDAExecutionProvider")
    args = parser.parse_args()

    provider = "CUDAExecutionProvider" if args.cuda else "CPUExecutionProvider"

    # --- Load backbone sessions ---
    print(f"Loading backbone ONNX models from: {args.model_dir}")
    sessions = load_sessions(args.model_dir, provider)
    print(f"  Loaded: {list(sessions.keys())}")

    # --- Load Higgs sessions ---
    print(f"Loading Higgs tokenizer ONNX models from: {args.higgs_dir}")
    higgs = load_higgs_sessions(args.higgs_dir, provider)
    print(f"  Loaded: {list(higgs.keys())}")

    # --- Tokenise input text ---
    print("\nTokenising text...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Prince-1/OmniVoice", trust_remote_code=True)
    text_tokens = tokenizer.encode(args.text, add_special_tokens=True)
    print(f"  Tokens ({len(text_tokens)}): {text_tokens[:8]}{'...' if len(text_tokens) > 8 else ''}")

    # --- Voice cloning: encode reference audio ---
    prefix_codes = None
    if args.ref_audio:
        print(f"\nEncoding reference audio: {args.ref_audio}")
        if args.ref_text:
            print(f"  Reference text: {args.ref_text!r}")
        prefix_codes = higgs_encode(higgs, args.ref_audio)
        print(f"  Reference codes: {prefix_codes.shape}  "
              f"(8 codebooks × {prefix_codes.shape[2]} frames "
              f"≈ {prefix_codes.shape[2] / 25:.1f}s)")

    # --- Generate audio codes ---
    mode = "voice cloning" if prefix_codes is not None else "auto-voice"
    print(f"\nGenerating audio [{mode}] ({args.num_steps} unmasking steps, "
          f"{args.num_audio_tokens} output frames ≈ "
          f"{args.num_audio_tokens * 960 / SR_24K:.1f}s)...")

    audio_codes = iterative_unmask(
        sessions,
        text_tokens,
        num_audio_tokens=args.num_audio_tokens,
        num_steps=args.num_steps,
        prefix_codes=prefix_codes,
    )
    print(f"  Generated codes: {audio_codes.shape}")

    # --- Decode codes → waveform using Higgs decoder ---
    print("\nDecoding audio codes → waveform...")
    # higgs_decoder expects (8, 1, T)
    codes_for_decoder = audio_codes[:, None, :]   # (8, 1, T_gen)
    waveform = higgs_decode(higgs, codes_for_decoder)
    duration = len(waveform) / SR_24K
    print(f"  Waveform: {len(waveform)} samples ({duration:.2f}s at {SR_24K} Hz)")

    # --- Save WAV ---
    save_wav(args.output, waveform, SR_24K)
    print(f"\nSaved: {args.output}  ({duration:.2f}s, {SR_24K} Hz)")


if __name__ == "__main__":
    main()
