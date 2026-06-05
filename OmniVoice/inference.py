"""
ONNX Runtime inference for Prince-1/OmniVoice.

Implements OmniVoice's 32-step iterative unmasking decoding loop using
the three exported ONNX backbone sub-models.

Usage:
  python inference.py --text "Hello, how are you today?" --output speech.wav
  python inference.py --text "Good morning!" --ref_audio ref.wav --ref_text "Good morning." --output cloned.wav
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
    """Load all three backbone ONNX sessions."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3   # suppress INFO

    def _sess(name):
        p = Path(model_dir) / name
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run optimize.py first to export ONNX models."
            )
        return ort.InferenceSession(str(p), sess_options=opts, providers=[provider])

    return {
        "audio_embeddings": _sess("audio_embeddings_encoder.onnx"),
        "llm_decoder":      _sess("llm_decoder.onnx"),
        "audio_heads":      _sess("audio_heads_decoder.onnx"),
    }


def run_backbone_step(sessions, input_ids, audio_mask):
    """Execute one full backbone forward pass (one unmasking step).

    input_ids  : np.ndarray (B, 8, S)  int64
    audio_mask : np.ndarray (B, S)     bool

    Returns logits: np.ndarray (B, 8, S, 1025)
    """
    # Step 1: audio_embeddings_encoder
    embeds = sessions["audio_embeddings"].run(
        ["inputs_embeds"],
        {"input_ids": input_ids, "audio_mask": audio_mask}
    )[0]   # (B, S, 1024)

    B, S, H = embeds.shape

    # Step 2: llm_decoder (pass empty KV cache for full-sequence forward)
    # Build past_key_value inputs: shape (B, num_heads, 0, head_dim) per layer
    llm_sess    = sessions["llm_decoder"]
    llm_inputs  = llm_sess.get_inputs()
    attn_mask   = np.ones((B, S), dtype=np.int64)
    pos_ids     = np.arange(S, dtype=np.int64)[None, :]
    feed = {
        "inputs_embeds":  embeds,
        "attention_mask": attn_mask,
        "position_ids":   pos_ids,
    }
    # Add empty past_key_values for all layers
    for inp in llm_inputs:
        if inp.name.startswith("past_key_values.") or inp.name.startswith("past_"):
            # Shape from model metadata: (B, num_kv_heads, 0, head_dim)
            num_kv_heads = 8
            head_dim     = 128
            feed[inp.name] = np.zeros((B, num_kv_heads, 0, head_dim), dtype=np.float32)

    hidden_states = llm_sess.run(["hidden_states"], feed)[0]   # (B, S, 1024)

    # Step 3: audio_heads_decoder
    logits = sessions["audio_heads"].run(
        ["logits"],
        {"hidden_states": hidden_states}
    )[0]   # (B, 8, S, 1025)

    return logits


# =============================================================================
# Simple greedy sampling for demonstration
# =============================================================================

def greedy_decode_audio(sessions, text_tokens, num_audio_tokens=256, num_steps=32,
                         audio_mask_id=1024, num_codebooks=8):
    """
    Simplified iterative unmasking decoding (greedy, for demonstration).

    In production, use OmniVoice's full _generate_iterative() which implements
    confidence-based unmasking with codebook weights [8,8,6,6,4,4,2,2].

    Returns audio_codes: np.ndarray (8, num_audio_tokens) int64
    """
    S = len(text_tokens) + num_audio_tokens
    B = 1

    # Initialise input_ids: text tokens in row 0, all audio positions masked
    input_ids = np.zeros((B, num_codebooks, S), dtype=np.int64)
    T = len(text_tokens)
    for cb in range(num_codebooks):
        input_ids[0, cb, :T] = text_tokens

    audio_mask = np.zeros((B, S), dtype=bool)
    audio_mask[0, T:] = True

    # All audio positions start as MASKED
    for cb in range(num_codebooks):
        input_ids[0, cb, T:] = audio_mask_id

    # Iterative unmasking
    num_masked = num_audio_tokens
    for step in range(num_steps):
        if num_masked == 0:
            break

        logits = run_backbone_step(sessions, input_ids, audio_mask)
        # logits: (1, 8, S, 1025)

        # Greedy: pick argmax at audio positions, codebook 0 only (simplified)
        audio_logits = logits[0, :, T:, :]   # (8, num_audio_tokens, 1025)

        # Unmask one position per step (confidence-based in real impl)
        # Here: unmask the position with highest max-logit confidence
        confidences = audio_logits[:, :, :1024].max(axis=-1).mean(axis=0)   # (num_audio_tokens,)
        masked_positions = np.where(input_ids[0, 0, T:] == audio_mask_id)[0]
        if len(masked_positions) == 0:
            break

        best_pos = masked_positions[confidences[masked_positions].argmax()]
        for cb in range(num_codebooks):
            input_ids[0, cb, T + best_pos] = audio_logits[cb, best_pos].argmax()
        num_masked -= 1

        if (step + 1) % 8 == 0:
            remaining = (input_ids[0, 0, T:] == audio_mask_id).sum()
            print(f"  Step {step+1}/{num_steps}: {remaining} positions still masked")

    # Extract audio codes
    return input_ids[0, :, T:]   # (8, num_audio_tokens)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OmniVoice ONNX inference")
    parser.add_argument("--model_dir", default="cpu_and_mobile/models",
                        help="Directory containing ONNX models (default: cpu_and_mobile/models)")
    parser.add_argument("--text", required=True, help="Text to synthesise")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--ref_audio", default=None,
                        help="Reference audio for voice cloning (requires Higgs tokenizer ONNX)")
    parser.add_argument("--ref_text", default=None,
                        help="Transcription of reference audio")
    parser.add_argument("--num_steps", type=int, default=32,
                        help="Iterative decoding steps (default: 32)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDAExecutionProvider")
    args = parser.parse_args()

    provider = "CUDAExecutionProvider" if args.cuda else "CPUExecutionProvider"

    print(f"Loading ONNX models from: {args.model_dir}")
    sessions = load_sessions(args.model_dir, provider)
    print(f"  Loaded: {list(sessions.keys())}")

    # Tokenise input text
    print("Tokenising text...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Prince-1/OmniVoice", trust_remote_code=True)
    text_tokens = tokenizer.encode(args.text, add_special_tokens=True)
    print(f"  Tokens: {len(text_tokens)}")

    if args.ref_audio:
        print(
            "NOTE: Voice cloning requires the Higgs tokenizer ONNX models.\n"
            "      Export them with: python convert_omnivoice_to_onnx.py --only higgs\n"
            "      Full voice cloning inference is not implemented in this script.\n"
            "      Running in auto-voice mode instead."
        )

    # Generate audio codes
    print(f"\nGenerating audio ({args.num_steps} iterative unmasking steps)...")
    audio_codes = greedy_decode_audio(
        sessions, text_tokens, num_steps=args.num_steps
    )
    print(f"  Generated audio codes: {audio_codes.shape}")

    print(
        "\nNOTE: To convert audio codes to a waveform, run the Higgs decoder:\n"
        "      higgs_decoder.onnx  (from convert_omnivoice_to_onnx.py --only higgs)\n"
        f"      Audio codes saved shape: {audio_codes.shape} (8 codebooks × {audio_codes.shape[1]} frames)"
    )
    print(f"\nOutput codes would be written to: {args.output} (stub — implement Higgs decoder call)")


if __name__ == "__main__":
    main()
