"""VibeVoice ASR inference (audio → text) over the exported ONNX sub-parts.

Handles BOTH ASR checkpoints (choose via the positional model — key or path):
  asr     (VibeVoice-ASR, codes/ family)  : acoustic_connector + semantic_connector, fused = sum
  asr-hf  (VibeVoice-ASR-HF, transformers) : multi_modal_projector fuses acoustic+semantic

Pipeline (reference algorithm on ONNX via common):
  audio → acoustic_encoder + semantic_encoder → (connectors | projector) → speech features
  → build [system + <speech_start> <speech_pad>*N <speech_end> + instruction] embeds, inject the
    speech features at the <speech_pad> positions → OnnxLLM.prefill (decoder KEEPS lm_head → logits)
  → greedy-decode tokens until EOS → tokenizer.decode → transcript.

Usage (prefer uv run) — the ONLY model input is the built ONNX dir; model_id/device/precision are
derived from the path and printed:
  uv run inference_asr.py --audio speech.wav asr-hf/cpu_int4/models
  uv run inference_asr.py --audio speech.wav asr/cuda_fp16/models

Note: the ASR LLM is Qwen2.5-7B; its int4 serialize is RAM-bound (see STATUS.md), so
llm_decoder.onnx may not be built. Without it this still runs the audio front-end and reports the
fused-feature shape, then explains that the 7B decoder must be built to produce text.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C

SPEECH_TOKENS = ["<|speech_start|>", "<|speech_pad|>", "<|speech_end|>"]


def _resolve_speech_ids(tok):
    ids = {}
    for t in SPEECH_TOKENS:
        i = tok.convert_tokens_to_ids(t)
        ids[t] = i if isinstance(i, int) and i >= 0 else None
    return ids


def encode_audio_features(key, src, onnx_dir, device, wav):
    """Return fused speech features [N, H] and frame count N from the ONNX front-end."""
    a_enc = C.OnnxOp(onnx_dir / "acoustic_encoder.onnx", device)
    s_enc = C.OnnxOp(onnx_dir / "semantic_encoder.onnx", device)
    x = wav[None, None, :].astype(np.float32)               # [1,1,T]
    al = a_enc.run(audio=x)                                  # [1,fa,64]
    sl = s_enc.run(audio=x)                                  # [1,fs,128]
    n = min(al.shape[1], sl.shape[1])
    al, sl = al[:, :n].astype(np.float32), sl[:, :n].astype(np.float32)
    proj = C.component_path(onnx_dir, "multi_modal_projector")
    if proj:                                                # asr-hf
        fused = C.OnnxOp(proj, device).run(acoustic_latents=al, semantic_latents=sl)   # [1,n,H]
    else:                                                   # asr: sum of the two connectors
        ac = C.OnnxOp(onnx_dir / "acoustic_connector.onnx", device).run(features=al)
        sc = C.OnnxOp(onnx_dir / "semantic_connector.onnx", device).run(features=sl)
        fused = ac + sc
    return fused[0].astype(np.float32), n                    # [n,H], n


def main():
    ap = argparse.ArgumentParser(description="VibeVoice ASR (ONNX)")
    ap.add_argument("model_path", help="built ONNX dir, e.g. asr-hf/cpu_int4/models")
    ap.add_argument("--audio", required=True, help="input wav")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt", default="Please transcribe the audio.")
    args = ap.parse_args()

    key, src, onnx_dir, device, precision, model_id = C.resolve_from_path(args.model_path)
    print(f"=== ASR | model_id={model_id}  device={device}  precision={precision} ===")
    if key not in ("asr", "asr-hf"):
        sys.exit(f"[error] {model_id} ({key}) is not an ASR model — "
                 f"use inference.py (1.5b) or inference_realtime.py (realtime)")
    for need in ("acoustic_encoder", "semantic_encoder"):
        if not (onnx_dir / f"{need}.onnx").exists():
            sys.exit(f"missing {need}.onnx in {onnx_dir} — build: uv run optimize.py {key}")

    wav = C.normalize_audio(C.load_audio(args.audio))
    feats, n = encode_audio_features(key, src, onnx_dir, device, wav)
    H = feats.shape[-1]
    print(f"  audio {len(wav)/C.SR:.2f}s → {n} speech frames → fused features [{n},{H}]")

    llm_path = onnx_dir / "llm_decoder.onnx"
    if not llm_path.exists():
        print(f"\n[front-end OK] llm_decoder.onnx not built (Qwen2.5-7B int4 is RAM-bound — see STATUS.md).")
        print(f"  Build it after freeing RAM: uv run optimize.py --components llm {key}")
        print(f"  Then re-run for the transcript.")
        return

    tok = C.load_tokenizer(onnx_dir, src)
    sid = _resolve_speech_ids(tok)
    # Prompt: system + user(<speech_start> <speech_pad>*n <speech_end> + instruction).
    pad = sid["<|speech_pad|>"]
    if pad is None:
        # tokenizer lacks the speech tokens (repo shipped none) — add them, then use a pad id.
        tok.add_special_tokens({"additional_special_tokens": SPEECH_TOKENS})
        sid = _resolve_speech_ids(tok); pad = sid["<|speech_pad|>"]
    start = sid["<|speech_start|>"] if sid["<|speech_start|>"] is not None else pad
    end = sid["<|speech_end|>"] if sid["<|speech_end|>"] is not None else pad
    pre = tok.encode(f"{args.prompt}\n")
    ids = pre + [start] + [pad] * n + [end]
    embeds = C.embed_tokens(src, ids, key)                 # [1,S,H]
    # inject at the n dedicated speech-pad slots (the [pad]*n block), not any incidental pad id
    pad_pos = list(range(len(pre) + 1, len(pre) + 1 + n))
    embeds[0, pad_pos, :] = feats[:len(pad_pos)]            # inject speech features

    llm = C.OnnxLLM(llm_path, device)
    logits = llm.prefill(embeds)                            # [1,S,V]
    eos = getattr(tok, "eos_token_id", None)
    out = []
    for _ in range(args.max_new_tokens):
        nxt = int(np.asarray(logits)[0, -1].argmax())
        if eos is not None and nxt == eos:
            break
        out.append(nxt)
        logits = llm.step(C.embed_tokens(src, [nxt], key))
    text = tok.decode(out, skip_special_tokens=True)
    print(f"\nTRANSCRIPT:\n{text}")


if __name__ == "__main__":
    main()
