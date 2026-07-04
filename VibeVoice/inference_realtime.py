"""Realtime-0.5B streaming TTS inference over the exported ONNX sub-parts.

Pipeline (streaming variant, on ONNX via common):
  text → tokenize → embed_tokens → OnnxLLM.prefill (tts_language_model backbone) → per frame:
    last hidden → DiffusionSampler → acoustic latent → acoustic_decoder.onnx → audio chunk
    (streamed/appended immediately) → acoustic_connector.onnx → OnnxLLM.step
  → concatenate chunks → 24 kHz wav.

Usage (prefer uv run) — the ONLY model input is the built ONNX dir; model_id/device/precision are
derived from the path and printed:
  uv run inference_realtime.py --text "Hello there." --out rt.wav realtime/cpu_int4/models
  uv run inference_realtime.py --text "..." --max-frames 300 realtime/cuda_fp16/models

Notes:
  * Realtime ships a DECODER-ONLY acoustic tokenizer (no encoder) — matches this text→audio use.
  * The full streaming model also has a small base language_model + a tts_eos_classifier that
    decides when to stop; those are NOT part of the exported sub-parts, so generation stops at
    --max-frames here (no learned EOS). The acoustic_decoder is frames-dynamic, so chunks decode
    per-frame for true streaming.
  * int4 is lossy in hidden space (conditions the diffusion head); prefer --precision fp16.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def main():
    ap = argparse.ArgumentParser(description="Realtime-0.5B streaming TTS (ONNX)")
    ap.add_argument("model_path", help="built ONNX dir, e.g. realtime/cpu_int4/models")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="realtime_out.wav")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    args = ap.parse_args()

    key, src, onnx_dir, device, precision, model_id = C.resolve_from_path(args.model_path)
    print(f"=== Realtime TTS | model_id={model_id}  device={device}  precision={precision} ===")
    if key != "realtime":
        sys.exit(f"[error] {model_id} ({key}) is not the Realtime model — "
                 f"use inference.py (1.5b) or inference_asr.py (asr/asr-hf)")
    for need in ("llm_decoder", "diffusion_head", "acoustic_decoder", "acoustic_connector"):
        if not (onnx_dir / f"{need}.onnx").exists():
            sys.exit(f"missing {need}.onnx in {onnx_dir} — build: uv run optimize.py {key}")

    cfg = json.loads((src / "config.json").read_text())
    dcfg = cfg["diffusion_head_config"]
    scale, bias = C.load_scaling(src)
    print(f"    scale={scale:.4f} bias={bias:.4f}")

    tok = C.load_tokenizer(onnx_dir, src)
    llm = C.OnnxLLM(onnx_dir / "llm_decoder.onnx", device)
    head = C.OnnxOp(onnx_dir / "diffusion_head.onnx", device)
    dec = C.OnnxOp(onnx_dir / "acoustic_decoder.onnx", device)
    conn = C.OnnxOp(onnx_dir / "acoustic_connector.onnx", device)
    sampler = C.DiffusionSampler(head, dcfg, device)

    ids = tok.encode(args.text)
    hidden = llm.prefill(C.embed_tokens(src, ids, key))
    H = hidden.shape[-1]
    neg = np.zeros(H, dtype=np.float32)

    chunks = []                                             # streamed per-frame (decoder is dynamic)
    for f in range(args.max_frames):
        cond = hidden[0, -1, :]
        latent = sampler.sample(cond, neg, cfg_scale=args.cfg_scale, n_frames=1, seed=f)  # [1,64]
        wav = C.acoustic_decode_to_wav(dec, latent[None], scale, bias)   # [1,samples] this frame
        chunks.append(np.asarray(wav).ravel())             # <-- emit immediately in a real stream
        hidden = llm.step(conn.run(features=latent[None]).astype(np.float32))
        if (f + 1) % 25 == 0:
            print(f"  streamed frame {f+1}/{args.max_frames}")

    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    C.save_wav(args.out, audio)
    print(f"wrote {args.out}  ({len(audio)} samples, {len(audio)/C.SR:.2f}s, {len(chunks)} frames)")


if __name__ == "__main__":
    main()
