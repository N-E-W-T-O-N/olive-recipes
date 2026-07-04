"""VibeVoice-1.5B TTS inference over the exported ONNX sub-parts.

Pipeline (reference algorithm, driven on ONNX via common):
  text → tokenize → embed_tokens → OnnxLLM.prefill → then per acoustic frame:
    last hidden state → DiffusionSampler (diffusion_head.onnx, DDPM/CFG) → acoustic latent
    → acoustic_decoder.onnx → waveform chunk
    → acoustic_connector.onnx (latent → LLM-space embed) → OnnxLLM.step (feeds next frame)
  concatenate chunks → 24 kHz wav.

Usage (prefer uv run) — the ONLY model input is the built ONNX dir; model_id/device/precision are
derived from the path and printed:
  uv run inference.py --text "Hello world." --out out.wav model/cpu_int4/models
  uv run inference.py --text "..." --max-frames 200 model/cuda_fp16/models

Notes:
  * The 1.5B decoder is exported WITHOUT lm_head (its head is the diffusion head), so there is
    no vocab EOS token to detect here — generation stops at --max-frames. Faithful EOS needs the
    lm_head / a stop-token classifier (not part of this sub-part decomposition).
  * int4 is lossy in raw-hidden space (what conditions the diffusion head); build a fp16 dir
    for higher-fidelity audio. Exact prompt layout/voice-cloning needs the original VibeVoice
    processor assets (repos ship none — we use the Qwen2.5 tokenizer).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common 


def main():
    ap = argparse.ArgumentParser(description="VibeVoice-1.5B TTS (ONNX)")
    ap.add_argument("model_path", help="built ONNX dir, e.g. model/cpu_int4/models")
    ap.add_argument("--text", required=True, help="text to synthesize")
    ap.add_argument("--out", default="tts_out.wav", help="output wav path")
    ap.add_argument("--max-frames", type=int, default=200, help="acoustic frames to generate")
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    args = ap.parse_args()

    key, src, onnx_dir, device, precision, model_id = common.resolve_from_path(args.model_path)
    print(f"=== TTS | model_id={model_id}  device={device}  precision={precision} ===")
    if key != "1.5b":
        sys.exit(f"[error] {model_id} ({key}) is not the 1.5B TTS model — "
                 f"use inference_asr.py (asr/asr-hf) or inference_realtime.py (realtime)")
    for need in ("llm_decoder", "diffusion_head", "acoustic_decoder", "acoustic_connector"):
        if not (onnx_dir / f"{need}.onnx").exists():
            sys.exit(f"missing {need}.onnx in {onnx_dir} — build it: uv run optimize.py {key}")

    cfg = json.loads((src / "config.json").read_text())
    dcfg = cfg["diffusion_head_config"]
    scale, bias = common.load_scaling(src)
    print(f"    scale={scale:.4f} bias={bias:.4f}")

    tok = common.load_tokenizer(onnx_dir, src)
    llm = common.OnnxLLM(onnx_dir / "llm_decoder.onnx", device)
    head = common.OnnxOp(onnx_dir / "diffusion_head.onnx", device)
    dec = common.OnnxOp(onnx_dir / "acoustic_decoder.onnx", device)
    conn = common.OnnxOp(onnx_dir / "acoustic_connector.onnx", device)
    sampler = common.DiffusionSampler(head, dcfg, device)

    # Prompt: embed the text tokens and prefill the backbone. (Voice-clone/speaker prompts
    # would prepend reference-audio acoustic embeds here — needs the processor's token layout.)
    ids = tok.encode(args.text)
    embeds = common.embed_tokens(src, ids, key)                  # [1,S,H]
    hidden = llm.prefill(embeds)                            # [1,S,H]
    H = hidden.shape[-1]
    neg = np.zeros(H, dtype=np.float32)                     # unconditional = zero hidden

    # Autoregressive frame loop: each latent is decoded together at the end (the conv codec has a
    # cross-frame receptive field, so one decode of the whole sequence avoids block-boundary seams).
    latents = []
    for f in range(args.max_frames):
        cond = hidden[0, -1, :]                             # condition on last hidden
        latent = sampler.sample(cond, neg, cfg_scale=args.cfg_scale, n_frames=1, seed=f)  # [1,64]
        latents.append(latent[0])
        nxt = conn.run(features=latent[None])              # [1,1,H] LLM-space embed
        hidden = llm.step(nxt.astype(np.float32))          # advance one frame
        if (f + 1) % 25 == 0:
            print(f"  frame {f+1}/{args.max_frames}")

    lat_seq = np.stack(latents)[None].astype(np.float32)   # [1,T,64]
    exp = dec.sess.get_inputs()[0].shape[1]                # decoder's frame axis (int if static)
    if isinstance(exp, int) and exp != lat_seq.shape[1]:
        sys.exit(f"acoustic_decoder expects {exp} frames but got {lat_seq.shape[1]} — this decoder "
                 f"was built with a static frame count; re-export with dynamic frames: "
                 f"uv run optimize.py --components acoustic_decoder {key}")
    audio = np.asarray(common.acoustic_decode_to_wav(dec, lat_seq, scale, bias)).ravel()
    common.save_wav(args.out, audio)
    print(f"wrote {args.out}  ({len(audio)} samples, {len(audio)/common.SR:.2f}s, {len(latents)} frames)")


if __name__ == "__main__":
    main()
