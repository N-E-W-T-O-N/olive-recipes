"""VibeVoice-1.5B TTS inference over the exported ONNX sub-parts.

Pipeline (reference algorithm, driven on ONNX via common):
  text → tokenize → embed_tokens → OnnxLLM.prefill → then per acoustic frame:
    last hidden state → DiffusionSampler (diffusion_head.onnx, DDPM/CFG) → acoustic latent
    → acoustic_decoder.onnx → waveform chunk
    → acoustic_connector.onnx (latent → LLM-space embed) → OnnxLLM.step (feeds next frame)
  concatenate chunks → 24 kHz wav.

Voice-cloning: the VibeVoice processor (vendored vibevoice/) builds the system + Speaker prompt and marks speech
placeholders; the reference voice is acoustic-encoded and spliced there (replicates
forward_speech_features). CFG uses a static "<|image_pad|>" negative. Single-shot (whole prompt in
one context) — chunking split phrasing and regressed some inputs.

Learned EOS: the 1.5B lm_head is TIED to embed_tokens, so logits = hidden @ embedᵀ (no lm_head
export needed). VibeVoice reuses vision tokens for speech, so generation stops when the model
predicts speech-end (<|vision_end|>) or EOS (<|endoftext|>). --max-frames is just a safety cap.

Usage — the ONLY model input is the built ONNX dir; model_id/device/precision derive from the path:
  uv run inference.py --text "Hello world." --voice samples/voices/en-Alice_woman.wav onnx/1.5b/cpu_fp32
  uv run inference.py --text "..." --cfg-scale 1.3 --max-frames 0 onnx/1.5b/cpu_fp32   # 0 = auto cap

Notes:
  * Use **fp32** — int4 is lossy in raw-hidden space (conditions the diffusion head) → audible hiss.
  * timbre-cloning is approximate and digits read weakly (1.5B model quality). See STATUS.md.
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
    ap.add_argument("--max-frames", type=int, default=0,
                    help="safety cap on acoustic frames (7.5/sec); generation normally stops earlier "
                         "via the learned EOS. 0 = auto-cap from text length (~5/word)")
    ap.add_argument("--cfg-scale", type=float, default=1.3)
    ap.add_argument("--voice", default="samples/voices/en-Alice_woman.wav",
                    help="reference voice wav for cloning (VibeVoice is voice-conditioned); "
                         "'' = text-only smoke-test (unconditioned, not intelligible)")
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

    # CFG negative (static "<|image_pad|>" — see below), computed once and reused for every chunk.
    neg_vec = None
    if args.cfg_scale != 1.0:
        neg_id = tok.convert_tokens_to_ids("<|image_pad|>")
        if neg_id is None or neg_id < 0:
            print("    [warn] no <|image_pad|> token — CFG disabled")
        else:
            nh = llm.prefill(common.embed_tokens(src, [neg_id], key))
            neg_vec = nh[0, -1, :].copy(); llm._reset()
            print(f"    CFG negative: static '<|image_pad|>' (id={neg_id}), scale={args.cfg_scale}")

    # Single-shot generation (chunking regressed some inputs — the whole prompt in one context is
    # what works). Voice-conditioned prefill, then the AR diffusion frame loop.
    from pathlib import Path as _P
    if args.voice and _P(args.voice).exists():
        print(f"    voice-cloning prompt from {args.voice}")
        embeds, ids = common.voice_prompt_embeds(src, onnx_dir, args.text, args.voice, scale, bias, device)
    else:
        if args.voice:
            print(f"    [warn] voice '{args.voice}' not found — text-only (unconditioned)")
        embeds = common.embed_tokens(src, tok.encode(args.text), key)
    llm._reset()
    hidden = llm.prefill(embeds)
    H = hidden.shape[-1]

    # Safety cap only — the learned EOS below normally stops first. Auto-estimated from text (~5/word).
    n_frames = args.max_frames if args.max_frames > 0 else min(400, 12 + len(args.text.split()) * 5)
    if args.max_frames <= 0:
        print(f"    frame cap={n_frames} ({len(args.text.split())} words, ~{n_frames/7.5:.1f}s) — EOS stops earlier")

    # Learned EOS: lm_head is TIED to embed_tokens, so logits = hidden @ embedᵀ. VibeVoice reuses
    # vision tokens for speech — stop when the model predicts speech-end (<|vision_end|>) or EOS
    # (<|endoftext|>). This removes the no-EOS overshoot ("jargon tail") instead of guessing a budget.
    emb_W = common.load_embed_matrix(src, key)                 # [vocab, H]
    end_ids = {tok.convert_tokens_to_ids(t) for t in ("<|vision_end|>", "<|endoftext|>")}
    end_ids = {i for i in end_ids if isinstance(i, int) and i >= 0}

    latents = []
    for f in range(n_frames):
        cond = hidden[0, -1, :]
        neg = neg_vec if neg_vec is not None else np.zeros(H, dtype=np.float32)
        latent = sampler.sample(cond, neg, cfg_scale=args.cfg_scale, n_frames=1, seed=f)
        latents.append(latent[0])
        hidden = llm.step(conn.run(features=latent[None]).astype(np.float32))
        if end_ids and int((hidden[0, -1] @ emb_W.T).argmax()) in end_ids:  # tied-lm_head EOS
            print(f"  EOS predicted at frame {f+1}/{n_frames} — stop")
            break
        if (f + 1) % 25 == 0:
            print(f"  frame {f+1}/{n_frames}")

    lat_seq = np.stack(latents)[None].astype(np.float32)
    audio = np.asarray(common.acoustic_decode_to_wav(dec, lat_seq, scale, bias)).ravel()
    common.save_wav(args.out, audio)
    print(f"wrote {args.out}  ({len(audio)} samples, {len(audio)/common.SR:.2f}s, {len(latents)} frames)")


if __name__ == "__main__":
    main()
