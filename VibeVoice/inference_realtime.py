"""Realtime-0.5B streaming TTS inference over the exported ONNX sub-parts.

CORRECT pipeline (fixed 2026-08-04 — see "the babble bug" below):
  text ids -> text_lm.onnx (the 4-layer `language_model`)          -> lm_hidden [1,T,H]
  lm_hidden + type_embed[1]                                        -> llm_decoder prefill
  per frame:
    last hidden -> DiffusionSampler -> acoustic latent
    latent -> acoustic_decoder.onnx -> audio chunk (streamed)
    latent -> acoustic_connector.onnx -> acoustic_embed
    acoustic_embed + type_embed[0]                                 -> llm_decoder step
  -> concatenate chunks -> 24 kHz wav.

THE BABBLE BUG (what this file used to do):
  Realtime has TWO language models. `tts_language_model` (20 layers) is the TTS backbone shipped as
  llm_decoder.onnx; `language_model` (4 layers) encodes the TEXT. The 4-layer text LM was never
  exported, so this driver fed RAW TEXT-TOKEN EMBEDDINGS straight into the TTS backbone — which was
  never trained to consume them. The backbone, diffusion head and acoustic decoder all worked, so
  the output had real voice timbre, but no text conditioning ever happened: fluent-sounding babble,
  no error, and `eval.py` passed 5/5 because it never runs generation. Both int4 and fp32 failed
  IDENTICALLY, which is the tell that a defect is structural rather than numeric.
  Upstream reference: vibevoice/modular/modeling_vibevoice_streaming_inference.py::forward_tts_lm
      start = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
      inputs_embeds[:, start:, :] = lm_last_hidden_state    # splice into the tail
      inputs_embeds += tts_input_types(tts_text_masks)      # 1 = text, 0 = speech
  Because the spliced region covers the whole (pseudo-pad) sequence, this reduces exactly to
  "TTS-LM input = lm_hidden (or acoustic_embed) + type_embed[type]" — what we do here.

Usage (prefer uv run) — the ONLY model input is the built ONNX dir:
  uv run inference_realtime.py --text "Hello there." --out rt.wav onnx/realtime/cpu_int4
  uv run inference_realtime.py --text "..." --max-frames 300 onnx/realtime/cuda_fp16

Notes:
  * Realtime ships a DECODER-ONLY acoustic tokenizer (no encoder) — matches this text->audio use.
  * `embeddings.onnx` is NOT used for realtime: text ids go through text_lm.onnx, which carries its
    own embed_tokens. It is still shipped for parity with the other checkpoints.
  * `tts_eos_classifier` IS exported (eos_head.npz, run in numpy) so generation stops on its own;
    --max-frames is only a hard cap. Without it every frame past the end of the text was drift.
  * CFG uses a real parallel NEGATIVE TTS-LM stream (--neg-mode stream, the default) and text is fed
    in 5-token windows interleaved with 6-frame speech windows, as upstream does.
  * int4 is lossy in hidden space (it conditions the diffusion head); prefer fp16/fp32 for quality.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def load_type_embed(onnx_dir: Path, H: int):
    """[2, H] `tts_input_types` lookup: row 1 = text, row 0 = speech. Added to EVERY TTS-LM input.

    Shipped as type_embed.npy next to the ONNX (2 rows — too small to justify a graph). Missing it
    is not fatal, but the type tag is part of what the backbone was trained on, so we warn loudly.
    """
    p = Path(onnx_dir) / "type_embed.npy"
    if p.exists():
        w = np.load(p).astype(np.float32)
        if w.shape == (2, H):
            return w
        print(f"  [type_embed][warn] {p} has shape {w.shape}, expected (2,{H}) — ignoring")
    else:
        print(f"  [type_embed][warn] {p} not found — rebuild with "
              f"`optimize.py --components text_lm realtime`. Output quality will suffer.")
    return np.zeros((2, H), dtype=np.float32)


def load_eos_head(onnx_dir: Path, H: int):
    """The `tts_eos_classifier` — the model's LEARNED STOP signal. Returns a callable or None.

    Shipped as eos_head.npz next to the ONNX. Upstream
    (modeling_vibevoice_streaming_inference.py:848) runs it on the TTS LM's last hidden state
    after each frame is fed back, and finishes the sample when sigmoid(logit) > 0.5.

    This matters far more than its size suggests. Without it the loop can only stop at
    --max-frames, so any frame beyond the end of the text is unconditioned drift: at
    TTS_SPEECH_WINDOW_SIZE=6 frames per 5-token text window, an 18-token prompt only
    justifies ~24 frames (3.2s at 7.5Hz) — a default of 300 would be >90% garbage.

    fc2(relu(fc1(x))) on one [H] vector, so numpy is plenty.
    """
    p = Path(onnx_dir) / "eos_head.npz"
    if not p.exists():
        print(f"  [eos][warn] {p} not found — NO LEARNED STOP; generation will run to "
              f"--max-frames and everything past the end of the text will be drift. "
              f"Rebuild with `optimize.py --components tts_lm realtime`.")
        return None
    z = np.load(p)
    fc1_w, fc1_b = z["fc1_w"].astype(np.float32), z["fc1_b"].astype(np.float32)
    fc2_w, fc2_b = z["fc2_w"].astype(np.float32), z["fc2_b"].astype(np.float32)
    if fc1_w.shape != (H, H) or fc2_w.shape != (1, H):
        print(f"  [eos][warn] eos_head.npz shapes {fc1_w.shape}/{fc2_w.shape} do not match "
              f"hidden size {H} — ignoring")
        return None

    def eos_prob(h):
        """h: [H] last hidden state -> P(stop) in [0,1]."""
        x = np.maximum(fc1_w @ h.astype(np.float32) + fc1_b, 0.0)   # relu(fc1)
        logit = (fc2_w @ x + fc2_b).item()      # fc2_w is [1,H] -> shape (1,), not a scalar
        return 1.0 / (1.0 + np.exp(-logit))                          # sigmoid

    print(f"  [eos] tts_eos_classifier loaded ({H}->{H}->1) — learned stop ACTIVE")
    return eos_prob


def _voice_kv(voice, stream):
    """Pull ([keys], [values]) for `stream` out of a loaded voice .npz, in layer order."""
    n = sum(1 for k in voice.files if k.startswith(f"{stream}.k."))
    return ([voice[f"{stream}.k.{i}"] for i in range(n)],
            [voice[f"{stream}.v.{i}"] for i in range(n)])


def resolve_voice(name):
    """--voice <name|path> -> loaded .npz, or None.

    Speaker identity in VibeVoice-Realtime lives ENTIRELY in a KV-cache prefix ("cached_prompt"
    upstream): the prompt's token ids are all pad_id, and the realtime checkpoint ships no acoustic
    encoder, so a voice cannot be built from a .wav here. Without a prompt the model generates
    unconditioned and pitch drifts across the clip (measured: 93->312 Hz in one 6 s sample).

    Run `uv run prepare_voices.py` once to make the .npz from the upstream .pt files.
    """
    if not name:
        return None
    p = Path(name)
    if not p.exists():
        vd = Path(__file__).parent / "voices" / "streaming_model"
        cands = sorted(vd.glob("*.npz"))
        if not cands:
            sys.exit(f"no voice prompts in {vd} — run `uv run prepare_voices.py` first "
                     f"(and download the .pt files from microsoft/VibeVoice "
                     f"demo/voices/streaming_model/).")
        key = name.lower()
        hit = ([c for c in cands if c.stem.lower() == key]
               or [c for c in cands if key in c.stem.lower()])
        if not hit:
            sys.exit(f"voice '{name}' not found. Available: "
                     f"{', '.join(c.stem for c in cands)}")
        p = hit[0]
    v = np.load(p)
    print(f"    voice prompt: {p.stem}")
    return v


def main():
    ap = argparse.ArgumentParser(description="Realtime-0.5B streaming TTS (ONNX)")
    ap.add_argument("model_path", help="built ONNX dir, e.g. onnx/realtime/cpu_int4")
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", default="realtime_out.wav")
    ap.add_argument("--max-frames", type=int, default=300,
                    help="Hard cap only. With the tts_eos_classifier exported, generation normally "
                         "stops on its own well before this.")
    ap.add_argument("--cfg-scale", type=float, default=3.0,
                    help="Classifier-free guidance. 3.0 default: judged clearly better than 1.3 on "
                         "this build (1.3 is mushy/less articulate).")
    ap.add_argument("--eos-threshold", type=float, default=0.5,
                    help="Stop when sigmoid(tts_eos_classifier) exceeds this (upstream uses 0.5). "
                         "Lower = stops earlier/more eagerly.")
    ap.add_argument("--decode-context", type=int, default=56, metavar="FRAMES",
                    help="Left-context frames replayed per step in --decode-mode stream. The "
                         "acoustic decoder's measured receptive field is ~56 frames (7.5 s); at 56 "
                         "the streamed output is bit-identical to a full batch decode. Lower it to "
                         "trade exactness for latency/compute: 48 = -52 dB, 32 = -44 dB, 16 = -31 dB.")
    ap.add_argument("--save-latents", metavar="PATH",
                    help="Dump the generated [1,F,64] acoustic latents to .npy. Needed for any "
                         "decoder analysis: fed RANDOM latents the decoder emits near-silence "
                         "(HANDOFF trap #7), so receptive-field/seam measurements are only "
                         "meaningful on real ones.")
    ap.add_argument("--no-eos", action="store_true",
                    help="Ignore the learned stop and always run to --max-frames (debug).")
    ap.add_argument("--voice", default="Carter",
                    help="Voice prompt name (Carter/Emma/Frank/Grace/Davis/Mike) or path to a .npz. "
                         "This is what fixes speaker identity — without it pitch drifts across the "
                         "clip. Pass --voice '' to generate unconditioned (debug).")
    ap.add_argument("--eos-min-frames", type=int, default=4,
                    help="Never honour the learned stop before this many frames, so a spuriously "
                         "confident first step cannot emit an empty clip.")
    ap.add_argument("--neg-mode", choices=["stream", "type", "zero"], default="stream",
                    help="CFG negative condition. 'stream' (default, FAITHFUL) = a parallel negative "
                         "TTS-LM stream prefilled with a single <|image_pad|> token and fed the same "
                         "acoustic frames; 'type'/'zero' are the old cheap approximations.")
    ap.add_argument("--text-window", type=int, default=5,
                    help="text tokens per window (upstream TTS_TEXT_WINDOW_SIZE=5)")
    ap.add_argument("--speech-window", type=int, default=6,
                    help="speech frames per text window (upstream TTS_SPEECH_WINDOW_SIZE=6)")
    ap.add_argument("--no-interleave", action="store_true",
                    help="prefill ALL text at once instead of interleaving (the old behaviour)")
    ap.add_argument("--decode-mode", choices=["batch", "stream"], default="batch",
                    help="'batch' (default, BEST QUALITY): collect all latents and run the acoustic "
                         "decoder ONCE, so its causal convolutions stay continuous. 'stream': decode "
                         "each frame separately for true low-latency streaming — but every frame "
                         "restarts the conv stack from zero padding, which adds a discontinuity at "
                         "each 3200-sample boundary (upstream avoids this with a streaming cache that "
                         "our stateless acoustic_decoder.onnx does not carry).")
    args = ap.parse_args()

    key, src, onnx_dir, device, precision, model_id = C.resolve_from_path(args.model_path)
    print(f"=== Realtime TTS | model_id={model_id}  device={device}  precision={precision} ===")
    if key != "realtime":
        sys.exit(f"[error] {model_id} ({key}) is not the Realtime model — "
                 f"use inference.py (1.5b) or inference_asr.py (asr/asr-hf)")
    need = ("llm_decoder", "text_lm", "diffusion_head", "acoustic_decoder", "acoustic_connector")
    for n in need:
        if not (onnx_dir / f"{n}.onnx").exists():
            extra = ("\n  text_lm.onnx is the 4-layer TEXT encoder. Without it the TTS backbone gets "
                     "raw text embeddings and the output is BABBLE — see this file's docstring."
                     if n == "text_lm" else "")
            sys.exit(f"missing {n}.onnx in {onnx_dir} — build: uv run optimize.py {key}{extra}")

    cfg = json.loads((src / "config.json").read_text())
    dcfg = cfg["diffusion_head_config"]
    scale, bias = C.load_scaling(src)
    print(f"    scale={scale:.4f} bias={bias:.4f}")

    tok = C.load_tokenizer(onnx_dir, src)
    # KV-cached: text windows accumulate context exactly as upstream's forward_lm does, and the
    # voice prompt's `lm` prefix can be seeded in (it exists ONLY as KV — upstream's prompt token
    # ids are all pad_id, so there is nothing to replay through a stateless graph).
    text_lm = C.OnnxLLM(onnx_dir / "text_lm.onnx", device)
    llm = C.OnnxLLM(onnx_dir / "llm_decoder.onnx", device)
    head = C.OnnxOp(onnx_dir / "diffusion_head.onnx", device)
    dec = C.OnnxOp(onnx_dir / "acoustic_decoder.onnx", device)
    conn = C.OnnxOp(onnx_dir / "acoustic_connector.onnx", device)
    sampler = C.DiffusionSampler(head, dcfg, device)

    # --- text ids: upstream encodes `text.strip() + "\n"` with add_special_tokens=False ---------
    # (VibeVoiceStreamingProcessor.process_input_with_cached_prompt). Chat/BOS specials are NOT
    # part of what this model saw in training; adding them shifts every downstream hidden state.
    ids_list = tok.encode(args.text.strip() + "\n", add_special_tokens=False)
    ids = np.asarray(ids_list, dtype=np.int64)[None]                        # [1,T]

    H = int(text_lm.step(ids[:, :1]).shape[-1])   # probe; seed_prompt/_reset below clears the cache
    text_lm._reset()
    type_embed = load_type_embed(onnx_dir, H)
    print(f"    text {ids.shape[1]} tokens (strip+\\n, no special tokens), hidden H={H}")

    voice = resolve_voice(args.voice)
    if voice is None:
        print("  [voice][warn] NO voice prompt — the model is unconditioned and pitch will drift "
              "across the clip. Pass --voice Carter (see prepare_voices.py).")
    else:
        # Seed the TEXT LM with the prompt's `lm` prefix before any text is fed. This is the last
        # structural gap vs the reference; with it, forward_lm's context is reproduced exactly.
        p = text_lm.seed_prompt(*_voice_kv(voice, "lm"))
        print(f"    seeded text LM with {p}-position voice prefix")

    def text_hidden(tok_ids):
        """[1,w] ids -> [1,w,H]. STATEFUL: appends to the text LM's KV cache, like forward_lm."""
        return np.asarray(text_lm.step(np.asarray(tok_ids, dtype=np.int64)), dtype=np.float32)

    # --- CFG negative: a PARALLEL TTS-LM stream, not a constant --------------------------------
    # Upstream prefills neg_lm/neg_tts_lm with ONE token: tokenizer("<|image_pad|>"), then feeds it
    # the SAME acoustic frames as the positive stream but never any text. So the guidance direction
    # is literally "with text" minus "without text". A constant vector (the old --neg-mode zero/type)
    # is not that, and weakens or misdirects the guidance.
    llm_neg = None
    neg_const = None
    if args.neg_mode == "stream":
        llm_neg = C.OnnxLLM(onnx_dir / "llm_decoder.onnx", device)       # its own KV cache
        if voice is not None:
            # The voice prompt ships its OWN negative prefix (neg_tts_lm) — use it rather than the
            # <|image_pad|> approximation, so positive and negative differ only by the voice/text
            # conditioning, which is exactly what CFG is supposed to contrast.
            p = llm_neg.seed_prompt(*_voice_kv(voice, "neg_tts_lm"))
            neg_hidden = voice["neg_tts_lm.last_hidden_state"]
            print(f"    cfg negative: voice prompt's own neg_tts_lm prefix ({p} positions)")
        else:
            neg_id = tok.convert_tokens_to_ids("<|image_pad|>")
            if not isinstance(neg_id, int) or neg_id < 0:
                print("  [cfg][warn] <|image_pad|> not in tokenizer — falling back to --neg-mode type")
                llm_neg = None
                neg_const = type_embed[0].astype(np.float32)
            else:
                neg_hidden = llm_neg.prefill(text_hidden([[neg_id]]) + type_embed[1])
                print(f"    cfg negative: parallel stream seeded with <|image_pad|> (id={neg_id})")
    else:
        neg_const = (type_embed[0] if args.neg_mode == "type"
                     else np.zeros(H, dtype=np.float32)).astype(np.float32)

    # --- interleaved generation: W text tokens -> S speech frames -> repeat ---------------------
    # Upstream TTS_TEXT_WINDOW_SIZE=5 / TTS_SPEECH_WINDOW_SIZE=6 — this is the model card's
    # "interleaved, windowed design". Prefilling all text at once (--no-interleave) is NOT what the
    # model was trained on: it never sees text arriving between speech frames.
    TW, SW = args.text_window, args.speech_window
    if args.no_interleave:
        TW, SW = ids.shape[1], args.max_frames
    print(f"    windows: {TW} text tokens : {SW} speech frames"
          f"{' (interleaving DISABLED)' if args.no_interleave else ''}")

    eos_prob = None if args.no_eos else load_eos_head(onnx_dir, H)

    # samples emitted per acoustic frame — `speech_tok_compress_ratio` in preprocessor_config.json.
    # Verified against the graph: a T-frame decode returns exactly T*3200 samples.
    SPF = 3200

    chunks, latents, f, ti, hidden, first = [], [], 0, 0, None, True
    if voice is not None:
        # Seed the POSITIVE stream with the voice prompt's KV prefix. first=False so the text is
        # appended via step() — prefill() would _reset() and throw the prompt away.
        p = llm.seed_prompt(*_voice_kv(voice, "tts_lm"))
        hidden = voice["tts_lm.last_hidden_state"]
        first = False
        print(f"    seeded TTS backbone with {p}-position voice prefix")
    done = False
    stop_reason = f"hit --max-frames ({args.max_frames})"
    while f < args.max_frames:
        if ti < ids.shape[1]:                                   # feed the next text window
            win = ids[:, ti:ti + TW]
            ti += win.shape[1]
            # Feed ONLY the new window: text_lm.onnx is KV-cached now, so it already holds the voice
            # prompt prefix plus every earlier window at their true rotary positions — exactly what
            # upstream's forward_lm does. (Before the KV export this had to re-encode the whole
            # prefix each window to fake the same context.)
            embeds = text_hidden(win) + type_embed[1]           # type 1 = TEXT
            hidden = llm.prefill(embeds) if first else llm.step(embeds)
            first = False
        elif first:                                             # no text at all — seed once
            hidden = llm.prefill(np.zeros((1, 1, H), np.float32) + type_embed[1])
            first = False

        for _ in range(SW):
            if f >= args.max_frames:
                break
            cond = hidden[0, -1, :]
            ncond = neg_hidden[0, -1, :] if llm_neg is not None else neg_const
            latent = sampler.sample(cond, ncond, cfg_scale=args.cfg_scale, n_frames=1, seed=f)
            latents.append(latent[0])
            if args.decode_mode == "stream":
                # TRUE streaming, and BIT-EXACT — not an approximation.
                #
                # The decoder is a causal conv stack, verified: decoding a PREFIX of n frames is
                # bit-identical to the first n frames of the full decode (max|diff| = 0.0). What it
                # cannot survive is losing its left context — upstream carries that in
                # VibeVoiceTokenizerStreamingCache; our exported graph is stateless, so we replay it.
                #
                # Measured receptive field: ~56 frames (7.5 s!). Sweeping left context against the
                # full decode: W=16 -> -31 dB, W=32 -> -44 dB, W=48 -> -52 dB, W=56 -> -115 dB,
                # W=64+ -> bit-identical. That length is exactly why naive per-frame decoding left an
                # audible seam every 3200 samples.
                #
                # Cost is BOUNDED (context+1 frames per step), not O(T^2) like a growing prefix.
                hist = np.stack(latents[-(args.decode_context + 1):], 0)[None]
                wav = np.asarray(C.acoustic_decode_to_wav(dec, hist, scale, bias)).ravel()
                chunks.append(wav[-SPF:])                       # emit only the newest frame
            acoustic_embed = np.asarray(conn.run(features=latent[None]), dtype=np.float32)
            step_in = acoustic_embed + type_embed[0]            # type 0 = SPEECH
            hidden = llm.step(step_in)
            if llm_neg is not None:
                neg_hidden = llm_neg.step(step_in)              # same frames, no text
            f += 1
            if f % 25 == 0:
                print(f"  streamed frame {f}/{args.max_frames}  (text {ti}/{ids.shape[1]})")

            # Learned stop, checked exactly where upstream checks it: on the POSITIVE stream's last
            # hidden state after the frame we just made was fed back in. The frame is kept.
            if eos_prob is not None and f >= args.eos_min_frames:
                p_stop = eos_prob(hidden[0, -1, :])
                if p_stop > args.eos_threshold:
                    stop_reason = (f"tts_eos_classifier fired at frame {f} "
                                   f"(p={p_stop:.3f} > {args.eos_threshold})")
                    done = True
                    break

        if done:
            break

    print(f"    generation stopped: {stop_reason}"
          f"  [{f} frames = {f / 7.5:.2f}s, text {ti}/{ids.shape[1]} tokens consumed]")
    if not done and eos_prob is not None and ti >= ids.shape[1]:
        print("    [warn] ran out of text WITHOUT the learned stop firing — the tail is drift; "
              "try --eos-threshold below 0.5")

    if args.save_latents and latents:
        np.save(args.save_latents, np.stack(latents, 0)[None].astype(np.float32))
        print(f"    saved latents {np.stack(latents,0)[None].shape} -> {args.save_latents}")

    if args.decode_mode == "batch" and latents:
        # ONE decode over the whole latent sequence: the acoustic decoder is a CAUSAL CONV stack, so
        # decoding [1, F, 64] in a single pass keeps its receptive field continuous across frames.
        # Upstream gets the same continuity in streaming mode via VibeVoiceTokenizerStreamingCache
        # (cache=acoustic_cache, use_cache=True); our exported decoder is stateless, so per-frame
        # decoding would re-zero-pad every 3200 samples and leave an audible seam per frame.
        seq = np.stack(latents, 0)[None]                        # [1, F, 64]
        print(f"    batch-decoding {seq.shape[1]} latents in one pass (continuous convolutions)")
        chunks = [np.asarray(C.acoustic_decode_to_wav(dec, seq, scale, bias)).ravel()]
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    C.save_wav(args.out, audio)
    print(f"wrote {args.out}  ({len(audio)} samples, {len(audio)/C.SR:.2f}s, {len(chunks)} frames)")


if __name__ == "__main__":
    main()
