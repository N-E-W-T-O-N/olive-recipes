"""VibeVoice ASR inference (audio → text) over the exported ONNX sub-parts.

Handles BOTH ASR checkpoints (choose via the positional model — key or path):
  asr     (VibeVoice-ASR, vibevoice/ family)  : acoustic_connector + semantic_connector, fused = sum
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

# Speech markers. Both ASR checkpoints REUSE existing Qwen2.5 grounding tokens (object_ref_start/end
# + box_start) rather than adding literal <|speech_*|> tokens — the audio features are injected at the
# `pad` positions.
#
# These are the FALLBACK defaults only. The authoritative source is the checkpoint's
# processor_config.json (audio_bos_token / audio_token / audio_eos_token), which
# `speech_markers()` reads — see the note there. Hardcoding `<|speech_*|>` for asr-hf was a real bug:
# those tokens do not exist in the released VibeVoice-ASR-HF vocab, so convert_tokens_to_ids()
# returned None, ZERO pad slots were found, no audio features were ever injected, and the model
# transcribed the literal prompt text ("This is a speech pad. This is a speech pad. …").
SYSTEM_PROMPT = "You are a helpful assistant that transcribes audio input into text output in JSON format."
SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]
SPEECH_MARKERS = {
    "asr":    {"start": "<|object_ref_start|>", "pad": "<|box_start|>", "end": "<|object_ref_end|>"},
    "asr-hf": {"start": "<|object_ref_start|>", "pad": "<|box_start|>", "end": "<|object_ref_end|>"},
}


def speech_markers(key, src, tok):
    """Resolve the (start, pad, end) speech markers for this checkpoint.

    Prefers the checkpoint's own processor_config.json — VibeVoiceAsrProcessor stores them as
    audio_bos_token / audio_token / audio_eos_token, and config.json carries the matching
    *_token_id values. Falls back to SPEECH_MARKERS[key] when the file is absent (e.g. the
    `asr` family, whose processor lives in the vendored source).

    Every resolved marker is verified against the tokenizer; an unknown token is fatal, because
    silently getting 0 pad slots produces confident-looking garbage rather than an error.
    """
    mk = dict(SPEECH_MARKERS.get(key, SPEECH_MARKERS["asr"]))
    pc = Path(src) / "processor_config.json"
    if pc.exists():
        d = json.loads(pc.read_text(encoding="utf-8"))
        got = {"start": d.get("audio_bos_token"), "pad": d.get("audio_token"),
               "end": d.get("audio_eos_token")}
        if all(got.values()):
            if got != mk:
                print(f"  [markers] from processor_config.json: {got}  (default was {mk})")
            mk = got
    bad = {k: v for k, v in mk.items() if tok.convert_tokens_to_ids(v) in (None, tok.unk_token_id)}
    if bad:
        sys.exit(f"[error] speech marker(s) {bad} are not in the tokenizer of this build.\n"
                 f"  Without a resolvable pad token no audio features can be injected and the "
                 f"transcript would be garbage.\n"
                 f"  Check {pc} against the tokenizer shipped in the built ONNX dir.")
    return mk


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
    mk = speech_markers(key, src, tok)
    pad_id = tok.convert_tokens_to_ids(mk["pad"])

    # Reconstruct the VibeVoiceASRProcessor prompt EXACTLY:
    #   system(apply_chat_template) + user(apply_chat_template of  start + pad*n + end + "\n" + suffix)
    # with add_generation_prompt so the assistant turn is primed. Speech features are injected at the
    # pad slots. (A plain "please transcribe" prompt without the chat template → degenerate output.)
    dur = len(wav) / C.SR
    suffix = (f"This is a {dur:.2f} seconds audio, please transcribe it with these keys: "
              + ", ".join(SHOW_KEYS))
    speech_ph = mk["start"] + mk["pad"] * n + mk["end"]
    # Render the whole chat to TEXT, then encode ourselves: apply_chat_template(tokenize=True) does
    # not preserve the reused special-token strings as single IDs, but tok.encode() does (verified).
    text = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": speech_ph + "\n" + suffix}],
        tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)

    pad_pos = [i for i, t in enumerate(ids) if t == pad_id]
    if not pad_pos:
        # Hard-fail: with no slots the audio is never seen by the LLM and it just re-reads the
        # prompt text, yielding a fluent but entirely fabricated transcript. That looks like a
        # model-quality problem and wastes a lot of time — make it an error, not a warning.
        sys.exit(f"[error] 0 speech slots found for pad token {mk['pad']!r} (id={pad_id}) in a "
                 f"{len(ids)}-token prompt — audio features cannot be injected.\n"
                 f"  The chat template likely dropped/re-split the marker. Compare the markers in "
                 f"{Path(src) / 'processor_config.json'} with the built tokenizer.")
    if len(pad_pos) != n:
        print(f"  [warn] pad slots {len(pad_pos)} != {n} frames — injecting min()")
    embeds = C.embed_tokens(src, ids, key, onnx_dir=onnx_dir)               # [1,S,H]
    m = min(len(pad_pos), n)
    embeds[0, pad_pos[:m], :] = feats[:m]                   # inject fused (acoustic+semantic) features
    print(f"  prompt {len(ids)} tokens ({len(pad_pos)} speech slots), decoding…")

    llm = C.OnnxLLM(llm_path, device)
    logits = llm.prefill(embeds)                            # [1,S,V]
    eos_ids = {i for i in (getattr(tok, "eos_token_id", None),
                           tok.convert_tokens_to_ids("<|im_end|>"),
                           tok.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0}
    out = []
    for _ in range(args.max_new_tokens):
        nxt = int(np.asarray(logits)[0, -1].argmax())
        if nxt in eos_ids:
            break
        out.append(nxt)
        logits = llm.step(C.embed_tokens(src, [nxt], key, onnx_dir=onnx_dir))
    text = tok.decode(out, skip_special_tokens=True)
    print(f"\nTRANSCRIPT:\n{text}")


if __name__ == "__main__":
    main()
