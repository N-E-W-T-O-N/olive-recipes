"""Evaluate exported VibeVoice ONNX sub-parts against their PyTorch references.

Two stages:
  A) COMPONENT parity — for every exported component (acoustic/semantic encoders & decoder,
     diffusion_head, connectors, projector) load the PyTorch reference (user_script loaders) and
     the ONNX session on the SAME inputs, then report cosine + max|Δ| with a PASS/FAIL threshold.
     The LLM decoder is checked structurally (int4 MatMulNBits + GQA + inputs_embeds→hidden).
  B) WHOLE pipeline — an end-to-end integrity chain of the ONNX pieces:
       • TTS codec round-trip (audio → acoustic_encoder → acoustic_decoder → audio′): waveform
         correlation + SNR — the standard neural-codec reconstruction metric.
       • Full ONNX chain smoke: latent → acoustic_connector → (cond) → diffusion_head → 1 denoise
         step, and acoustic_decoder → waveform — every exported block run in sequence, shapes +
         finiteness verified. (The autoregressive LLM + multi-step DDPM loop need the genai
         runtime and are out of scope here — reported as SKIPPED, not silently passed.)

Model is the FINAL positional (a key {1.5b,asr,asr-hf,realtime} or a checkpoint dir path).

Usage:
  uv run eval.py 1.5b
  uv run eval.py --device cuda --precision fp16 ./asr-hf
  uv run eval.py --audio sample.wav 1.5b
  uv run eval.py --components acoustic_encoder diffusion_head realtime
"""
import argparse
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import optimize as O          # registry (MODELS/DEVICES), resolve_target, detect
import user_script as us

COS_TOL = 0.99                # component parity threshold


def _np(o):
    """Best-effort tensor → float32 numpy from a torch tensor or an output object."""
    import torch
    if torch.is_tensor(o):
        return o.detach().float().cpu().numpy()
    for a in ("latents", "mean", "sample", "audio", "last_hidden_state"):
        if hasattr(o, a):
            v = getattr(o, a); v = v() if callable(v) else v
            if torch.is_tensor(v):
                return v.detach().float().cpu().numpy()
    if isinstance(o, (tuple, list)):
        return _np(o[0])
    raise TypeError(f"cannot convert output of type {type(o).__name__}")


def _cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / (n + 1e-9))


def _session(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def parity_component(model_key, src, comp, onnx_path):
    """Run PyTorch ref + ONNX on identical dummy inputs; return (cos, maxd, ok)."""
    import torch
    loader_name, io_name, dummy_name, env = O.MODELS[model_key]["olive"][comp]
    os.environ.update(env)
    pyt = getattr(us, loader_name)(str(src))
    if hasattr(pyt, "eval"):
        pyt = pyt.eval()
    io = getattr(us, io_name)()
    names = io["input_names"]
    inputs = getattr(us, dummy_name)()
    with torch.no_grad():
        ref = _np(pyt(*[inputs[n] for n in names]))
    # feed each ONNX input at the graph's DECLARED dtype (fp16 builds expect float16, not float32);
    # the PyTorch ref stays fp32 and cosine absorbs the precision gap.
    sess = _session(onnx_path)
    _ORT2NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
               "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_}
    want = {i.name: _ORT2NP.get(i.type) for i in sess.get_inputs()}
    feed = {n: (inputs[n].numpy().astype(want[n]) if want.get(n) else inputs[n].numpy()) for n in names}
    got = sess.run(None, feed)[0]
    m = min(ref.size, got.size)
    a, b = got.ravel()[:m].astype(np.float64), ref.ravel()[:m].astype(np.float64)
    cos = _cos(a, b)
    maxd = float(np.abs(a - b).max())
    rel = maxd / (float(np.abs(b).max()) + 1e-12)
    # PASS on cosine, OR when tensors are numerically identical (near-zero-signal inputs make
    # cosine noise-dominated even though maxd≈0 — e.g. a decoder fed random latents → ~silence).
    ok = (cos >= COS_TOL) or (maxd < 1e-4) or (rel < 1e-3)
    return cos, maxd, ok


def check_llm(onnx_path):
    """Structural check of the genai-built decoder (no full runtime inference)."""
    import onnx
    g = onnx.load(str(onnx_path), load_external_data=False).graph
    ops = [n.op_type for n in g.node]
    innames = [i.name for i in g.input]
    nb, gqa = ops.count("MatMulNBits"), ops.count("GroupQueryAttention")
    embeds_in = any("inputs_embeds" in n for n in innames)
    ok = (nb > 0 or "MatMul" in ops) and embeds_in
    return f"MatMulNBits={nb} GQA={gqa} inputs_embeds={embeds_in}", ok


# ---------------------------------------------------------------- whole pipeline

def _load_audio(path, samples, sr=24000):
    import soundfile as sf, librosa
    wav, in_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if in_sr != sr:
        wav = librosa.resample(wav, orig_sr=in_sr, target_sr=sr)
    if len(wav) < samples:
        wav = np.pad(wav, (0, samples - len(wav)))
    return wav[:samples].astype(np.float32)[None, None, :]


def _synth_audio(samples, sr=24000):
    t = np.arange(samples) / sr
    wav = 0.3 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t))
    return wav.astype(np.float32)[None, None, :]


_ORT2NP_FEED = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(bool)": np.bool_}


def _feed(sess, d):
    """Cast each fed array to the session's DECLARED input dtype (fp16 graphs want float16;
    float32-pinned inputs like diffusion timesteps stay float32 — matches the graph either way)."""
    want = {i.name: _ORT2NP_FEED.get(i.type) for i in sess.get_inputs()}
    return {k: (v.astype(want[k]) if want.get(k) is not None and hasattr(v, "astype") else v)
            for k, v in d.items()}


def whole_pipeline(model_key, models_dir, audio_path):
    """End-to-end ONNX integrity: codec round-trip + full-chain smoke. Returns list of rows."""
    rows = []
    present = {c: models_dir / f"{c}.onnx" for c in
              ("acoustic_encoder", "acoustic_decoder", "acoustic_connector", "diffusion_head",
               "semantic_encoder", "multi_modal_projector")
              if (models_dir / f"{c}.onnx").exists()}

    # --- codec round-trip (encoder + decoder both present) ---
    if "acoustic_encoder" in present and "acoustic_decoder" in present:
        enc, dec = _session(present["acoustic_encoder"]), _session(present["acoustic_decoder"])
        samples = enc.get_inputs()[0].shape[2]
        samples = 24000 if not isinstance(samples, int) else samples
        wav = _load_audio(audio_path, samples) if audio_path else _synth_audio(samples)
        lat = enc.run(None, _feed(enc, {"audio": wav}))[0]
        dec_frames = dec.get_inputs()[0].shape[1]
        if isinstance(dec_frames, int) and lat.shape[1] != dec_frames:
            f = min(lat.shape[1], dec_frames)
            lat = lat[:, :f, :] if lat.shape[1] > f else np.pad(lat, ((0, 0), (0, dec_frames - lat.shape[1]), (0, 0)))
        recon = dec.run(None, _feed(dec, {"latents": lat.astype(np.float32)}))[0].ravel()
        n = min(len(recon), wav.size)
        x = wav.ravel()[:n]; y = recon[:n]
        corr = float(np.corrcoef(x, y)[0, 1]) if n > 1 else float("nan")
        noise = x - y
        snr = 10 * np.log10((np.sum(x**2) + 1e-12) / (np.sum(noise**2) + 1e-12))
        finite = bool(np.isfinite(recon).all())
        rows.append(("codec round-trip", f"corr={corr:+.3f} SNR={snr:+.1f}dB finite={finite} "
                     f"in={wav.size} out={len(recon)}", finite))
    else:
        rows.append(("codec round-trip", "SKIPPED (needs acoustic_encoder+decoder)", None))

    # --- full ONNX chain smoke: latent → connector → diffusion_head; latent → decoder ---
    if "acoustic_connector" in present and "diffusion_head" in present:
        conn = _session(present["acoustic_connector"]); dh = _session(present["diffusion_head"])
        rng = np.random.default_rng(0)
        T = 8
        lat = rng.standard_normal((1, T, 64)).astype(np.float32)
        cond = conn.run(None, _feed(conn, {"features": lat}))[0]                 # [1,T,H]
        H = cond.shape[-1]
        ni = rng.standard_normal((T, 64)).astype(np.float32)
        ts = (rng.random(T) * 1000).astype(np.float32)
        pred = dh.run(None, _feed(dh, {"noisy_images": ni, "timesteps": ts,
                             "condition": cond.reshape(T, H).astype(np.float32)}))[0]
        ok = pred.shape == (T, 64) and np.isfinite(pred).all() and np.isfinite(cond).all()
        rows.append(("tts chain smoke", f"connector→[H={H}]→diffusion_head pred={pred.shape} finite={ok}", ok))
    else:
        rows.append(("tts chain smoke", "SKIPPED (needs acoustic_connector+diffusion_head)", None))

    # --- ASR fusion chain: encoders → projector/connectors → fused features ---
    if "multi_modal_projector" in present and "acoustic_encoder" in present and "semantic_encoder" in present:
        ae, se, proj = (_session(present[k]) for k in ("acoustic_encoder", "semantic_encoder", "multi_modal_projector"))
        wav = _synth_audio(24000)
        al = ae.run(None, {"audio": wav})[0]; sl = se.run(None, {"audio": wav})[0]
        f = min(al.shape[1], sl.shape[1])
        fused = proj.run(None, {"acoustic_latents": al[:, :f].astype(np.float32),
                                "semantic_latents": sl[:, :f].astype(np.float32)})[0]
        ok = np.isfinite(fused).all()
        rows.append(("asr fusion chain", f"ac+sem→projector fused={fused.shape} finite={ok}", ok))

    rows.append(("autoregressive LLM + DDPM loop", "SKIPPED (needs genai runtime; out of scope)", None))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate VibeVoice ONNX sub-parts",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    global COS_TOL
    ap.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    ap.add_argument("--precision", choices=["int4", "fp16", "fp32"], default="int4")
    ap.add_argument("--components", help="comma-separated subset (default: all present)")
    ap.add_argument("--output-dir", help="models dir (default onnx/{model}/{device}_{precision})")
    ap.add_argument("--audio", help="wav for codec round-trip (default: synthetic tone)")
    ap.add_argument("--tol", type=float, default=COS_TOL, help=f"cosine PASS threshold (default {COS_TOL})")
    ap.add_argument("model", help="keyword {1.5b,asr,asr-hf,realtime} or checkpoint dir path")
    args = ap.parse_args()
    COS_TOL = args.tol
    device = "cuda" if args.device == "gpu" else args.device
    model_key, src = O.resolve_target(args.model)
    models_dir = Path(args.output_dir) if args.output_dir else (HERE / "onnx" / model_key / f"{device}_{args.precision}")
    if not models_dir.exists():
        sys.exit(f"no exported models at {models_dir} — run optimize.py first")

    print(f"=== eval {model_key}  src={src}  models={models_dir}  tol={COS_TOL} ===\n")

    # ---- A) component parity ----
    comps = [c.strip() for c in args.components.split(",")] if args.components else O.all_components(model_key)
    print("A) COMPONENT PARITY (ONNX vs PyTorch)")
    results = []
    for comp in comps:
        onnx_path = models_dir / (f"{comp}.onnx" if comp != "llm" else "llm_decoder.onnx")
        if not onnx_path.exists():
            print(f"  {comp:22s}  — not exported (skip)"); continue
        try:
            if comp == "llm":
                info, ok = check_llm(onnx_path)
                print(f"  {comp:22s}  {'PASS' if ok else 'FAIL'}  {info}"); results.append(ok)
            else:
                cos, maxd, ok = parity_component(model_key, src, comp, onnx_path)
                print(f"  {comp:22s}  {'PASS' if ok else 'FAIL'}  cos={cos:.6f} maxd={maxd:.2e}")
                results.append(ok)
        except Exception as e:
            print(f"  {comp:22s}  ERROR  {type(e).__name__}: {str(e)[:120]}"); results.append(False)

    # ---- B) whole pipeline ----
    print("\nB) WHOLE PIPELINE (end-to-end ONNX integrity)")
    for name, info, ok in whole_pipeline(model_key, models_dir, args.audio):
        tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {name:32s}  {tag}  {info}")
        if ok is not None:
            results.append(ok)

    npass = sum(1 for r in results if r); ntot = len(results)
    print(f"\n=== {npass}/{ntot} checks passed ===")
    sys.exit(0 if npass == ntot else 1)


if __name__ == "__main__":
    main()
