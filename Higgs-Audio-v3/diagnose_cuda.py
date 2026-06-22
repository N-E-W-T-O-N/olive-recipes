# /// script
# requires-python = ">=3.9"
# dependencies = ["onnxruntime-gpu", "numpy"]
# ///
"""Diagnose the cuda_int4 silent-zeros bug on a target device (e.g. Jetson Orin / ARM64).

Runs each ONNX sub-part on BOTH the CUDA and CPU providers with identical random inputs
and compares the outputs. Pinpoints whether the failure is GroupQueryAttention (only in
llm_decoder) or MatMulNBits (also in audio_heads), and uses ORT profiling to show which
provider each op actually ran on (catches silent CPU fallbacks / dead CUDA kernels).

Run ON THE DEVICE (uses its onnxruntime-gpu build):
    python diagnose_cuda.py --model-path onnx/cuda_int4
    # or: python diagnose_cuda.py --model-path onnx/cpu_int4   (same int4 graph)

No model weights / text needed — random inputs are enough to expose all-zero output.
Paste the full output into the issue thread.
"""
import argparse, json, glob, os, collections
from pathlib import Path
import numpy as np
import onnxruntime as ort


def cosine(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n else float("nan")


def cast_feeds(sess, feeds):
    """Cast each feed to the dtype the graph expects (CUDA/fp16 decoders want float16;
    int4/cpu want float32). Without this, a float32 feed to an fp16 input errors with
    'Unexpected input data type ... expected float16'."""
    want = {i.name: i.type for i in sess.get_inputs()}
    out = {}
    for k, v in feeds.items():
        t = want.get(k, "")
        if "float16" in t:
            out[k] = np.asarray(v, np.float16)
        elif "float" in t:
            out[k] = np.asarray(v, np.float32)
        else:
            out[k] = v
    return out


def make_session(path, ep, profile=False):
    so = ort.SessionOptions()
    so.log_severity_level = 1 if profile else 3
    if profile:
        so.enable_profiling = True
    return ort.InferenceSession(str(path), so, providers=[ep])


def dummy_feeds(name, sm, rng):
    """Random inputs for each sub-part (shapes from the manifest)."""
    if name == "text_embed":
        return {"input_ids": rng.integers(0, 1000, (1, 8), dtype=np.int64)}
    if name == "audio_embed":
        return {"codes": rng.integers(0, 1024, (1, 8, 8), dtype=np.int64)}
    if name == "audio_heads":
        return {"hidden_states": rng.standard_normal((1, 8, 2560)).astype(np.float32)}
    if name == "audio_tokenizer":
        return {"audio_codes": rng.integers(0, 1024, (1, 8, 25), dtype=np.int64)}
    if name == "audio_encoder":
        return {"input_values": rng.standard_normal((1, 1, 24000)).astype(np.float32)}
    if name == "llm_decoder":
        meta = sm["llm_decoder"]
        L = meta.get("num_layers", 36); kv = meta.get("num_kv_heads", 8)
        hd = meta.get("head_dim", 128); H = meta.get("hidden_size", 2560)
        S = 8
        feeds = {"inputs_embeds": rng.standard_normal((1, S, H)).astype(np.float32),
                 "attention_mask": np.ones((1, S), dtype=np.int64)}
        z = np.zeros((1, kv, 0, hd), dtype=np.float32)
        for i in range(L):
            feeds[f"past_key_values.{i}.key"] = z
            feeds[f"past_key_values.{i}.value"] = z
        return feeds
    return None


def first_real_output(sess, outs):
    """Pick the main float output (hidden_states / logits / waveform / embeds)."""
    names = [o.name for o in sess.get_outputs()]
    for pref in ("hidden_states", "logits", "audio_logits", "inputs_embeds",
                 "audio_embeds", "waveform", "audio_codes"):
        if pref in names:
            return outs[names.index(pref)]
    return outs[0]


def parse_profile(sess, want_ops):
    """Read the profiling JSON → {op_type: Counter(provider)}."""
    pf = sess.end_profiling()
    placement = collections.defaultdict(collections.Counter)
    try:
        data = json.loads(Path(pf).read_text())
    except Exception:
        return placement, pf
    for e in data:
        if e.get("cat") != "Node":
            continue
        args = e.get("args", {})
        op = args.get("op_name") or args.get("op_type")
        prov = args.get("provider", "?")
        if op in want_ops:
            placement[op][prov] += 1
    return placement, pf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="onnx/{device}_{precision} dir")
    ap.add_argument("--cuda-ep", default="CUDAExecutionProvider")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.model_path)
    sm = json.loads((root / "manifest.json").read_text())["sub_models"]
    avail = ort.get_available_providers()

    print("=" * 70)
    print("onnxruntime", ort.__version__)
    try:
        print("build_info:", ort.get_build_info())
    except Exception as e:
        print("build_info: <unavailable>", e)
    print("available_providers:", avail)
    print("device:", os.uname() if hasattr(os, "uname") else os.name)
    print("=" * 70)

    if args.cuda_ep not in avail:
        print(f"\n[FATAL] {args.cuda_ep} not available — nothing to compare. "
              f"Install onnxruntime-gpu for this device."); return

    rng0 = np.random.default_rng(args.seed)
    components = [c for c in ("text_embed", "audio_embed", "audio_heads",
                              "audio_tokenizer", "audio_encoder", "llm_decoder") if c in sm]

    print(f"\n{'component':16} {'CPU |x|mean':>12} {'CUDA |x|mean':>13} {'cos(CUDA,CPU)':>14}  verdict")
    print("-" * 78)
    results = {}
    for name in components:
        path = root / sm[name]["filename"]
        feeds = dummy_feeds(name, sm, np.random.default_rng(args.seed))  # same per component
        try:
            cpu = make_session(path, "CPUExecutionProvider")
            cpu_out = first_real_output(cpu, cpu.run(None, cast_feeds(cpu, feeds)))
            gpu = make_session(path, args.cuda_ep)
            gpu_out = first_real_output(gpu, gpu.run(None, cast_feeds(gpu, feeds)))
        except Exception as e:
            print(f"{name:16} ERROR: {e}")
            continue
        cpu_m = float(np.abs(cpu_out).mean())
        gpu_m = float(np.abs(gpu_out).mean())
        cos = cosine(cpu_out, gpu_out)
        zero = gpu_m < 1e-8
        verdict = "ALL-ZERO on CUDA ✗" if zero else ("MISMATCH ✗" if cos < 0.99 else "ok ✓")
        results[name] = (cpu_m, gpu_m, cos, zero)
        print(f"{name:16} {cpu_m:12.4f} {gpu_m:13.4f} {cos:14.4f}  {verdict}")

    # op placement for llm_decoder (which ran on CUDA vs fell back to CPU)
    if "llm_decoder" in sm:
        print("\n--- llm_decoder op placement (profiling) ---")
        try:
            sess = make_session(root / sm["llm_decoder"]["filename"], args.cuda_ep, profile=True)
            sess.run(None, cast_feeds(sess, dummy_feeds("llm_decoder", sm, np.random.default_rng(args.seed))))
            placement, pf = parse_profile(sess, {"MatMulNBits", "GroupQueryAttention"})
            for op in ("GroupQueryAttention", "MatMulNBits"):
                print(f"  {op:22}: {dict(placement.get(op, {})) or '<not in profile>'}")
            print(f"  (profile: {pf})")
        except Exception as e:
            print("  profiling failed:", e)

    # verdict
    print("\n" + "=" * 70)
    dec = results.get("llm_decoder"); heads = results.get("audio_heads")
    if dec and dec[3]:
        if heads and not heads[3] and heads[2] > 0.99:
            print("VERDICT: llm_decoder is ALL-ZERO on CUDA, but audio_heads (MatMulNBits) is")
            print("  CORRECT on CUDA → MatMulNBits works; GroupQueryAttention is the broken op.")
            print("  → fp16/int4 won't help (both use GQA). Fixes: CPU EP, source-build ORT with")
            print("    CUDA contrib kernels for this SM, or re-export the decoder GQA-free (eager attn).")
        else:
            print("VERDICT: llm_decoder ALL-ZERO on CUDA; audio_heads also off → MatMulNBits (and")
            print("  possibly GQA) broken on this runtime. Try an fp16 build (drops MatMulNBits)")
            print("  and re-test; if still zero, GQA is also implicated.")
    elif dec:
        print("VERDICT: llm_decoder looks OK on CUDA here — could not reproduce the all-zero bug.")
    print("=" * 70)


if __name__ == "__main__":
    main()
