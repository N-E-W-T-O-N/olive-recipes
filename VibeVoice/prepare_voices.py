"""Convert VibeVoice-Realtime voice-prompt caches (.pt) into torch-free .npz.

Why this exists
---------------
VibeVoice-Realtime carries speaker identity ONLY as a KV-cache prefix. Upstream's
`process_input_with_cached_prompt` takes a `cached_prompt` dict and builds the prompt's token ids
as all `pad_id` — so 100% of the voice information lives in the cached hidden states / KV, and
there is no reference audio to encode at inference time.

That is also why the realtime checkpoint ships **no acoustic encoder** (0 encoder tensors vs 276
decoder): Microsoft distributes voice prompts "in an embedded format" to limit deepfake misuse.
You cannot make a new voice from a .wav with this checkpoint alone.

The .pt files live in microsoft/VibeVoice at demo/voices/streaming_model/*.pt (MIT).

What this script does
---------------------
Loads the .pt behind a RESTRICTED unpickler (see SAFETY below), converts bfloat16 -> float32
(numpy has no bfloat16), and writes a flat .npz the inference driver can load with numpy alone.

Output keys:
    tts_lm.k.{i} / tts_lm.v.{i}      i in [0, 20)   the TTS backbone prompt prefix
    neg_tts_lm.k.{i} / neg_tts_lm.v.{i}             the CFG negative prefix
    tts_lm.last_hidden_state, neg_tts_lm.last_hidden_state
    lm.last_hidden_state                            text-encoder prompt (see LIMITATION)

SAFETY
------
`torch.load(weights_only=True)` fails on these files because they contain
`transformers.modeling_outputs.BaseModelOutputWithPast`, which the safe unpickler refuses to
SETITEMS into. Rather than fall back to `weights_only=False` (arbitrary code execution), this
script statically verifies the pickle first and then loads behind an allowlist unpickler that
permits only the five tensor-container globals these files legitimately need. Anything else — os,
subprocess, builtins, codecs — raises. Run --audit to see the static scan on its own.

LIMITATION
----------
The `lm` (4-layer text encoder) prefix is exported for completeness but is NOT currently usable:
text_lm.onnx was exported stateless (`input_ids -> hidden`) with no past_key_values inputs, so its
prompt prefix cannot be injected. Only the `tts_lm` prefix is wired up, which is where the acoustic
/ speaker conditioning lives. Re-exporting text_lm with KV inputs would close the remaining gap.

Usage:
  uv run prepare_voices.py                 # convert every .pt in voices/streaming_model
  uv run prepare_voices.py --audit         # static pickle scan only, no loading
"""
import argparse
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
VOICE_DIR = HERE / "voices" / "streaming_model"

# The only globals these caches legitimately contain. Verified by --audit against the real files.
ALLOWED_GLOBALS = {
    ("transformers.modeling_outputs", "BaseModelOutputWithPast"),
    ("transformers.cache_utils", "DynamicCache"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "BFloat16Storage"),
    ("torch", "FloatStorage"),
    ("torch", "HalfStorage"),
    ("collections", "OrderedDict"),
}


def audit_pickle(pt_path: Path):
    """Statically list every GLOBAL the pickle references, WITHOUT executing it."""
    import pickletools

    with zipfile.ZipFile(pt_path) as z:
        name = next(n for n in z.namelist() if n.endswith("data.pkl"))
        buf = z.read(name)
    found = set()
    for op, arg, _ in pickletools.genops(io.BytesIO(buf)):
        if op.name == "GLOBAL":
            mod, _, cls = str(arg).partition(" ")
            found.add((mod, cls))
    unexpected = found - ALLOWED_GLOBALS
    return found, unexpected


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"blocked disallowed global {module}.{name} while loading a voice prompt")
        return super().find_class(module, name)


class _RestrictedPickleModule:
    Unpickler = _RestrictedUnpickler

    @staticmethod
    def load(f, **kw):
        return _RestrictedUnpickler(f, **kw).load()


def _kv_lists(cache):
    """DynamicCache -> (keys, values) lists, tolerant of the transformers layout change."""
    if hasattr(cache, "key_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    return [l.keys for l in cache.layers], [l.values for l in cache.layers]


def convert(pt_path: Path, out_path: Path):
    import torch

    found, unexpected = audit_pickle(pt_path)
    if unexpected:
        raise SystemExit(f"[{pt_path.name}] REFUSING: unexpected pickle globals {unexpected}")

    obj = torch.load(pt_path, map_location="cpu", weights_only=False,
                     pickle_module=_RestrictedPickleModule)

    def f32(t):
        return t.float().numpy().astype(np.float32)   # bfloat16 has no numpy dtype

    out = {}
    # All four streams: the 20-layer TTS backbone (tts_lm) AND the 4-layer text encoder (lm), plus
    # both CFG negatives. text_lm.onnx is now exported WITH past_key_values, so the `lm` prefix is
    # injectable too — that prefix exists only as KV, since upstream's prompt token ids are all
    # pad_id and cannot be replayed.
    for stream in ("tts_lm", "neg_tts_lm", "lm", "neg_lm"):
        keys, vals = _kv_lists(obj[stream]["past_key_values"])
        for i, (k, v) in enumerate(zip(keys, vals)):
            out[f"{stream}.k.{i}"] = f32(k)
            out[f"{stream}.v.{i}"] = f32(v)
        out[f"{stream}.last_hidden_state"] = f32(obj[stream]["last_hidden_state"])

    np.savez(out_path, **out)
    n_layers = sum(1 for k in out if k.startswith("tts_lm.k."))
    prefix = out["tts_lm.k.0"].shape[2]
    print(f"  {pt_path.name:22s} -> {out_path.name:22s} "
          f"{n_layers} layers, {prefix}-position prefix, {out_path.stat().st_size/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice-dir", default=str(VOICE_DIR))
    ap.add_argument("--audit", action="store_true", help="static pickle scan only; do not load")
    args = ap.parse_args()

    vd = Path(args.voice_dir)
    pts = sorted(vd.glob("*.pt"))
    if not pts:
        raise SystemExit(f"no .pt voice prompts in {vd}")

    if args.audit:
        for p in pts:
            found, unexpected = audit_pickle(p)
            status = "CLEAN" if not unexpected else f"UNEXPECTED {unexpected}"
            print(f"  {p.name:22s} {len(found)} globals  {status}")
            for g in sorted(found):
                print(f"      {g[0]}.{g[1]}")
        return

    print(f"converting {len(pts)} voice prompts in {vd}")
    for p in pts:
        convert(p, p.with_suffix(".npz"))


if __name__ == "__main__":
    main()
