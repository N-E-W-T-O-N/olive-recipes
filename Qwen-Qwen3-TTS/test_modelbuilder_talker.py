# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers==4.57.3",
#   "torch",
#   "numpy",
#   "safetensors",
#   "huggingface_hub",
#   "accelerate",
#   "librosa",
#   "soundfile",
# ]
# ///
"""Empirical proof: would `create_model` (standard RoPE) match the real MROPE talker?

onnxruntime-genai ModelBuilder has NO MROPE support (0 refs in builder.py) — it only
emits standard RoPE. The Qwen3-TTS talker uses MROPE (mrope_section=[24,20,20]), giving
text vs audio-timed tokens different position semantics.

This test runs the SAME talker on the SAME inputs_embeds twice:
  (A) MROPE  — distinct per-axis position_ids (what the real pipeline feeds)
  (B) RoPE   — all 3 mrope axes set equal  ==  what create_model would compute

If (A) and (B) diverge, a create_model export (which can only produce (B)) is
semantically wrong for this model. Run:  uv run test_modelbuilder_talker.py
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import torch

from user_script import get_talker_model

torch.manual_seed(0)
talker = get_talker_model("voicedesign")

B, T, H = 1, 48, 2048
inputs_embeds = torch.randn(B, T, H)
attention_mask = torch.ones(B, T, dtype=torch.long)

# (A) MROPE: distinct position components — emulates interleaved text + audio-timed tokens.
# axis 0 = global/sequence, axis 1 & 2 = audio time grids (Qwen3-TTS uses 13 pos/sec).
seq = torch.arange(T)
pos_mrope = torch.stack([seq, seq // 2, seq // 3])[:, None, :]          # [3, B, T] distinct
# (B) standard RoPE: all three axes identical → exactly what create_model emits
pos_rope = seq[None, None, :].expand(3, B, T).contiguous()             # [3, B, T] all-equal

with torch.no_grad():
    logits_mrope = talker(inputs_embeds, pos_mrope, attention_mask).numpy()
    logits_rope = talker(inputs_embeds, pos_rope, attention_mask).numpy()


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


tok_mrope = logits_mrope[0].argmax(-1)
tok_rope = logits_rope[0].argmax(-1)
agree = float((tok_mrope == tok_rope).mean())

print("=" * 68)
print("Talker: MROPE (correct) vs standard-RoPE collapse (== create_model)")
print("=" * 68)
print(f"logits shape         : {logits_mrope.shape}")
print(f"cosine(MROPE, RoPE)  : {cos(logits_mrope, logits_rope):.4f}")
print(f"max |Δlogit|         : {np.abs(logits_mrope - logits_rope).max():.3f}")
print(f"argmax token agree   : {agree:.1%}  ({int(agree*T)}/{T} positions)")
print()
if agree < 0.99:
    print("RESULT: MROPE and standard-RoPE outputs DIVERGE → a create_model export")
    print("        (standard RoPE only) would produce DIFFERENT, wrong codes for the")
    print("        interleaved text/audio sequence. create_model is NOT viable here;")
    print("        Olive (real MROPE forward) is required.")
else:
    print("RESULT: near-identical — MROPE collapses to standard RoPE for these positions.")
