"""Shared helpers for the jina-embeddings-v4 (vLLM merged, per-task) → ONNX sub-part scripts.

The model is decomposed into three shared sub-parts (cf. chandra: vision / embedding / decoder) so
the heavy backbone is stored ONCE and reused by both the text and image paths:

  vision.onnx       pixel_values (grid baked)            → image_features [N, 2048]
  embeddings.onnx   input_ids (+ image_features)         → inputs_embeds  [B, S, 2048]
  backbone.onnx     inputs_embeds, attention_mask,       → last_hidden    [B, S, 2048]
                    position_ids (MROPE, host-computed)
  pooling           (driver) masked mean + L2-norm       → embedding      [B, 2048]

Compose at inference (all ONNX; driver just wires sessions):
  text  : embeddings(ids)                    → backbone → mean-pool(attn_mask)   → embedding
  image : vision(px) → embeddings(ids, feats)→ backbone → mean-pool(vision-span) → embedding

CPU only (this env's torch/onnxruntime are CPU builds). The exported ONNX is execution-provider
agnostic — the same files run on CUDA later via onnxruntime-gpu, no rebuild and no device flag.
Stock Qwen2.5-VL, repo project env (transformers 5.x, torchvision for the image processor).

Entry points:  build.py  eval.py  inference.py  (this module is imported, not run).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

HIDDEN = 2048
MATRYOSHKA = [128, 256, 512, 1024, 2048]
IMAGE_PROMPT = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"


# --------------------------------------------------------------------------- misc
def hf_name(model_dir):
    """Model id read from the checkpoint (config `_name_or_path`, else README), not the folder."""
    d = Path(model_dir)
    try:
        nm = json.loads((d / "config.json").read_text(encoding="utf-8")).get("_name_or_path")
        if nm:
            return nm
    except Exception:
        pass
    r = d / "README.md"
    if r.exists():
        t = r.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"jina-embeddings-v4-vllm-[a-z0-9-]+", t)
        if m:
            return f"jinaai/{m.group(0)}"
    return str(d.name)


def quiet():
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as _tl
        _tl.set_verbosity_error()
    except Exception:
        pass


def make_session(path):
    """CPU ORT session with log level raised to ERROR — silences the harmless 'can't constant-fold
    Where node' optimization notice (the node still runs; parity is unaffected)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def load_model(model_dir, dtype=torch.float32, attn="eager"):
    """Stock Qwen2_5_VLForConditionalGeneration (task LoRA merged), on CPU. eager attn: the vision
    tower's SDPA sets enable_gqa=True which the legacy ONNX exporter rejects (pytorch/pytorch#162258)."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(Path(model_dir).resolve()), torch_dtype=dtype, attn_implementation=attn)
    return m.eval()


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path(model_dir).resolve()))


# --------------------------------------------------------------------------- tensors
def text_position_ids(attention_mask):
    """MROPE position_ids [3,B,S] for TEXT (no vision tokens): standard cumulative positions."""
    pos1d = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    return pos1d.unsqueeze(0).expand(3, -1, -1).contiguous()


def make_inputs(tokenizer, texts, prefix="Query"):
    enc = tokenizer([f"{prefix}: {t}" for t in texts], return_tensors="pt", padding="longest")
    return enc["input_ids"], enc["attention_mask"]


def make_image_inputs(model_dir, image, size):
    from transformers import AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(str(Path(model_dir).resolve()))
    if image is None:
        img = Image.fromarray((np.random.RandomState(0).rand(size, size, 3) * 255).astype("uint8"))
    else:
        img = Image.open(image).convert("RGB").resize((size, size))
    return proc(text=[IMAGE_PROMPT], images=[img], return_tensors="pt")


def image_rope_and_mask(model, batch):
    """Host MROPE position_ids [3,B,S] + vision-span pool mask [B,S] (<vision_start>..<vision_end>)."""
    cfg = model.config
    ids, am = batch["input_ids"], batch["attention_mask"]
    mm = torch.zeros_like(ids); mm[ids == cfg.image_token_id] = 1
    pos, _ = model.model.get_rope_index(input_ids=ids, mm_token_type_ids=mm,
                                        image_grid_thw=batch["image_grid_thw"], video_grid_thw=None,
                                        second_per_grid_ts=None, attention_mask=am)
    s = int((ids[0] == cfg.vision_start_token_id).nonzero()[0])
    e = int((ids[0] == cfg.vision_end_token_id).nonzero()[0])
    vm = torch.zeros_like(ids, dtype=torch.float32); vm[0, s:e + 1] = 1.0
    return pos, vm


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def mean_pool(hidden, mask):
    m = mask[..., None].astype(hidden.dtype) if isinstance(hidden, np.ndarray) else mask.unsqueeze(-1)
    pooled = (hidden * m).sum(1) / m.sum(1)
    if isinstance(pooled, np.ndarray):
        return pooled / (np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-12)
    return torch.nn.functional.normalize(pooled, dim=-1)


# --------------------------------------------------------------------------- sub-part modules
class VisionSub(torch.nn.Module):
    """pixel_values → image_features [N,2048]. grid_thw baked (fixed resolution)."""
    def __init__(self, model, grid_thw):
        super().__init__()
        self.visual = model.model.visual
        self.register_buffer("grid_thw", grid_thw)

    def forward(self, pixel_values):
        # the full model uses vision_outputs.pooler_output (merged [N,2048]) as the image embeds,
        # NOT last_hidden_state (pre-merge [patches,1280]) — see Qwen2_5_VLModel.get_image_features.
        o = self.visual(pixel_values, grid_thw=self.grid_thw)
        return o.pooler_output if hasattr(o, "pooler_output") else o


class EmbeddingsSub(torch.nn.Module):
    """input_ids, image_features → inputs_embeds [B,S,2048]: token embeds with image_features
    scattered into <image_pad> positions (empty image_features → text-only path)."""
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens
        self.image_token_id = model.config.image_token_id

    def forward(self, input_ids, image_features):
        emb = self.embed_tokens(input_ids)
        mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(emb)
        return emb.masked_scatter(mask, image_features.to(emb.dtype))


class BackboneSub(torch.nn.Module):
    """inputs_embeds, attention_mask, position_ids → last_hidden [B,S,2048]."""
    def __init__(self, model):
        super().__init__()
        self.lm = model.model.language_model

    def forward(self, inputs_embeds, attention_mask, position_ids):
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                      position_ids=position_ids, use_cache=False)
        return out.last_hidden_state


# --------------------------------------------------------------------------- compose (eval + inference)
def load_sessions(onnx_dir, need_vision):
    out = Path(onnx_dir)
    s = {"embeddings": make_session(out / "embeddings.onnx"),
         "backbone": make_session(out / "backbone.onnx")}
    if need_vision:
        s["vision"] = make_session(out / "vision.onnx")
    return s


def embed_text_onnx(sess, tok, text, prefix, npdt):
    ids, am = make_inputs(tok, [text], prefix=prefix)
    pos = text_position_ids(am)
    empty = np.zeros((0, HIDDEN), dtype=npdt)
    e = sess["embeddings"].run(None, {"input_ids": ids.numpy(), "image_features": empty})[0]
    h = sess["backbone"].run(None, {"inputs_embeds": e, "attention_mask": am.numpy(),
                                    "position_ids": pos.numpy()})[0]
    return mean_pool(h, am.numpy())


def embed_image_onnx(sess, proc_dir, image, size, npdt, meta):
    """No model load: processor gives pixel_values; the fixed prompt tensors come from image_meta."""
    batch = make_image_inputs(proc_dir, image, size)     # only pixel_values is used from here
    f = sess["vision"].run(None, {"pixel_values": batch["pixel_values"].numpy().astype(npdt)})[0]
    e = sess["embeddings"].run(None, {"input_ids": meta["input_ids"], "image_features": f})[0]
    h = sess["backbone"].run(None, {"inputs_embeds": e, "attention_mask": meta["attention_mask"],
                                    "position_ids": meta["position_ids"]})[0]
    return mean_pool(h, meta["vision_mask"])


def npdt_of(manifest):
    return np.float16 if manifest.get("precision") == "fp16" else np.float32


def describe_precision(manifest):
    """Human label for a build dir from its manifest: base precision + any quantized sub-parts."""
    base = manifest.get("precision", "fp32")
    q = manifest.get("quantized") or {}
    if q:
        return base + " (" + ", ".join(f"{k}:{v}" for k, v in q.items()) + ")"
    return base
