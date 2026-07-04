"""Shared building blocks for VibeVoice ONNX inference (TTS / ASR / Realtime).

The three drivers (inference.py, inference_asr.py, inference_realtime.py) all lean on the
same primitives, factored here:

  * resolve()            — model key + checkpoint src + ONNX dir (reuses optimize.py's registry)
  * OnnxLLM              — drives the genai-exported llm_decoder.onnx directly via onnxruntime,
                           managing the 28×(k,v) KV cache + growing attention_mask. Works for
                           BOTH the TTS backbone (outputs hidden_states) and — with lm_head kept —
                           the ASR decoder. One prefill(embeds)->hidden + step(embeds)->hidden loop.
  * OnnxOp               — thin wrapper over a single-file ONNX session (encoders/decoder/
                           connector/diffusion_head/projector).
  * DiffusionSampler     — the DDPM/DPM denoise loop (diffusion_head.onnx + codes/ scheduler + CFG).
  * audio load / normalize (-25 dBFS) / save  — 24 kHz mono (VibeVoice standard).
  * load_tokenizer, load_scaling — Qwen2.5 tokenizer + the stored speech scaling/bias factors.

These wrap the ALREADY-verified ONNX sub-parts (see eval.py); they don't re-load PyTorch.
Bit-exact TTS text prompts require the original VibeVoice processor/tokenizer assets (the repos
ship none — we fetch the plain Qwen2.5 tokenizer); where that matters it's called out in the driver.
"""
import glob
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import optimize as O

SR = 24000                     # VibeVoice sample rate
TARGET_DB_FS = -25.0
VAE_DIM = 64


# --------------------------------------------------------------------------- resolve / io
def resolve(model_arg, device="cpu", precision="int4", onnx_dir=None):
    key, src = O.resolve_target(model_arg)
    od = Path(onnx_dir) if onnx_dir else (src / f"{device}_{precision}" / "models")
    return key, src, od


# Friendly model ids for display.
MODEL_IDS = {
    "1.5b": "microsoft/VibeVoice-1.5B",
    "asr": "microsoft/VibeVoice-ASR",
    "asr-hf": "microsoft/VibeVoice-ASR-HF",
    "realtime": "microsoft/VibeVoice-Realtime-0.5B",
}


def resolve_from_path(model_path):
    """Inference input: a built ONNX dir in the new layout onnx/{key}/{device}_{precision}
    (e.g. onnx/asr/cuda_fp16). Also accepts the legacy <checkpoint>/{device}_{precision}[/models].

    Returns (key, src_checkpoint, onnx_dir, device, precision, model_id). SystemExit on bad path.
    """
    p = Path(model_path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"path does not exist: {p} (build it: uv run optimize.py <model>)")
    if list(p.glob("*.onnx")):
        onnx_dir = p
    elif (p / "models").is_dir() and list((p / "models").glob("*.onnx")):
        onnx_dir = p / "models"
    else:
        raise SystemExit(f"no .onnx files under {p} — pass a built dir, "
                         f"e.g. onnx/asr/cuda_fp16 (run: uv run optimize.py <model>)")
    dp = onnx_dir.parent if onnx_dir.name == "models" else onnx_dir
    device, sep, precision = dp.name.partition("_")
    if not sep or device not in ("cpu", "cuda"):
        raise SystemExit(f"cannot parse device/precision from '{dp.name}' — expected "
                         f"'<device>_<precision>' (e.g. cpu_int4, cuda_fp16)")
    if dp.parent.name in O.MODELS:                          # new layout: onnx/{key}/{dev}_{prec}
        key = dp.parent.name
        src = (HERE / O.MODELS[key]["dir"]).resolve()
    elif (dp.parent / "config.json").exists():             # legacy: <checkpoint>/{dev}_{prec}[/models]
        src = dp.parent
        key = O.detect_model_type(src)
    else:
        raise SystemExit(f"cannot locate the checkpoint for {dp} — expected an onnx/<key>/... layout "
                         f"or a config.json in {dp.parent}")
    if not (src / "config.json").exists():
        raise SystemExit(f"checkpoint dir missing config.json: {src}")
    return key, src, onnx_dir, device, precision, MODEL_IDS.get(key, key)


def _ort():
    import onnxruntime as ort
    return ort


def _providers(device):
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


class OnnxOp:
    """Single-input/-output-ish ONNX op. run(**named_np) -> first output array."""
    def __init__(self, path, device="cpu"):
        self.sess = _ort().InferenceSession(str(path), providers=_providers(device))
        self.inames = [i.name for i in self.sess.get_inputs()]

    def run(self, **feed):
        feed = {k: np.asarray(v, dtype=np.float32) if v.dtype != np.int64 else v for k, v in feed.items()}
        return self.sess.run(None, {k: feed[k] for k in self.inames})[0]


# --------------------------------------------------------------------------- audio
def load_audio(path, sr=SR):
    import soundfile as sf
    wav, in_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if in_sr != sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=in_sr, target_sr=sr)
    return wav.astype(np.float32)


def normalize_audio(wav, target_db_fs=TARGET_DB_FS, eps=1e-6):
    rms = np.sqrt(np.mean(wav ** 2)) + eps
    wav = wav * (10 ** (target_db_fs / 20) / rms)
    peak = np.abs(wav).max()
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32)


def save_wav(path, wav, sr=SR):
    import soundfile as sf
    wav = np.asarray(wav, dtype=np.float32).ravel()
    sf.write(str(path), wav, sr)
    return path


# --------------------------------------------------------------------------- tokenizer / scaling
def load_tokenizer(onnx_dir, src):
    """Qwen2.5 tokenizer saved next to the ONNX (by ModelBuilder) or the standalone dir."""
    from transformers import AutoTokenizer
    for cand in (onnx_dir, onnx_dir.parent.parent, src):
        if (Path(cand) / "tokenizer.json").exists() or (Path(cand) / "tokenizer_config.json").exists():
            return AutoTokenizer.from_pretrained(str(cand))
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


def load_scaling(src):
    """model.speech_scaling_factor / model.speech_bias_factor (stored scalars) or (1.0, 0.0)."""
    from safetensors.torch import load_file
    scale, bias = 1.0, 0.0
    for sf in glob.glob(str(Path(src) / "*.safetensors")):
        d = load_file(sf)
        for k, v in d.items():
            if k.endswith("speech_scaling_factor"):
                scale = float(v.reshape(-1)[0])
            elif k.endswith("speech_bias_factor"):
                bias = float(v.reshape(-1)[0])
    return scale, bias


# --------------------------------------------------------------------------- codes/ scheduler
def make_scheduler(diffusion_cfg):
    """Instantiate codes/ DPMSolverMultistepScheduler from a diffusion_head_config dict."""
    for name, sub in [("vibevoice", ""), ("vibevoice.schedule", "schedule")]:
        m = types.ModuleType(name); m.__path__ = [str(HERE / "codes" / "vibevoice" / sub)]
        sys.modules.setdefault(name, m)
    import importlib
    dpm = importlib.import_module("vibevoice.schedule.dpm_solver")
    beta = diffusion_cfg.get("ddpm_beta_schedule", "cosine")
    return dpm.DPMSolverMultistepScheduler(
        num_train_timesteps=diffusion_cfg.get("ddpm_num_steps", 1000),
        beta_schedule=beta,
        prediction_type=diffusion_cfg.get("prediction_type", "v_prediction"),
        algorithm_type="dpmsolver++",
    )


class DiffusionSampler:
    """DDPM/DPM sampling of one acoustic latent per frame via diffusion_head.onnx + CFG.

    condition/neg_condition: [H] float32 (LLM hidden state / negative). Returns latent [B,64]."""
    def __init__(self, head_op: OnnxOp, diffusion_cfg, device="cpu"):
        self.head = head_op
        self.cfg = diffusion_cfg
        self.steps = int(diffusion_cfg.get("ddpm_num_inference_steps", 20))

    def sample(self, condition, neg_condition=None, cfg_scale=1.3, n_frames=1, seed=0):
        import torch
        sched = make_scheduler(self.cfg)
        sched.set_timesteps(self.steps)
        rng = np.random.default_rng(seed)
        cond = np.asarray(condition, dtype=np.float32).reshape(1, -1).repeat(n_frames, 0)
        use_cfg = neg_condition is not None and cfg_scale != 1.0
        if use_cfg:
            neg = np.asarray(neg_condition, dtype=np.float32).reshape(1, -1).repeat(n_frames, 0)
            cond_all = np.concatenate([cond, neg], 0)      # [2n, H]: cond half, uncond half
        else:
            cond_all = cond
        # ONE sample of n frames. CFG runs the SAME sample through both cond and uncond branches
        # (duplicate for the head call only) — the reference does this each step; denoising must not
        # mix guidance with a noise difference between two independent samples.
        x = rng.standard_normal((n_frames, VAE_DIM)).astype(np.float32)
        for t in sched.timesteps:
            xin = np.concatenate([x, x], 0) if use_cfg else x
            tf = np.full((xin.shape[0],), float(t), dtype=np.float32)
            eps = self.head.run(noisy_images=xin, timesteps=tf, condition=cond_all)
            if use_cfg:
                c, u = np.split(eps, 2, 0)
                eps = u + cfg_scale * (c - u)              # [n, 64]
            out = sched.step(torch.from_numpy(eps), t, torch.from_numpy(x))
            x = out.prev_sample.numpy().astype(np.float32)
        return x


# --------------------------------------------------------------------------- LLM (raw ORT KV cache)
class OnnxLLM:
    """Drives a genai-exported decoder (llm_decoder.onnx) directly via onnxruntime.

    Inputs:  inputs_embeds[B,S,H], attention_mask[B,T], past_key_values.{i}.{key,value}[B,kv,P,hd]
    Outputs: hidden_states[B,S,H], present.{i}.{key,value}
    Rotary/positions are computed inside the GQA op from the attention_mask, so we only grow a
    ones mask. Stateful: prefill(embeds) then step(embeds); .hidden holds the last hidden states."""
    def __init__(self, path, device="cpu"):
        import onnx
        self.sess = _ort().InferenceSession(str(path), providers=_providers(device))
        g = onnx.load(str(path), load_external_data=False).graph
        self.in_names = [i.name for i in self.sess.get_inputs()]
        kv = [n for n in self.in_names if n.startswith("past_key_values.")]
        self.n_layers = len({n.split(".")[1] for n in kv})
        shp = next(i for i in g.input if i.name == "past_key_values.0.key").type.tensor_type.shape.dim
        self.kv_heads = shp[1].dim_value
        self.head_dim = shp[3].dim_value
        outs = [o.name for o in self.sess.get_outputs()]
        # TTS backbone emits 'hidden_states'; an ASR decoder that kept lm_head emits 'logits'.
        self.out_name = "logits" if "logits" in outs else "hidden_states"
        self.hidden = None
        self._reset()

    def _reset(self, batch=1):
        z = np.zeros((batch, self.kv_heads, 0, self.head_dim), dtype=np.float32)
        self.past = {}
        for i in range(self.n_layers):
            self.past[f"past_key_values.{i}.key"] = z.copy()
            self.past[f"past_key_values.{i}.value"] = z.copy()
        self.total = 0
        self.batch = batch

    def _run(self, embeds):
        embeds = np.asarray(embeds, dtype=np.float32)
        b, s, _ = embeds.shape
        self.total += s
        feed = {"inputs_embeds": embeds,
                "attention_mask": np.ones((b, self.total), dtype=np.int64)}
        feed.update(self.past)
        outs = self.sess.run(None, feed)
        names = [o.name for o in self.sess.get_outputs()]
        out = dict(zip(names, outs))
        for i in range(self.n_layers):
            self.past[f"past_key_values.{i}.key"] = out[f"present.{i}.key"]
            self.past[f"past_key_values.{i}.value"] = out[f"present.{i}.value"]
        self.hidden = out[self.out_name]
        return self.hidden

    def prefill(self, inputs_embeds):
        """inputs_embeds [B,S,H] -> hidden [B,S,H]. Resets state to a fresh sequence."""
        self._reset(batch=inputs_embeds.shape[0])
        return self._run(inputs_embeds)

    def step(self, inputs_embeds):
        """inputs_embeds [B,s,H] appended to the running KV cache -> hidden [B,s,H]."""
        return self._run(inputs_embeds)


# The exact embed_tokens key per model — the ONNX decoder excludes embeddings, so inference looks
# them up here. Realtime ships BOTH a base language_model AND a tts_language_model embed table (they
# differ); the exported decoder is the tts backbone, so we must match that exact key, not first-win.
EMBED_KEY = {
    "1.5b": "model.language_model.embed_tokens.weight",
    "asr": "model.language_model.embed_tokens.weight",
    "asr-hf": "language_model.model.embed_tokens.weight",
    "realtime": "model.tts_language_model.embed_tokens.weight",
}


def embed_tokens(src, token_ids, model_key):
    """Look up token embeddings for the exported decoder of `model_key`.
    Returns [1, len, H] float32. token_ids: 1-D list/array."""
    from safetensors.torch import load_file
    key = EMBED_KEY.get(model_key)
    if key is None:
        raise RuntimeError(f"no embed_tokens key mapping for model '{model_key}'")
    ids = np.asarray(token_ids, dtype=np.int64).ravel()
    for sf in glob.glob(str(Path(src) / "*.safetensors")):
        d = load_file(sf)
        if key in d:
            return d[key].float().numpy()[ids][None].astype(np.float32)
    raise RuntimeError(f"{key} not found in {src}")


def acoustic_decode_to_wav(dec_op: OnnxOp, latents, scale, bias):
    """latents [B,T,64] (LLM-space) -> waveform. Applies /scale - bias then acoustic_decoder."""
    lat = (np.asarray(latents, dtype=np.float32) / (scale if scale else 1.0)) - bias
    return dec_op.run(latents=lat)


def component_path(onnx_dir, comp):
    p = Path(onnx_dir) / f"{comp}.onnx"
    return p if p.exists() else None
