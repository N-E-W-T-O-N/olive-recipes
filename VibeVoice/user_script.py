"""Olive/ModelBuilder helpers for microsoft/VibeVoice-1.5B → ONNX sub-parts.

VibeVoice = a Qwen2.5-1.5B LLM backbone + a diffusion acoustic head + acoustic/semantic
tokenizers (VAEs) + connectors. Checkpoint key groups (see model.safetensors.index.json):
  model.language_model.*    → Qwen2 backbone (28L, 1536, q/k/v bias) — NO lm_head
                              (the "head" is the diffusion prediction_head, not a vocab head)
  model.acoustic_tokenizer.* → acoustic VAE/codec
  model.semantic_tokenizer.* → semantic tokenizer
  model.prediction_head.*    → DiT-style diffusion denoiser (adaLN + ffn)
  model.acoustic_connector.* / model.semantic_connector.* → projection MLPs into LLM space

Sub-model plan (template: OmniVoice / chandra):
  llm_decoder      → ModelBuilder INT4, inputs_embeds → hidden_states
                     (exclude_embeds + exclude_lm_head: text embed + audio connectors are a
                      separate fusion step; the head is the diffusion head — same shape as the
                      Higgs / OmniVoice decoders).  ← implemented here (no vibevoice pkg needed)
  acoustic_tokenizer / diffusion_head / connectors → Olive, need the `vibevoice` package to
                     instantiate the custom modules (auto_map is null; not in transformers).
"""
import json
import os
import shutil
from pathlib import Path

# Qwen2 tokenizer source (VibeVoice ships no tokenizer; it uses the Qwen2.5 vocab = 151936).
QWEN2_TOKENIZER_ID = "Qwen/Qwen2.5-1.5B"
LM_PREFIX = "model.language_model."


def extract_qwen2_standalone(model_path: str, output_dir: str) -> str:
    """Write a standalone Qwen2ForCausalLM HF dir from VibeVoice's `language_model.*` weights.

    ModelBuilder (onnxruntime-genai) needs a stock Qwen2 directory. We remap
    `model.language_model.<rest>` → `model.<rest>`, keep `model.embed_tokens.weight`
    (tied head), write a Qwen2 config from `decoder_config`, and fetch the Qwen2.5
    tokenizer (absent from the VibeVoice repo). Returns the standalone dir path.
    """
    from safetensors.torch import load_file, save_file
    import glob

    src = Path(model_path)
    out = Path(output_dir) / "qwen2_standalone"
    out.mkdir(parents=True, exist_ok=True)

    # 1. config.json — decoder_config IS a Qwen2ForCausalLM config
    full = json.loads((src / "config.json").read_text())
    dec = dict(full["decoder_config"])
    dec["architectures"] = ["Qwen2ForCausalLM"]
    dec["model_type"] = "qwen2"
    (out / "config.json").write_text(json.dumps(dec, indent=2))

    # 2. tokenizer — pull Qwen2.5-1.5B's (VibeVoice repo has none)
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(QWEN2_TOKENIZER_ID).save_pretrained(str(out))
        print(f"  [tok] fetched {QWEN2_TOKENIZER_ID} tokenizer")
    except Exception as e:
        print(f"  [tok][warn] could not fetch tokenizer ({e}); genai_config will lack it")

    # 3. weights: remap language_model.* → standard Qwen2 names
    idx = src / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
              if idx.exists() else ["model.safetensors"])
    state = {}
    for shard in shards:
        for k, v in load_file(str(src / shard)).items():
            if k.startswith(LM_PREFIX):
                state["model." + k[len(LM_PREFIX):]] = v
    n_layers = len({k.split(".")[2] for k in state if k.startswith("model.layers.")})
    assert "model.embed_tokens.weight" in state, "embed_tokens missing"
    assert n_layers == dec["num_hidden_layers"], f"{n_layers} != {dec['num_hidden_layers']}"
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"  [LLM] standalone Qwen2 → {out}  ({len(state)} tensors, {n_layers} layers)")
    return str(out)


# ASR-HF language model: Qwen2.5-7B, key groups `language_model.model.*` + `language_model.lm_head.*`
# (untied → real vocab head; ASR generates text, so KEEP lm_head — unlike TTS-1.5B).
ASRHF_LM_MODEL_PREFIX = "language_model.model."
ASRHF_LM_HEAD_PREFIX = "language_model.lm_head."


def extract_qwen2_asrhf(model_path: str, output_dir: str) -> str:
    """Standalone Qwen2ForCausalLM dir from VibeVoice-ASR-HF's `language_model.*` weights.

    ASR generates text, so we keep the lm_head (`language_model.lm_head.*` → `lm_head.*`) and
    remap `language_model.model.*` → `model.*`. Streams shard-by-shard and saves incrementally
    to stay memory-frugal (the 7B is ~15 GB bf16). Config = text_config. Fetches Qwen2.5-7B tok.
    """
    from safetensors.torch import load_file, save_file
    src = Path(model_path)
    out = Path(output_dir) / "qwen2_asrhf_standalone"
    out.mkdir(parents=True, exist_ok=True)

    dec = dict(json.loads((src / "config.json").read_text())["text_config"])
    dec["architectures"] = ["Qwen2ForCausalLM"]
    dec["model_type"] = "qwen2"
    (out / "config.json").write_text(json.dumps(dec, indent=2))

    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B").save_pretrained(str(out))
        print("  [tok] fetched Qwen/Qwen2.5-7B tokenizer")
    except Exception as e:
        print(f"  [tok][warn] {e}")

    idx = src / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
              if idx.exists() else ["model.safetensors"])
    state, n_head = {}, 0
    for shard in shards:                                   # one shard resident at a time
        d = load_file(str(src / shard))
        for k, v in d.items():
            if k.startswith(ASRHF_LM_MODEL_PREFIX):
                state["model." + k[len(ASRHF_LM_MODEL_PREFIX):]] = v
            elif k.startswith(ASRHF_LM_HEAD_PREFIX):
                state["lm_head." + k[len(ASRHF_LM_HEAD_PREFIX):]] = v; n_head += 1
        del d
    n_layers = len({k.split(".")[2] for k in state if k.startswith("model.layers.")})
    assert "model.embed_tokens.weight" in state, "embed_tokens missing"
    assert n_head >= 1, "lm_head missing (ASR needs the vocab head)"
    assert n_layers == dec["num_hidden_layers"], f"{n_layers} != {dec['num_hidden_layers']}"
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"  [LLM] standalone Qwen2-7B → {out} ({len(state)} tensors, {n_layers} layers, +lm_head)")
    return str(out)


# =============================================================================
# Acoustic tokenizer (VAE codec) — vendored `codes/` matches VibeVoice-1.5B EXACTLY
# (552 weights, 0 missing). We import ONLY the tokenizer module in isolation (the
# package __init__ pulls the streaming/diffusion chain → diffusers + a qwen2-tokenizer
# import that transformers 5.10.2 renamed), and shim Auto*.register so it coexists with
# transformers' built-in vibevoice_acoustic_tokenizer registration.
# NOTE: acoustic tokenizers DIFFER per checkpoint (1.5B: downsample_layers; Realtime:
# stages/head; ASR-HF: conv_layers) — this loader targets the 1.5B one.
# =============================================================================
import sys as _sys
_CODES = str(Path(__file__).parent / "codes")


def _codes_import(submodule):
    """Isolated import of a single codes/vibevoice/modular/<submodule> module. Shims
    Auto*.register (coexist with transformers) and injects empty `vibevoice[.modular]`
    parent packages so the package __init__ (diffusers + renamed qwen2-tokenizer) never runs."""
    import types, importlib
    from transformers import AutoConfig, AutoModel
    for cls in (AutoConfig, AutoModel):            # tolerate double-registration
        _r = cls.register
        def _safe(*a, __r=_r, **k):
            try: __r(*a, **k)
            except Exception: pass
        cls.register = staticmethod(_safe)
    for name, sub in [("vibevoice", ""), ("vibevoice.modular", "modular")]:  # empty parent pkgs
        m = types.ModuleType(name); m.__path__ = [os.path.join(_CODES, "vibevoice", sub)]
        _sys.modules[name] = m
    return importlib.import_module("vibevoice.modular." + submodule)


def _codes_tokenizer():
    tok = _codes_import("modular_vibevoice_tokenizer")
    from vibevoice.modular.configuration_vibevoice import VibeVoiceAcousticTokenizerConfig as ACfg
    return tok, ACfg


def _load_acoustic(model_path):
    """Load VibeVoice-1.5B's acoustic tokenizer (VAE) via vendored codes/, weights loaded."""
    import glob
    from safetensors.torch import load_file
    tok, ACfg = _codes_tokenizer()
    cfg = ACfg(**json.loads((Path(model_path) / "config.json").read_text())["acoustic_tokenizer_config"])
    model = tok.VibeVoiceAcousticTokenizerModel(cfg).eval()
    state = {}
    for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
        for k, v in load_file(sf).items():
            if k.startswith("model.acoustic_tokenizer."):
                state[k[len("model.acoustic_tokenizer."):]] = v
    miss, unexp = model.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"acoustic weight mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return model.float()


class AcousticEncoderWrapper:
    """audio [B,1,T] → latents (VAE mean) [B,8,64]."""
    pass


def get_acoustic_encoder_model(model_path=None):
    import torch.nn as nn
    codec = _load_acoustic(model_path)
    if hasattr(codec, "decoder"):
        codec.decoder = None          # drop the unused half → smaller graph + less memory

    class Enc(nn.Module):
        def __init__(s): super().__init__(); s.codec = codec
        def forward(s, audio):
            return s.codec.encode(audio, use_cache=False).mean
    return Enc().eval()


def get_acoustic_encoder_io_config(model=None):
    return {"input_names": ["audio"], "output_names": ["latents"],
            "input_shapes": [[1, 1, 24000]], "input_types": ["float32"],
            "dynamic_axes": {"audio": {0: "batch", 2: "samples"},
                             "latents": {0: "batch", 2: "frames"}}}


def get_acoustic_encoder_dummy_inputs(model=None):
    import torch
    return {"audio": torch.randn(1, 1, 24000, dtype=torch.float32)}


def get_acoustic_decoder_model(model_path=None):
    import torch.nn as nn
    codec = _load_acoustic(model_path)
    if hasattr(codec, "encoder"):
        codec.encoder = None          # drop the unused half → smaller graph + less memory

    class Dec(nn.Module):
        def __init__(s): super().__init__(); s.codec = codec
        def forward(s, latents):
            out = s.codec.decode(latents, use_cache=False)
            return out.sample if hasattr(out, "sample") else (out.audio if hasattr(out, "audio") else out[0])
    return Dec().eval()


def get_acoustic_decoder_io_config(model=None):
    # latents [B, frames, 64]: the frame axis is dim 1 (dim 2 is the vae_dim). Use dynamo
    # dynamic_shapes so the exported decoder accepts a variable number of frames.
    return {"input_names": ["latents"], "output_names": ["audio"],
            "input_shapes": [[1, 8, 64]], "input_types": ["float32"],
            "dynamic_shapes": {"latents": {0: "batch", 1: "frames"}}}


def get_acoustic_decoder_dummy_inputs(model=None):
    import torch
    return {"latents": torch.randn(1, 8, 64, dtype=torch.float32)}


# =============================================================================
# ASR-HF acoustic encoder — transformers-NATIVE (VibeVoiceAcousticTokenizerEncoderModel),
# different arch than 1.5B (conv_layers). No codes/ / shim needed. Loads only the
# `acoustic_tokenizer_encoder.*` weights (not the 7B LLM) so it fits in memory.
# =============================================================================

def _load_asrhf_encoder(model_path, cfg_key, prefix):
    """Generic ASR-HF tokenizer-encoder loader. Both the acoustic and semantic encoders share
    the transformers-native `VibeVoiceAcousticTokenizerEncoderModel` class (model_type
    `vibevoice_acoustic_tokenizer_encoder`); they differ only in config + weight prefix.
    Loads ONLY the `<prefix>.*` weights (not the 7B LLM) so it fits in memory."""
    from transformers import VibeVoiceAcousticTokenizerEncoderModel
    from transformers.models.vibevoice_acoustic_tokenizer.configuration_vibevoice_acoustic_tokenizer \
        import VibeVoiceAcousticTokenizerEncoderConfig
    from safetensors import safe_open
    p = Path(model_path)
    cfg = VibeVoiceAcousticTokenizerEncoderConfig(
        **json.loads((p / "config.json").read_text())[cfg_key])
    model = VibeVoiceAcousticTokenizerEncoderModel(cfg).eval()
    idxp = p / "model.safetensors.index.json"
    pfx = prefix + "."
    if idxp.exists():
        wm = json.loads(idxp.read_text())["weight_map"]
        shards = {v for k, v in wm.items() if k.startswith(pfx)}
    else:
        shards = [x.name for x in p.glob("*.safetensors")]
    state = {}
    for sh in shards:
        with safe_open(str(p / sh), "pt") as h:
            for k in h.keys():
                if k.startswith(pfx):
                    state[k[len(pfx):]] = h.get_tensor(k)
    miss, unexp = model.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"{prefix} mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return model.float()


def _load_asrhf_acoustic_encoder(model_path):
    return _load_asrhf_encoder(model_path, "acoustic_tokenizer_encoder_config",
                               "acoustic_tokenizer_encoder")


def _load_asrhf_semantic_encoder(model_path):
    return _load_asrhf_encoder(model_path, "semantic_tokenizer_encoder_config",
                               "semantic_tokenizer_encoder")


class _EncWrap:
    pass


def _enc_wrapper(enc):
    import torch.nn as nn

    class Enc(nn.Module):
        def __init__(s): super().__init__(); s.enc = enc
        def forward(s, audio):
            o = s.enc(audio)
            return o.latents if hasattr(o, "latents") else (o[0] if isinstance(o, (tuple, list)) else o)
    return Enc().eval()


def _enc_io_config():
    return {"input_names": ["audio"], "output_names": ["latents"],
            "input_shapes": [[1, 1, 24000]], "input_types": ["float32"],
            "dynamic_axes": {"audio": {0: "batch", 2: "samples"},
                             "latents": {0: "batch", 1: "frames"}}}


def _enc_dummy():
    import torch
    return {"audio": torch.randn(1, 1, 24000, dtype=torch.float32)}


def get_asrhf_acoustic_encoder_model(model_path=None):
    return _enc_wrapper(_load_asrhf_acoustic_encoder(model_path))


def get_asrhf_acoustic_encoder_io_config(model=None):
    return _enc_io_config()


def get_asrhf_acoustic_encoder_dummy_inputs(model=None):
    return _enc_dummy()


def get_asrhf_semantic_encoder_model(model_path=None):
    return _enc_wrapper(_load_asrhf_semantic_encoder(model_path))


def get_asrhf_semantic_encoder_io_config(model=None):
    return _enc_io_config()


def get_asrhf_semantic_encoder_dummy_inputs(model=None):
    return _enc_dummy()


# =============================================================================
# ASR-HF multi_modal_projector — fuses acoustic latents [B,T,64] + semantic latents
# [B,T,128] → LLM-space features [B,T,3584] (VibeVoiceAsrMultiModalProjector, native).
# =============================================================================

def _load_asrhf_projector(model_path):
    from transformers import AutoConfig
    from transformers.models.vibevoice_asr.modeling_vibevoice_asr import VibeVoiceAsrMultiModalProjector
    from safetensors import safe_open
    p = Path(model_path)
    cfg = AutoConfig.from_pretrained(str(p))
    model = VibeVoiceAsrMultiModalProjector(cfg).eval()
    idxp = p / "model.safetensors.index.json"
    pfx = "multi_modal_projector."
    shards = ({v for k, v in json.loads(idxp.read_text())["weight_map"].items() if k.startswith(pfx)}
              if idxp.exists() else [x.name for x in p.glob("*.safetensors")])
    state = {}
    for sh in shards:
        with safe_open(str(p / sh), "pt") as h:
            for k in h.keys():
                if k.startswith(pfx):
                    state[k[len(pfx):]] = h.get_tensor(k)
    miss, unexp = model.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"projector mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return model.float()


def get_asrhf_projector_model(model_path=None):
    return _load_asrhf_projector(model_path)


def get_asrhf_projector_io_config(model=None):
    return {"input_names": ["acoustic_latents", "semantic_latents"],
            "output_names": ["features"],
            "input_shapes": [[1, 8, 64], [1, 8, 128]],
            "input_types": ["float32", "float32"],
            "dynamic_shapes": {"acoustic_latents": {0: "batch", 1: "frames"},
                               "semantic_latents": {0: "batch", 1: "frames"}}}


def get_asrhf_projector_dummy_inputs(model=None):
    import torch
    return {"acoustic_latents": torch.randn(1, 8, 64, dtype=torch.float32),
            "semantic_latents": torch.randn(1, 8, 128, dtype=torch.float32)}


# =============================================================================
# Realtime-0.5B (`vibevoice_streaming`, auto_map null → codes/) — a streaming TTS
# checkpoint. Key groups: model.tts_language_model.* (Qwen2.5-0.5B backbone, 20 layers,
# no lm_head), model.acoustic_tokenizer.* (DECODER-ONLY, 276 — no encoder shipped, since
# inference only DECODES generated latents → audio), model.language_model.* (4-layer base),
# model.prediction_head.* (diffusion), model.acoustic_connector.*, tts_eos_classifier.*.
# The acoustic decoder matches the codes/ class EXACTLY (stages/head naming; decoder 0
# missing / 0 unexpected) — NOT transformers-native (conv_layers/convtr naming).
# =============================================================================
RT_TTS_LM_PREFIX = "model.tts_language_model."


def extract_qwen2_realtime(model_path: str, output_dir: str) -> str:
    """Standalone Qwen2ForCausalLM dir from Realtime's `tts_language_model.*` backbone.

    This is a TTS backbone (like VibeVoice-1.5B): NO lm_head (the head is the diffusion
    prediction_head), so it's built exclude_embeds+exclude_lm_head → inputs_embeds→hidden.
    Config = decoder_config, but num_hidden_layers overridden to the ACTUAL stored count
    (tts_backbone_num_hidden_layers = 20; decoder_config says 24). Tokenizer = Qwen2.5-0.5B.
    """
    from safetensors.torch import load_file, save_file
    import glob
    src = Path(model_path)
    out = Path(output_dir) / "qwen2_realtime_standalone"
    out.mkdir(parents=True, exist_ok=True)

    full = json.loads((src / "config.json").read_text())
    dec = dict(full["decoder_config"])
    n_real = full.get("tts_backbone_num_hidden_layers", dec["num_hidden_layers"])
    dec["num_hidden_layers"] = n_real
    dec["architectures"] = ["Qwen2ForCausalLM"]
    dec["model_type"] = "qwen2"
    (out / "config.json").write_text(json.dumps(dec, indent=2))

    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(str(out))
        print("  [tok] fetched Qwen/Qwen2.5-0.5B tokenizer")
    except Exception as e:
        print(f"  [tok][warn] {e}")

    state = {}
    for sf in glob.glob(str(src / "*.safetensors")):
        for k, v in load_file(sf).items():
            if k.startswith(RT_TTS_LM_PREFIX):
                state["model." + k[len(RT_TTS_LM_PREFIX):]] = v
    n_layers = len({k.split(".")[2] for k in state if k.startswith("model.layers.")})
    assert "model.embed_tokens.weight" in state, "embed_tokens missing"
    assert n_layers == n_real, f"{n_layers} != {n_real}"
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"  [LLM] standalone Qwen2-0.5B → {out} ({len(state)} tensors, {n_layers} layers, no lm_head)")
    return str(out)


def _load_realtime_acoustic_decoder(model_path):
    """Realtime acoustic tokenizer (DECODER-ONLY) via codes/. Loads only `decoder.*` weights
    (encoder absent from the checkpoint), drops the encoder module. codes/ matches exactly."""
    import glob
    from safetensors.torch import load_file
    tok, ACfg = _codes_tokenizer()
    cfg = ACfg(**json.loads((Path(model_path) / "config.json").read_text())["acoustic_tokenizer_config"])
    model = tok.VibeVoiceAcousticTokenizerModel(cfg).eval()
    state = {}
    for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
        for k, v in load_file(sf).items():
            if k.startswith("model.acoustic_tokenizer.decoder."):
                state[k[len("model.acoustic_tokenizer."):]] = v
    miss, unexp = model.load_state_dict(state, strict=False)
    dec_miss = [k for k in miss if k.startswith("decoder.")]
    assert not dec_miss and not unexp, f"rt acoustic decoder mismatch: dec_missing={len(dec_miss)} unexpected={len(unexp)}"
    model.encoder = None
    return model.float()


def get_realtime_acoustic_decoder_model(model_path=None):
    import torch.nn as nn
    codec = _load_realtime_acoustic_decoder(model_path)

    class Dec(nn.Module):
        def __init__(s): super().__init__(); s.codec = codec
        def forward(s, latents):
            out = s.codec.decode(latents, use_cache=False)
            return out.sample if hasattr(out, "sample") else (out.audio if hasattr(out, "audio") else out[0])
    return Dec().eval()


def get_realtime_acoustic_decoder_io_config(model=None):
    return {"input_names": ["latents"], "output_names": ["audio"],
            "input_shapes": [[1, 10, 64]], "input_types": ["float32"],
            "dynamic_shapes": {"latents": {0: "batch", 1: "frames"}}}


def get_realtime_acoustic_decoder_dummy_inputs(model=None):
    import torch
    return {"latents": torch.randn(1, 10, 64, dtype=torch.float32)}


# =============================================================================
# Diffusion prediction_head + speech connectors (shared 1.5B / Realtime, via codes/).
#   diffusion_head: ONE denoise step (noisy_images[B,64], timesteps[B], condition[B,H]) → pred[B,64].
#                   The ~20-step DDPM sampling loop stays in the pipeline; ONNX = one step.
#   connector: SpeechConnector fc1(in→H) → RMSNorm(H) → fc2(H→H). 1.5B: acoustic 64→1536,
#              semantic 128→1536; Realtime: acoustic 64→896. (semantic_connector: 1.5B only.)
# =============================================================================

def _load_diffusion_head(model_path):
    import glob
    from safetensors.torch import load_file
    dh = _codes_import("modular_vibevoice_diffusion_head")
    from vibevoice.modular.configuration_vibevoice import VibeVoiceDiffusionHeadConfig as DCfg
    cfg = DCfg(**json.loads((Path(model_path) / "config.json").read_text())["diffusion_head_config"])
    model = dh.VibeVoiceDiffusionHead(cfg).eval()
    state = {}
    for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
        for k, v in load_file(sf).items():
            if k.startswith("model.prediction_head."):
                state[k[len("model.prediction_head."):]] = v
    miss, unexp = model.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"diffusion head mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return model.float(), cfg


def get_diffusion_head_model(model_path=None):
    model, _ = _load_diffusion_head(model_path)
    return model


def get_diffusion_head_io_config(model=None):
    return {"input_names": ["noisy_images", "timesteps", "condition"],
            "output_names": ["pred"],
            "dynamic_shapes": {"noisy_images": {0: "batch"},
                               "timesteps": {0: "batch"},
                               "condition": {0: "batch"}}}


def _diffusion_head_hidden(model_path):
    return int(json.loads((Path(model_path) / "config.json").read_text())["diffusion_head_config"]["hidden_size"])


def get_diffusion_head_dummy_inputs(model=None):
    import torch, os as _os
    # hidden_size differs per checkpoint (1.5B=1536, Realtime=896); read from config via env or default.
    h = int(_os.environ.get("VV_HEAD_HIDDEN", "1536"))
    # timesteps must be FLOAT: TimestepEmbedder casts its sinusoidal embedding back to t.dtype
    # before the (float) MLP, so int64 would break the matmul.
    return {"noisy_images": torch.randn(4, 64, dtype=torch.float32),
            "timesteps": torch.rand(4, dtype=torch.float32) * 1000,
            "condition": torch.randn(4, h, dtype=torch.float32)}


class _SpeechConnector:
    pass


def _load_connector(model_path, which):
    import torch.nn as nn
    import glob
    from safetensors.torch import load_file
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    pfx = f"model.{which}_connector."
    st = {}
    for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
        for k, v in load_file(sf).items():
            if k.startswith(pfx):
                st[k[len(pfx):]] = v
    assert st, f"no weights for {which}_connector"
    in_dim = st["fc1.weight"].shape[1]; out_dim = st["fc1.weight"].shape[0]

    class SpeechConnector(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc1 = nn.Linear(in_dim, out_dim); s.norm = LlamaRMSNorm(out_dim, eps=1e-6)
            s.fc2 = nn.Linear(out_dim, out_dim)
        def forward(s, features):
            return s.fc2(s.norm(s.fc1(features)))
    m = SpeechConnector().eval()
    miss, unexp = m.load_state_dict(st, strict=False)
    assert not miss and not unexp, f"{which}_connector mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return m.float(), in_dim


def get_acoustic_connector_model(model_path=None):
    m, _ = _load_connector(model_path, "acoustic"); return m


def get_semantic_connector_model(model_path=None):
    m, _ = _load_connector(model_path, "semantic"); return m


def _connector_io_config():
    return {"input_names": ["features"], "output_names": ["hidden"],
            "dynamic_shapes": {"features": {0: "batch", 1: "frames"}}}


def get_acoustic_connector_io_config(model=None):
    return _connector_io_config()


def get_semantic_connector_io_config(model=None):
    return _connector_io_config()


def get_acoustic_connector_dummy_inputs(model=None):
    import torch
    return {"features": torch.randn(1, 8, 64, dtype=torch.float32)}


def get_semantic_connector_dummy_inputs(model=None):
    import torch
    return {"features": torch.randn(1, 8, 128, dtype=torch.float32)}


# =============================================================================
# VibeVoice-ASR (`vibevoice`, VibeVoiceForASRTraining, codes/) — an audio→text ASR model:
# same codes/ family as 1.5B TTS but the LLM is Qwen2.5-7B WITH lm_head (generates text) and
# there is NO prediction_head (no audio generation). Front-end = full acoustic tokenizer (552,
# enc+dec) + semantic tokenizer (276, ENCODE-only) + acoustic/semantic connectors — all load via
# the existing codes/ loaders (`_load_acoustic`, `_load_connector`, `_load_semantic`).
# Weight layout: model.language_model.* (338) + top-level lm_head.weight (unlike ASR-HF).
# =============================================================================

def _load_semantic(model_path):
    """Semantic tokenizer (ENCODE-only, deterministic latent = encode().mean) via codes/."""
    import glob
    from safetensors.torch import load_file
    tok = _codes_import("modular_vibevoice_tokenizer")
    from vibevoice.modular.configuration_vibevoice import VibeVoiceSemanticTokenizerConfig as SCfg
    cfg = SCfg(**json.loads((Path(model_path) / "config.json").read_text())["semantic_tokenizer_config"])
    model = tok.VibeVoiceSemanticTokenizerModel(cfg).eval()
    state = {}
    for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
        for k, v in load_file(sf).items():
            if k.startswith("model.semantic_tokenizer."):
                state[k[len("model.semantic_tokenizer."):]] = v
    miss, unexp = model.load_state_dict(state, strict=False)
    assert not miss and not unexp, f"semantic tokenizer mismatch: missing={len(miss)} unexpected={len(unexp)}"
    return model.float()


def get_semantic_tokenizer_encoder_model(model_path=None):
    import torch.nn as nn
    codec = _load_semantic(model_path)

    class Enc(nn.Module):
        def __init__(s): super().__init__(); s.codec = codec
        def forward(s, audio):
            return s.codec.encode(audio, use_cache=False).mean
    return Enc().eval()


def get_semantic_tokenizer_encoder_io_config(model=None):
    return {"input_names": ["audio"], "output_names": ["latents"],
            "input_shapes": [[1, 1, 24000]], "input_types": ["float32"],
            "dynamic_axes": {"audio": {0: "batch", 2: "samples"},
                             "latents": {0: "batch", 1: "frames"}}}


def get_semantic_tokenizer_encoder_dummy_inputs(model=None):
    import torch
    return {"audio": torch.randn(1, 1, 24000, dtype=torch.float32)}


def extract_qwen2_asr(model_path: str, output_dir: str) -> str:
    """Standalone Qwen2ForCausalLM dir from VibeVoice-ASR's `model.language_model.*` + top-level
    `lm_head.weight` (ASR emits text → KEEP lm_head). Config = decoder_config (Qwen2.5-7B).
    Streams shards to stay memory-frugal; fetches Qwen2.5-7B tokenizer."""
    from safetensors.torch import load_file, save_file
    src = Path(model_path)
    out = Path(output_dir) / "qwen2_asr_standalone"
    out.mkdir(parents=True, exist_ok=True)

    dec = dict(json.loads((src / "config.json").read_text())["decoder_config"])
    dec["architectures"] = ["Qwen2ForCausalLM"]
    dec["model_type"] = "qwen2"
    (out / "config.json").write_text(json.dumps(dec, indent=2))
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B").save_pretrained(str(out))
        print("  [tok] fetched Qwen/Qwen2.5-7B tokenizer")
    except Exception as e:
        print(f"  [tok][warn] {e}")

    idx = src / "model.safetensors.index.json"
    shards = (sorted(set(json.loads(idx.read_text())["weight_map"].values()))
              if idx.exists() else ["model.safetensors"])
    state, n_head = {}, 0
    for shard in shards:
        d = load_file(str(src / shard))
        for k, v in d.items():
            if k.startswith(LM_PREFIX):                       # model.language_model.* → model.*
                state["model." + k[len(LM_PREFIX):]] = v
            elif k == "lm_head.weight":
                state["lm_head.weight"] = v; n_head += 1
        del d
    n_layers = len({k.split(".")[2] for k in state if k.startswith("model.layers.")})
    assert "model.embed_tokens.weight" in state, "embed_tokens missing"
    assert n_head >= 1, "lm_head missing (ASR needs the vocab head)"
    assert n_layers == dec["num_hidden_layers"], f"{n_layers} != {dec['num_hidden_layers']}"
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"  [LLM] standalone Qwen2-7B → {out} ({len(state)} tensors, {n_layers} layers, +lm_head)")
    return str(out)
