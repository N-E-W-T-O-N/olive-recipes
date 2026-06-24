"""Olive PyTorchModel loaders for Qwen3-TTS sub-models (12 Hz).

Loaded via the vendored qwen_tts package (codes/) under transformers 4.57.3.
optimize.py declares the deps as PEP-723 inline metadata and runs with `uv run`,
so the shared 5.10.2 venv is untouched.

Sub-models
----------
TTS model (qwen3_tts):
  talker          — Qwen3TTSTalkerModel (28 L, MROPE) + codec_head → first-codebook
                    logits / hidden_states (the LLM sub-part)
  code_predictor  — 5 L, 16 per-group heads → residual code groups
  speaker_encoder — ECAPA-TDNN speaker embedding (custom-voice / clone)   [optional]

Tokenizer (qwen3_tts_tokenizer_12hz):
  tok_encoder     — waveform → codes
  tok_decoder     — codes → waveform
"""
import math
import os
import sys

import torch
import torch.nn as nn


def _patch_mimi_static_padding():
    """Make Mimi conv 'same-length' padding pure-Python so torch.export sees
    concrete conv output lengths.

    Stock `MimiConv1d._get_extra_padding_for_conv1d` computes the extra padding as
    a 0-dim *tensor* (torch.ceil(...).to(int64)). Under torch.export each value is
    realized via `.item()` → an unbacked symint (u0…u29). Those flow into the frame
    count, so the RVQ `torch.cdist` later branches on an unbacked `frames > 25`
    (its matmul heuristic) and export aborts (GuardOnDataDependentSymNode).

    With a static input length the padding is a constant, so compute it in Python
    ints; conv output lengths then become concrete and the cdist guard resolves.
    Idempotent.
    """
    from transformers.models.mimi import modeling_mimi as mm
    if getattr(mm.MimiConv1d, "_static_pad_patched", False):
        return

    def _get_extra_padding_for_conv1d(self, hidden_states):
        # kernel_size/stride/padding_total are 0-dim tensors in this model → int()
        length = int(hidden_states.shape[-1])
        kernel = int(self.kernel_size)
        stride = int(self.stride)
        pad_total = int(self.padding_total)
        n_frames = (length - kernel + pad_total) / stride + 1
        n_frames = math.ceil(n_frames) - 1
        ideal_length = n_frames * stride + kernel - pad_total
        return ideal_length - length          # plain Python int

    mm.MimiConv1d._get_extra_padding_for_conv1d = _get_extra_padding_for_conv1d
    mm.MimiConv1d._static_pad_patched = True


def _intify_mimi_convs(model):
    """Replace MimiConv1d's tensor buffers (stride/kernel_size/padding_total) and
    the derived padding_left/right with plain Python ints.

    In this model these are registered as 0-dim int64 buffers. Reading them inside
    forward (even via int()) becomes a tensor `.item()` under torch.export → an
    unbacked symint. Converting them to real ints up front keeps the conv padding
    math fully static.
    """
    from transformers.models.mimi import modeling_mimi as mm
    for m in model.modules():
        if isinstance(m, mm.MimiConv1d):
            for name in ("stride", "kernel_size", "padding_total"):
                val = int(getattr(m, name))
                if name in m._buffers:
                    del m._buffers[name]
                setattr(m, name, val)
            m.padding_right = int(m.padding_total) // 2
            m.padding_left = int(m.padding_total) - m.padding_right
    return model

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.join(_HERE, "codes") not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, "codes"))


# ── loading helpers ───────────────────────────────────────────────────────────

# Talker dims used by io_config/dummy funcs. Defaults are the 1.7B values, so those exports
# are byte-identical to before; _load_tts() overwrites them from the loaded model's config so
# OTHER sizes (e.g. 0.6B) export with their own dims. Olive calls model_loader BEFORE
# io_config/dummy_inputs_func, so these are populated by the time the dummies are built.
_DIMS = {"hidden": 2048, "n_layers": 28, "n_kv": 8, "head_dim": 128, "n_groups": 16}


def _load_tts(model_path: str):
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    tc = model.config.talker_config
    _DIMS.update(
        hidden=tc.hidden_size,
        n_layers=tc.num_hidden_layers,
        n_kv=getattr(tc, "num_key_value_heads", tc.num_attention_heads),
        head_dim=getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads),
        n_groups=tc.num_code_groups,
    )
    return model


def _load_tokenizer(tok_path: str):
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model)
    tok = Qwen3TTSTokenizerV2Model.from_pretrained(
        tok_path, dtype=torch.float32, attn_implementation="eager")
    tok.eval()
    return tok


def _tts_dims(model):
    tc = model.config.talker_config
    return dict(hidden=tc.hidden_size, text_vocab=tc.text_vocab_size,
                codec_vocab=tc.vocab_size, n_groups=tc.num_code_groups,
                cp_hidden=tc.code_predictor_config.hidden_size,
                cp_vocab=tc.code_predictor_config.vocab_size)


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDING PRIMITIVES (learned stages of the talker prefill).
#
# The full prefill in Qwen3TTSForConditionalGeneration.generate is control flow
# (variable text len, optional voice-clone / ICL branches, concat, MROPE position
# bookkeeping) — that belongs in inference.py, not a single ONNX graph. The only
# *learned* pieces are these two lookups (+ the text ResizeMLP), exported here so
# every weight runs through ONNX while orchestration stays in Python.
# ════════════════════════════════════════════════════════════════════════════

class TextEmbedWrapper(nn.Module):
    """text_ids [B,T] → text_projection(text_embedding(ids))  [B,T,hidden].

    Same call as generate(): talker.text_projection(talker.get_text_embeddings()(ids)).
    Covers tts_bos/eos/pad embeds too — those are just specific text ids.
    """
    def __init__(self, talker):
        super().__init__()
        self.text_embedding = talker.get_text_embeddings()   # (text_vocab, text_hidden)
        self.text_projection = talker.text_projection        # ResizeMLP → hidden

    def forward(self, text_ids):                              # [B,T] int64
        return self.text_projection(self.text_embedding(text_ids))


class CodecEmbedWrapper(nn.Module):
    """codec_ids [B,T] → codec_embedding(ids)  [B,T,hidden]  (first-codebook embed)."""
    def __init__(self, talker):
        super().__init__()
        self.codec_embedding = talker.model.codec_embedding  # (vocab, hidden)

    def forward(self, codec_ids):                            # [B,T] int64
        return self.codec_embedding(codec_ids)


def get_text_embed_model(model_path=None):
    return TextEmbedWrapper(_load_tts(model_path).talker).eval()


def get_text_embed_io_config(model=None):
    return {
        "input_names": ["text_ids"], "output_names": ["text_embeds"],
        "input_shapes": [[1, 16]], "input_types": ["int64"],
        "dynamic_axes": {"text_ids": {0: "batch", 1: "seq"},
                         "text_embeds": {0: "batch", 1: "seq"}},
    }


def get_text_embed_dummy_inputs(model=None):
    return {"text_ids": torch.randint(0, 1000, (1, 16), dtype=torch.int64)}


def get_codec_embed_model(model_path=None):
    return CodecEmbedWrapper(_load_tts(model_path).talker).eval()


def get_codec_embed_io_config(model=None):
    return {
        "input_names": ["codec_ids"], "output_names": ["codec_embeds"],
        "input_shapes": [[1, 16]], "input_types": ["int64"],
        "dynamic_axes": {"codec_ids": {0: "batch", 1: "seq"},
                         "codec_embeds": {0: "batch", 1: "seq"}},
    }


def get_codec_embed_dummy_inputs(model=None):
    return {"codec_ids": torch.randint(0, 2048, (1, 16), dtype=torch.int64)}


# ════════════════════════════════════════════════════════════════════════════
# TALKER  (LLM sub-part): input codec/text token embeds → codec_head logits
# ════════════════════════════════════════════════════════════════════════════

class TalkerWrapper(nn.Module):
    """inputs_embeds [B,T,H] + position_ids [3,B,T] (MROPE) →
       (codec logits [B,T,V], last_hidden_state [B,T,H]).

    Both outputs are needed for generation: the logits give the first-codebook
    token, and last_hidden_state is the `past_hidden` that conditions the
    code_predictor for the residual codes (see modeling forward, line ~1670).
    use_cache=False (no-cache forward — the AR loop re-runs the full prefix).
    """
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model          # Qwen3TTSTalkerModel
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, position_ids, attention_mask):
        out = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask, use_cache=False,
                         return_dict=True)
        hidden = out.last_hidden_state
        return self.codec_head(hidden), hidden


def get_talker_model(model_path=None):
    return TalkerWrapper(_load_tts(model_path).talker).eval()


def get_talker_io_config(model=None):
    return {
        "input_names": ["inputs_embeds", "position_ids", "attention_mask"],
        "output_names": ["logits", "hidden_states"],
        "input_shapes": [[1, 32, _DIMS["hidden"]], [3, 1, 32], [1, 32]],
        "input_types": ["float32", "int64", "int64"],
        "dynamic_axes": {
            "inputs_embeds": {0: "batch", 1: "seq"},
            "position_ids": {1: "batch", 2: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
            "hidden_states": {0: "batch", 1: "seq"},
        },
        # dynamo ignores dynamic_axes and requires dynamic_shapes (Olive conversion.py
        # line ~368). Without this the seq dim specializes to the dummy's 32 and the
        # AR loop can't grow the prefix. Shared "seq" name ties the three inputs.
        "dynamic_shapes": {
            "inputs_embeds": {0: "batch", 1: "seq"},
            "position_ids": {1: "batch", 2: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
    }


def get_talker_dummy_inputs(model=None):
    return {
        "inputs_embeds": torch.randn(1, 32, _DIMS["hidden"], dtype=torch.float32),
        "position_ids": torch.zeros(3, 1, 32, dtype=torch.long),
        "attention_mask": torch.ones(1, 32, dtype=torch.long),
    }


# ════════════════════════════════════════════════════════════════════════════
# CODE PREDICTOR: talker hidden + first codes → residual code-group logits
# ════════════════════════════════════════════════════════════════════════════

class CodePredictorWrapper(nn.Module):
    """Teacher-forced residual code predictor (parity-faithful).

    Mirrors `Qwen3TTSTalkerForConditionalGeneration.forward_sub_talker_finetune`
    + `code_predictor.forward_finetune`:
      seq = [ talker_hidden,
              talker.codec_embedding(code0),
              cp.codec_embedding[i-1](code_i) for i in 1..14 ]            # [B,16,2048]
      → small_to_mtp_projection → [B,16,1024] → predictor decoder
      → lm_head[i-1](h[:, i]) for i in 1..15 → logits [B,15,vocab].

    All embedding/projection stages are IN-GRAPH (no Python-side model math).
    Needs all 16 codes up front, so this is for parity, not generation.
    """
    def __init__(self, talker):
        super().__init__()
        self.talker_codec_embedding = talker.model.codec_embedding   # first code (2048)
        self.cp = talker.code_predictor
        self.n_groups = self.cp.config.num_code_groups               # 16

    def forward(self, talker_hidden, codec_ids):     # [B,2048], [B,16] int64
        parts = [talker_hidden.unsqueeze(1),
                 self.talker_codec_embedding(codec_ids[:, :1])]
        for i in range(1, self.n_groups - 1):
            parts.append(self.cp.model.codec_embedding[i - 1](codec_ids[:, i:i + 1]))
        emb = torch.cat(parts, dim=1)                                # [B,16,2048]
        emb = self.cp.small_to_mtp_projection(emb)                   # [B,16,1024]
        out = self.cp.model(inputs_embeds=emb, use_cache=False, return_dict=True)
        h = out.last_hidden_state
        logits = [self.cp.lm_head[i - 1](h[:, i]) for i in range(1, self.n_groups)]
        return torch.stack(logits, dim=1)                            # [B,15,vocab]


def get_code_predictor_model(model_path=None):
    return CodePredictorWrapper(_load_tts(model_path).talker).eval()


def get_code_predictor_io_config(model=None):
    return {
        "input_names": ["talker_hidden", "codec_ids"],
        "output_names": ["group_logits"],
        "input_shapes": [[1, _DIMS["hidden"]], [1, _DIMS["n_groups"]]],
        "input_types": ["float32", "int64"],
        "dynamic_axes": {"talker_hidden": {0: "batch"},
                         "codec_ids": {0: "batch"},
                         "group_logits": {0: "batch"}},
    }


def get_code_predictor_dummy_inputs(model=None):
    return {"talker_hidden": torch.randn(1, _DIMS["hidden"], dtype=torch.float32),
            "codec_ids": torch.randint(0, 2048, (1, _DIMS["n_groups"]), dtype=torch.int64)}


# ════════════════════════════════════════════════════════════════════════════
# RESIDUAL EMBED: the per-step next-token embedding for the talker AR loop.
#
# After a talker step yields the 16 codes (code0 + 15 residuals), the next
# talker input is  codec_hiddens.sum(1)  (modeling forward, lines 1682-1687):
#   sum( talker.model.codec_embedding(code0),
#        code_predictor.model.codec_embedding[i](code_{i+1}) for i in 0..14 )
# code0 lives in the talker's codec table; the 15 residual codes use the
# predictor's OWN per-group embedding ModuleList — which is buried inside the
# code_predictor graph and not otherwise exposed. This wrapper sums all 16 to
# the [B,hidden] vector the AR loop adds to the trailing-text hidden.
# ════════════════════════════════════════════════════════════════════════════

class ResidualEmbedWrapper(nn.Module):
    """codec_ids [B,16] → codec_hiddens.sum(1)  [B,hidden]  (next talker input)."""
    def __init__(self, talker):
        super().__init__()
        self.talker_codec_embedding = talker.model.codec_embedding   # code0 (2048)
        self.cp_codec_embedding = talker.code_predictor.model.codec_embedding  # 15× (2048)
        self.n_groups = talker.code_predictor.config.num_code_groups  # 16

    def forward(self, codec_ids):                     # [B,16] int64
        acc = self.talker_codec_embedding(codec_ids[:, 0])           # [B,2048]
        for i in range(self.n_groups - 1):
            acc = acc + self.cp_codec_embedding[i](codec_ids[:, i + 1])
        return acc                                                   # [B,2048]


def get_residual_embed_model(model_path=None):
    return ResidualEmbedWrapper(_load_tts(model_path).talker).eval()


def get_residual_embed_io_config(model=None):
    return {
        "input_names": ["codec_ids"], "output_names": ["step_embed"],
        "input_shapes": [[1, _DIMS["n_groups"]]], "input_types": ["int64"],
        "dynamic_axes": {"codec_ids": {0: "batch"}, "step_embed": {0: "batch"}},
    }


def get_residual_embed_dummy_inputs(model=None):
    return {"codec_ids": torch.randint(0, 2048, (1, _DIMS["n_groups"]), dtype=torch.int64)}


# ════════════════════════════════════════════════════════════════════════════
# TALKER (KV-CACHE): unified prefill+decode with past_key_values.
#
# Same math as TalkerWrapper but threads a DynamicCache so the AR loop is O(n)
# instead of re-running the whole prefix each step. ONE model handles both:
#   prefill  → past seq = 0, cur = T_prefill
#   decode   → past seq = T, cur = 1
# Olive converts the dummy legacy-list past_key_values → DynamicCache before
# export and the output DynamicCache → flattened present tensors (conversion.py
# `past_key_values_to_dynamic_cache` / `_convert_dynamic_shapes_for_dynamic_cache`).
# 28 layers, 8 KV heads, head_dim 128.
# ════════════════════════════════════════════════════════════════════════════

class TalkerCacheWrapper(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds, position_ids, attention_mask, past_kv):
        # past_kv is a LIST of [key, value] tensor pairs (legacy layout). Named `past_kv`
        # (not `past_key_values`) so Olive does NOT auto-convert its dynamic_shapes into the
        # DynamicCache pytree (which torch.export rejected). Build the cache in-graph instead.
        from transformers import DynamicCache
        legacy = tuple((past_kv[i][0], past_kv[i][1]) for i in range(len(past_kv)))
        dc = DynamicCache.from_legacy_cache(legacy)
        out = self.model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                         attention_mask=attention_mask, past_key_values=dc,
                         use_cache=True, return_dict=True)
        hidden = out.last_hidden_state
        present = out.past_key_values.to_legacy_cache()       # tuple of (key, value)
        return self.codec_head(hidden), hidden, present


def _talker_cache_dims(model):
    tc = model.config.talker_config
    n_kv = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
    hd = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
    return tc.num_hidden_layers, n_kv, hd


def get_talker_cache_model(model_path=None):
    return TalkerCacheWrapper(_load_tts(model_path).talker).eval()


def get_talker_cache_io_config(model=None):
    return {
        "input_names": ["inputs_embeds", "position_ids", "attention_mask", "past_kv"],
        "output_names": ["logits", "hidden_states", "present"],
        "dynamic_axes": {
            "inputs_embeds": {0: "batch", 1: "cur"},
            "position_ids": {1: "batch", 2: "cur"},
            "attention_mask": {0: "batch", 1: "total"},
        },
        "dynamic_shapes": _talker_cache_dynamic_shapes(),
    }


def _talker_cache_dynamic_shapes(n_layers=None):
    # `past_kv` is a plain list of [k,v] tensor pairs (NOT a DynamicCache), so a nested-list
    # dynamic_shapes matches the input pytree directly — no DynamicCache pytree conversion.
    n_layers = n_layers or _DIMS["n_layers"]
    return {
        "inputs_embeds": {0: "batch", 1: "cur"},
        "position_ids": {1: "batch", 2: "cur"},
        "attention_mask": {0: "batch", 1: "total"},
        "past_kv": [[{2: "past"}, {2: "past"}] for _ in range(n_layers)],
    }


def get_talker_cache_dummy_inputs(model_path=None):
    L, n_kv, hd, H = _DIMS["n_layers"], _DIMS["n_kv"], _DIMS["head_dim"], _DIMS["hidden"]
    cur, past = 8, 4
    past_kv = [[torch.randn(1, n_kv, past, hd, dtype=torch.float32),
                torch.randn(1, n_kv, past, hd, dtype=torch.float32)] for _ in range(L)]
    return {
        "inputs_embeds": torch.randn(1, cur, H, dtype=torch.float32),
        "position_ids": torch.arange(past, past + cur).view(1, 1, -1).expand(3, 1, -1).contiguous(),
        "attention_mask": torch.ones(1, past + cur, dtype=torch.long),
        "past_kv": past_kv,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOKENIZER: encode (waveform → codes) / decode (codes → waveform)
# ════════════════════════════════════════════════════════════════════════════

class TokEncoderWrapper(nn.Module):
    """audio [B,1,T] → codes [B, T_frames, 16].

    Mirrors MimiModel._encode_frame on tok.encoder (a MimiModel), bypassing the
    high-level model.encode() — whose streaming/padding bookkeeping (padding_cache,
    get_audio_codes_mask) uses data-dependent ops that torch.export can't trace and
    which crash inside Mimi.encode. Explicit attention_mask avoids create_causal_mask's
    untraceable path; export with the dynamo exporter + static input length.
    """
    def __init__(self, tok):
        super().__init__()
        self.enc = tok.encoder                       # Qwen3TTSTokenizerV2Encoder(MimiModel)
        self.valid = tok.encoder_valid_num_quantizers   # 16

    def forward(self, audio):                        # [B, 1, T] float32 @ 24 kHz
        emb = self.enc.encoder(audio)                # [B, dim, frames]
        hidden = emb.transpose(1, 2)                 # [B, frames, dim]
        attn = torch.ones(hidden.shape[0], hidden.shape[1], dtype=torch.long)
        out = self.enc.encoder_transformer(
            hidden, attention_mask=attn, use_cache=False, return_dict=True)
        emb = out[0].transpose(1, 2)                 # [B, dim, frames]
        emb = self.enc.downsample(emb)               # [B, dim, frames]
        codes = self.enc.quantizer.encode(emb)       # [nq, B, frames]
        codes = codes.transpose(0, 1)[:, :self.valid]  # [B, 16, frames]
        return codes.transpose(1, 2)                 # [B, frames, 16]  (matches decoder)


class TokDecoderWrapper(nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, audio_codes):              # [B, T, nq] int64
        # Bypass model.decode's chunked_decode (Python while-loop, untraceable):
        # clamp + transpose to [B, nq, T] and call the decoder directly.
        codes = torch.clamp(audio_codes, min=0).transpose(1, 2)   # [B, nq, T]
        wav = self.tok.decoder(codes)            # Decoder.forward (single pass)
        return wav


def get_tok_encoder_model(model_path=None):
    _patch_mimi_static_padding()          # concrete conv lengths for torch.export
    tok = _intify_mimi_convs(_load_tokenizer(model_path))
    return TokEncoderWrapper(tok).eval()


def get_tok_encoder_io_config(model=None):
    return {
        "input_names": ["audio"],
        "output_names": ["audio_codes"],
        # static sample length (Mimi conv padding is data-dependent under dynamo);
        # only batch is dynamic. Feed fixed 1-second (24000-sample) chunks.
        "input_shapes": [[1, 1, 24000]],
        "input_types": ["float32"],
        "dynamic_axes": {"audio": {0: "batch"},
                         "audio_codes": {0: "batch"}},
    }


def get_tok_encoder_dummy_inputs(model=None):
    return {"audio": torch.randn(1, 1, 24000, dtype=torch.float32)}


def get_tok_decoder_model(model_path=None):
    return TokDecoderWrapper(_load_tokenizer(model_path)).eval()


def get_tok_decoder_io_config(model=None):
    # model.decode wants codes (batch, codes_length, num_quantizers) = [B, T, 16]
    return {
        "input_names": ["audio_codes"],
        "output_names": ["waveform"],
        "input_shapes": [[1, 25, 16]],
        "input_types": ["int64"],
        "dynamic_axes": {"audio_codes": {0: "batch", 1: "frames"},
                         "waveform": {0: "batch", 2: "samples"}},
    }


def get_tok_decoder_dummy_inputs(model=None):
    return {"audio_codes": torch.randint(0, 2048, (1, 25, 16), dtype=torch.int64)}


# ════════════════════════════════════════════════════════════════════════════
# SPEAKER ENCODER (Base model only): reference audio → x-vector for voice cloning.
# Folds the mel front-end (librosa filterbank + torch.stft, n_fft 1024 / hop 256 /
# 128 mels / fmax 12000) into the graph so inference feeds raw 24 kHz audio.
# Mirrors Qwen3TTSForConditionalGeneration.extract_speaker_embedding (modeling L1941).
# ════════════════════════════════════════════════════════════════════════════

class SpeakerEncoderWrapper(nn.Module):
    """audio [B, T] (mono, 24 kHz float32) → speaker x-vector [B, D].

    Reimplements `mel_spectrogram` (modeling L399) inline WITHOUT its data-dependent
    debug branches (`if torch.min(y) < -1.0: print(...)`) which break torch.export
    (GuardOnDataDependentSymNode). Numerically identical: same librosa mel filterbank,
    reflect pad, hann STFT, magnitude, log compression (clamp 1e-5).
    """
    _NFFT, _HOP, _WIN, _MELS, _FMIN, _FMAX, _SR = 1024, 256, 1024, 128, 0, 12000, 24000

    def __init__(self, model):
        super().__init__()
        if getattr(model, "speaker_encoder", None) is None:
            raise ValueError("model has no speaker_encoder (not a `base` checkpoint)")
        self.speaker_encoder = model.speaker_encoder
        from librosa.filters import mel as librosa_mel_fn
        mb = librosa_mel_fn(sr=self._SR, n_fft=self._NFFT, n_mels=self._MELS,
                            fmin=self._FMIN, fmax=self._FMAX)
        self.register_buffer("mel_basis", torch.from_numpy(mb).float(), persistent=False)
        self.register_buffer("hann", torch.hann_window(self._WIN), persistent=False)

    def forward(self, audio):                         # audio [B, T]
        pad = (self._NFFT - self._HOP) // 2
        y = torch.nn.functional.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(y, self._NFFT, hop_length=self._HOP, win_length=self._WIN,
                          window=self.hann, center=False, pad_mode="reflect",
                          normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
        mel = torch.matmul(self.mel_basis, spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))   # dynamic_range_compression
        mels = mel.transpose(1, 2)                    # [B, T_mel, 128]
        return self.speaker_encoder(mels)[0]


def get_speaker_encoder_model(model_path=None):
    return SpeakerEncoderWrapper(_load_tts(model_path)).eval()


def get_speaker_encoder_io_config(model=None):
    return {
        "input_names": ["audio"], "output_names": ["speaker_embedding"],
        "input_shapes": [[1, 144000]], "input_types": ["float32"],   # 6 s dummy; dynamic below
        "dynamic_axes": {"audio": {0: "batch", 1: "samples"},
                         "speaker_embedding": {0: "batch"}},
        "dynamic_shapes": {"audio": {0: "batch", 1: "samples"}},      # dynamo needs this
    }


def get_speaker_encoder_dummy_inputs(model=None):
    return {"audio": torch.randn(1, 144000, dtype=torch.float32)}
