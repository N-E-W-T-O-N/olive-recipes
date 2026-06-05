"""
OmniVoice sub-model wrappers for ONNX export via Olive.

OmniVoice backbone consists of three SEPARATE models:

  audio_embeddings_encoder
      (B, C=8, S) input_ids  +  (B, S) audio_mask
          →  (B, S, H=1024) inputs_embeds
      Combines Qwen3 text-token embeddings with OmniVoice audio-token
      embeddings. Replicates OmniVoice._prepare_embed_inputs().

  llm_decoder  [exported via ModelBuilder, NOT these wrappers]
      (B, S, H=1024) inputs_embeds  →  (B, S, H=1024) hidden_states
      Qwen3 28-layer backbone.  ModelBuilder handles this with
          exclude_embeds=True   → accepts inputs_embeds directly
          exclude_lm_head=True  → outputs hidden_states (not logits)

  audio_heads_decoder
      (B, S, H=1024) hidden_states
          →  (B, C=8, S, V=1025) logits
      nn.Linear(H → C*V) + reshape. Projects LLM hidden states to
      per-codebook audio-token logits.

Architecture constants (from config.json):
  HIDDEN_SIZE     = 1024
  NUM_CODEBOOKS   = 8
  AUDIO_VOCAB     = 1025   (1024 real + 1 MASK at index 1024)
"""

import torch
import torch.nn as nn

HIDDEN_SIZE   = 1024
NUM_CODEBOOKS = 8
AUDIO_VOCAB   = 1025


class AudioEmbeddingsEncoderWrapper(nn.Module):
    """
    Replicates OmniVoice._prepare_embed_inputs().

    At text positions (audio_mask=False):  uses the Qwen3 text embedding.
    At audio positions (audio_mask=True):  sums the 8 per-codebook audio
    embeddings (shifted by codebook_layer_offsets) and uses that sum.

    Inputs
    ------
    input_ids  : (B, C, S)  int64  — C=8 codebook IDs; row 0 = text tokens
    audio_mask : (B, S)     bool   — True at audio (codec) positions

    Output
    ------
    inputs_embeds : (B, S, H)  float32
    """

    def __init__(
        self,
        text_embed:    nn.Embedding,   # Qwen3 text embedding (vocab_size, H)
        audio_embed:   nn.Embedding,   # OmniVoice audio embedding (total_audio_vocab, H)
        layer_offsets: torch.Tensor,   # (C,) int64 codebook offset per layer
    ):
        super().__init__()
        self.text_embed  = text_embed
        self.audio_embed = audio_embed
        self.register_buffer("layer_offsets", layer_offsets.clone())

    def forward(
        self,
        input_ids:  torch.Tensor,   # (B, C, S)  int64
        audio_mask: torch.Tensor,   # (B, S)     bool
    ) -> torch.Tensor:              # (B, S, H)
        # Text embeddings (using row 0 of input_ids)
        text_e = self.text_embed(input_ids[:, 0, :])

        # Audio embeddings: shift each codebook row by its layer offset,
        # look up, sum across C dimension
        shifted = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.layer_offsets.view(1, -1, 1)
        audio_e = self.audio_embed(shifted).sum(dim=1)

        # Select text or audio based on audio_mask
        return torch.where(audio_mask.unsqueeze(-1), audio_e, text_e)


class AudioHeadsDecoderWrapper(nn.Module):
    """
    Projects LLM hidden states to per-codebook audio-token logits.

    This is a SEPARATE component from the Qwen3 LLM.  The LLM outputs
    hidden_states; this module maps them to (B, C, S, V) logits for
    OmniVoice's 8-codebook iterative-unmasking decoding.

    Input  : hidden_states  (B, S, H=1024)
    Output : logits         (B, C=8, S, V=1025)
    """

    def __init__(
        self,
        heads:  nn.Linear,
        num_cb: int = NUM_CODEBOOKS,
        vocab:  int = AUDIO_VOCAB,
    ):
        super().__init__()
        self.heads  = heads
        self.num_cb = num_cb
        self.vocab  = vocab

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, H)
        B, S, _ = hidden_states.shape
        flat = self.heads(hidden_states)                        # (B, S, C*V)
        return flat.view(B, S, self.num_cb, self.vocab).permute(0, 2, 1, 3)
        # → (B, C, S, V)


# =============================================================================
# Higgs Audio V2 Tokenizer wrappers
# =============================================================================
# The HiggsAudioV2TokenizerModel encodes/decodes audio at two sample rates:
#   24 kHz (acoustic DAC encoder/decoder)
#   16 kHz (HuBERT semantic encoder)
#
# All four components are exported as separate ONNX models so inference can
# pipeline them independently.  Pre-tracing and weight_norm stripping are
# handled in user_script.py before the wrappers are passed to Olive.
#
# Audio constants:
#   SR_24K          = 24 000 Hz  (acoustic + codec decoder sample rate)
#   SR_16K          = 16 000 Hz  (semantic / HuBERT sample rate)
#   DOWNSAMPLE_FACTOR = 320      (overall samples → codec frames ratio at 24 kHz)
#   N_CODEBOOKS     = 8          (number of RVQ quantizer codebooks exported)

SR_24K            = 24_000
SR_16K            = 16_000
DOWNSAMPLE_FACTOR = 320
HIGGS_N_CB        = 8     # number of codebooks in the Higgs quantizer
HIGGS_CB_SIZE     = 1024  # RVQ codebook size


def _strip_weight_norm(module: "nn.Module") -> None:
    """Recursively remove weight_norm parametrizations and re-register as nn.Parameter.

    Why this matters
    ----------------
    weight_norm stores the normalised weight as a computed tensor derived from
    weight_g and weight_v.  After removal, the computed tensor is set back as
    `sub.weight` but may still be a *plain tensor attribute* (not nn.Parameter)
    with requires_grad=True.  torch.onnx.export treats plain tensor attributes
    as graph constants and crashes when they require grad.
    Re-registering as nn.Parameter(weight.detach()) fixes both issues.
    """
    import torch.nn as nn
    for sub in module.modules():
        stripped = False
        try:
            from torch.nn.utils.parametrize import remove_parametrizations
            if hasattr(sub, "parametrizations") and "weight" in getattr(sub, "parametrizations", {}):
                remove_parametrizations(sub, "weight", leave_parametrized=True)
                stripped = True
        except Exception:
            pass
        if not stripped:
            try:
                from torch.nn.utils import remove_weight_norm
                remove_weight_norm(sub)
                stripped = True
            except (ValueError, AttributeError):
                pass
        if stripped and hasattr(sub, "weight") and isinstance(sub.weight, torch.Tensor):
            sub.weight = nn.Parameter(sub.weight.detach())

    # Also detach any remaining plain tensor attributes with requires_grad=True
    module.requires_grad_(False)
    for sub in module.modules():
        for attr_name in list(vars(sub)):
            v = getattr(sub, attr_name, None)
            if (isinstance(v, torch.Tensor)
                    and not isinstance(v, nn.Parameter)
                    and v.requires_grad):
                setattr(sub, attr_name, v.detach())


class HiggsAcousticEncoderWrapper(nn.Module):
    """DAC acoustic encoder: waveform_24k → acoustic_features.

    Input  : waveform_24k  (B, 1, T_samples)  float32  [24 kHz audio]
    Output : acoustic_feat (B, D_acoustic, T_frames)  float32

    Temporal dimension is dynamic (any audio length).

    Uses tok.acoustic_encoder (transformers: self.acoustic_encoder = AutoModel from
    config.acoustic_model_config — the DAC encoder nn.Module).
    NOT tok.encode which is the full encode method (acoustic+semantic → codes).

    weight_norm is stripped via _prepare_tok(tok) on the full tokenizer before
    this wrapper is created.  _strip_weight_norm is NOT called on encoder inside
    __init__ to avoid issues if encoder is a bound method in other implementations.

    Do NOT pre-trace before returning to Olive: ScriptModule export fails with
    "args contained None's after flattening".  Olive's OnnxConversion traces
    the returned nn.Module via torch.onnx.export, correctly resolving the
    Python branches (if channels != 1, if padding > 0) to constants.
    """

    def __init__(self, encoder) -> None:
        super().__init__()
        # encoder is tok.acoustic_encoder — an nn.Module.
        # weight_norm already stripped via _prepare_tok() in the loader function.
        self.encoder = encoder

    def forward(self, waveform_24k: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform_24k)


class HiggsSemanticEncoderWrapper(nn.Module):
    """HuBERT semantic encoder: waveform_16k → semantic_features.

    Input  : waveform_16k  (B, T_samples)  float32  [16 kHz audio, no channel dim]
    Output : semantic_feat (B, D_semantic, T_frames)  float32

    Runs HuBERT then passes through encoder_semantic conv layers.
    """

    def __init__(self, semantic_model: "nn.Module", encoder_semantic: "nn.Module") -> None:
        super().__init__()
        _strip_weight_norm(semantic_model)
        _strip_weight_norm(encoder_semantic)
        self.semantic_model   = semantic_model
        self.encoder_semantic = encoder_semantic

    def forward(self, waveform_16k: torch.Tensor) -> torch.Tensor:
        out    = self.semantic_model(waveform_16k)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.encoder_semantic(hidden.transpose(1, 2))


class HiggsQuantizerEncoderWrapper(nn.Module):
    """RVQ encoder: acoustic + semantic features → discrete codes.

    Inputs : acoustic_feat  (B, D_acoustic, T_frames)  float32
             semantic_feat  (B, D_semantic, T_frames)  float32
    Output : codes          (N_Q=8, B, T_frames)       int64

    The two feature streams are concatenated along the channel dimension
    (merge_mode='concat') or summed (merge_mode='add') before the RVQ step.
    """

    def __init__(
        self,
        fc_prior:   "nn.Module",
        quantizer:  "nn.Module",
        merge_mode: str = "concat",
    ) -> None:
        super().__init__()
        _strip_weight_norm(fc_prior)
        self.fc_prior  = fc_prior
        self.quantizer = quantizer
        self.merge_mode = merge_mode

    def forward(
        self,
        acoustic_feat: torch.Tensor,   # (B, D_a, T)
        semantic_feat: torch.Tensor,   # (B, D_s, T)
    ) -> torch.Tensor:                 # (N_Q, B, T)
        if self.merge_mode == "concat":
            merged = torch.cat([acoustic_feat, semantic_feat], dim=1)
        else:
            merged = acoustic_feat + semantic_feat
        z = self.fc_prior(merged.transpose(1, 2)).transpose(1, 2)
        return self.quantizer.encode(z)


class HiggsDecoderWrapper(nn.Module):
    """RVQ + DAC decoder: codes → waveform_24k.

    Input  : codes         (N_Q=8, B, T_frames)  int64
    Output : waveform_24k  (B, 1, T_samples)      float32  [24 kHz]

    Runs RVQ decode → projection → DAC decoder.
    weight_norm is stripped in __init__ on decoder and fc_post2.

    Transformers HiggsAudioV2TokenizerModel attribute mapping:
      boson_multimodal  →  transformers
      fc_post2          →  tok.fc2              (Linear: RVQ output → DAC input)
      decoder           →  tok.acoustic_decoder (DAC decoder nn.Module)
      quantizer         →  tok.quantizer        (same name)

    Pass the correct attributes from the loader:
        HiggsDecoderWrapper(tok.quantizer, tok.fc2, tok.acoustic_decoder)
    """

    def __init__(
        self,
        quantizer: "nn.Module",
        fc_post2:  "nn.Module",   # tok.fc2 in transformers
        decoder:   "nn.Module",   # tok.acoustic_decoder in transformers
    ) -> None:
        super().__init__()
        _strip_weight_norm(decoder)
        _strip_weight_norm(fc_post2)
        self.quantizer = quantizer
        self.fc_post2  = fc_post2
        self.decoder   = decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:  # (N_Q, B, T) → (B, 1, T_audio)
        z_q = self.quantizer.decode(codes)
        z_a = self.fc_post2(z_q.transpose(1, 2)).transpose(1, 2)
        return self.decoder(z_a)
