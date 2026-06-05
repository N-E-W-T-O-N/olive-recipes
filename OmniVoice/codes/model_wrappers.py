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
