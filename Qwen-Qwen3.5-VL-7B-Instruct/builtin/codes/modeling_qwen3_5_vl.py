#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file is adapted from modeling_qwen3_vl.py for Qwen3.5-VL ONNX export.
#           Vision encoder architecture is identical to Qwen3-VL. The text decoder
#           uses Hybrid Gated DeltaNet and is NOT exported here — ModelBuilder (text.json)
#           handles it. Only embed_tokens is included for the embedding sub-model.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#
# Architecture differences vs Qwen3-VL:
#   - Text decoder: Hybrid Gated DeltaNet (3 DeltaNet : 1 standard attention)
#     with torch_chunk_gated_delta_rule — exported by ModelBuilder only.
#   - Vision encoder: IDENTICAL (same ViT, absolute pos embed, rotary, merger).
#   - Preprocessing: patch_size=16, spatial_merge_size=2, temporal_patch_size=2.
#     mean/std=[0.5,0.5,0.5]; min_pixels/max_pixels loaded from preprocessor_config.
#
# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

logger = logging.get_logger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


# =============================================================================
# Vision encoder — identical architecture to Qwen3-VL
# =============================================================================

class Qwen3_5VLVisionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3_5VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3_5VLVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, seqlen) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class Qwen3_5VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        merge_size = config.spatial_merge_size
        self.hidden_size = config.hidden_size * (merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3_5VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = 0.0
        self.is_causal = False
        self._attn_implementation = getattr(config, "_attn_implementation", "sdpa")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        if getattr(torch.compiler, "is_exporting", lambda: False)():
            # ONNX export path: emit custom PackedAttention op.
            # Olive's PackedAttentionToLoopMHA graph surgery rewrites this node
            # into a loop-based MHA that ONNX Runtime GenAI can execute.
            attn_output = torch.onnx.ops.symbolic(
                "custom::PackedAttention",
                (query_states, key_states, value_states, cu_seqlens),
                dict(scale=self.scaling, num_heads=self.num_heads),
                dtype=query_states.dtype,
                shape=(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[1],
                    query_states.shape[3],
                ),
                version=1,
            )
            attn_output = attn_output.to(self.proj.weight.device)
        else:
            # Non-export: chunked SDPA over individual images in the packed sequence
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(t, lengths.tolist(), dim=2)
                for t in (query_states, key_states, value_states)
            ]
            attn_outputs = []
            for q, k, v in zip(*splits):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None,
                    dropout_p=0.0, scale=self.scaling, is_causal=False,
                )
                attn_outputs.append(out.transpose(1, 2))
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class Qwen3_5VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3_5VLVisionAttention(config=config)
        self.mlp = Qwen3_5VLVisionMLP(config=config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings, **kwargs,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3_5VLVisionTransformerModel(nn.Module):
    """
    Qwen3.5-VL vision encoder for ONNX export.

    Identical architecture to Qwen3-VL:
      patch_embed (Conv3d) → absolute pos embed (bilinear interpolation) →
      rotary pos embed → ViT blocks (PackedAttention) → PatchMerger.

    The deepstack_merger_list weights are loaded from the checkpoint (strict=False)
    but are NOT invoked during ONNX export — the text model handles deepstack features.
    """

    def __init__(self, vision_config) -> None:
        super().__init__()
        self.config = vision_config
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.patch_size = vision_config.patch_size

        self.patch_embed = Qwen3_5VLVisionPatchEmbed(config=vision_config)

        # Absolute positional embedding (num_position_embeddings = grid_side^2)
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)
        self.pos_embed = nn.Embedding(self.num_position_embeddings, vision_config.hidden_size)

        # Rotary positional embedding
        head_dim = vision_config.hidden_size // vision_config.num_heads
        self.rotary_pos_emb = Qwen3_5VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen3_5VLVisionBlock(vision_config) for _ in range(vision_config.depth)]
        )
        self.merger = Qwen3_5VLVisionPatchMerger(config=vision_config, use_postshuffle_norm=False)

        # DeepStack mergers: loaded from checkpoint but not used during ONNX export
        deepstack_indexes = getattr(vision_config, "deepstack_visual_indexes", [8, 16, 24])
        self.deepstack_visual_indexes = deepstack_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_5VLVisionPatchMerger(config=vision_config, use_postshuffle_norm=True)
                for _ in range(len(deepstack_indexes))
            ]
        )
        self.gradient_checkpointing = False

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Vectorised bilinear interpolation of learnable 2-D position embeddings.

        Uniform-grid assumption: all images in a call share the same (t, h, w).
        This keeps num_images symbolic during Dynamo export.
        """
        merge_size = self.spatial_merge_size
        dev = self.pos_embed.weight.device
        dtype = self.pos_embed.weight.dtype
        n = self.num_grid_per_side

        num_images = grid_thw.shape[0]   # symbolic when exporting dynamic shape
        t = grid_thw[0, 0]
        h = grid_thw[0, 1]
        w = grid_thw[0, 2]

        torch._check(t.item() >= 1)
        torch._check(h.item() >= 2)
        torch._check(w.item() >= 2)

        # Evenly-spaced sample positions in [0, n-1]
        h_idxs = torch.arange(h, dtype=torch.float32, device=dev) * ((n - 1) / (h - 1))
        w_idxs = torch.arange(w, dtype=torch.float32, device=dev) * ((n - 1) / (w - 1))

        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_floor + 1).clamp(max=n - 1)
        w_ceil = (w_floor + 1).clamp(max=n - 1)

        dh = (h_idxs - h_floor.float()).to(dtype)
        dw = (w_idxs - w_floor.float()).to(dtype)

        base_h  = h_floor.long() * n
        base_hc = h_ceil.long()  * n

        idx_00 = (base_h[:, None]  + w_floor.long()[None]).reshape(-1)
        idx_01 = (base_h[:, None]  + w_ceil.long()[None]).reshape(-1)
        idx_10 = (base_hc[:, None] + w_floor.long()[None]).reshape(-1)
        idx_11 = (base_hc[:, None] + w_ceil.long()[None]).reshape(-1)

        wt_00 = ((1.0 - dh)[:, None] * (1.0 - dw)[None]).reshape(-1)
        wt_01 = ((1.0 - dh)[:, None] * dw[None]).reshape(-1)
        wt_10 = (dh[:, None]          * (1.0 - dw)[None]).reshape(-1)
        wt_11 = (dh[:, None]          * dw[None]).reshape(-1)

        pos = (
            self.pos_embed(idx_00) * wt_00[:, None]
            + self.pos_embed(idx_01) * wt_01[:, None]
            + self.pos_embed(idx_10) * wt_10[:, None]
            + self.pos_embed(idx_11) * wt_11[:, None]
        )  # [h*w, hidden_size]

        # Repeat across temporal frames then apply merge-block permutation
        pos = pos.repeat(t, 1)
        pos = (
            pos.reshape(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        return pos.repeat(num_images, 1)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Vectorised rotary position embeddings.

        Uniform-grid assumption: all images share the same (t, h, w).
        """
        merge_size = self.spatial_merge_size
        max_hw = grid_thw[:, 1:].max()
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        num_images = grid_thw.shape[0]
        num_frames = grid_thw[0, 0]
        height = grid_thw[0, 1]
        width = grid_thw[0, 2]
        merged_h, merged_w = height // merge_size, width // merge_size

        torch._check(merged_h.item() >= 1)
        torch._check(merged_w.item() >= 1)
        torch._check(num_frames.item() >= 1)

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row  = torch.arange(merge_size, device=device)
        intra_col  = torch.arange(merge_size, device=device)

        row_idx = (
            block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
        ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
        col_idx = (
            block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
        ).expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)
        coords = coords.repeat(num_frames, 1)
        single_emb = freq_table[coords].flatten(1)
        return single_emb.repeat(num_images, 1)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states: (total_patches, channels_per_patch) raw pixel patches.
            grid_thw: (num_images, 3) — temporal, height, width per image.
        Returns:
            Merged image features (total_logical_patches, out_hidden_size).
        """
        hidden_states = self.patch_embed(hidden_states)

        # Add absolute positional embedding (bilinear interpolation from learned table)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Compute rotary position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # cu_seqlens: boundaries of each image's patches in the packed sequence.
        # Uniform-grid: all images share (t, h, w), so tile [h*w] t*num_images times.
        hw0 = grid_thw[0, 1] * grid_thw[0, 2]
        t0  = grid_thw[0, 0]
        num_images_fwd = grid_thw.shape[0]
        cu_vals = hw0.unsqueeze(0).expand(t0 * num_images_fwd)
        cu_seqlens = cu_vals.cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # NOTE: DeepStack features are NOT exported for ONNX.
        # The text model builder (ModelBuilder in text.json) handles them internally.
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return self.merger(hidden_states)


# =============================================================================
# Minimal text embedding — only embed_tokens is needed for ONNX export
# =============================================================================

class Qwen3_5VLTextEmbedding(nn.Module):
    """
    Thin wrapper around embed_tokens.

    Qwen3.5-VL's text decoder uses Hybrid Gated DeltaNet (torch_chunk_gated_delta_rule),
    which creates very large ONNX graphs and is NOT exported through this file.
    ModelBuilder handles the full text decoder via text.json.  Only embed_tokens is
    needed here for the embedding sub-model export (get_fused_input_embeddings).
    """

    def __init__(self, text_config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


# =============================================================================
# Full VL model wrapper
# =============================================================================

class Qwen3_5VLModel(nn.Module):
    """
    Qwen3.5-VL model wrapper for Olive ONNX export.

    Exposes two sub-model entry points (swapped via forward/method exchange in user_script.py):
      - get_image_features(pixel_values, image_grid_thw) -> merged vision features
      - get_fused_input_embeddings(input_ids, image_features) -> fused text+vision embeddings

    The text decoder (Gated DeltaNet) is NOT included here — it is exported separately
    by ModelBuilder via the text.json Olive config.

    Checkpoint key remapping (Qwen3.5-VL HF format):
      HF saves keys as `model.visual.*` and `model.language_model.*`.
      user_script._load_base_model strips the leading `model.` prefix before
      calling load_state_dict(strict=False), so our `visual.*` and
      `language_model.embed_tokens.*` keys are populated correctly.
      All other language_model.* keys (DeltaNet layers etc.) are UNEXPECTED and
      safely ignored due to strict=False.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3_5VLVisionTransformerModel(config.vision_config)
        self.language_model = Qwen3_5VLTextEmbedding(config.text_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Encode raw pixel patches into merged vision features."""
        pixel_values = pixel_values.type(self.visual.dtype)
        return self.visual(pixel_values, grid_thw=image_grid_thw)

    def get_fused_input_embeddings(
        self,
        input_ids: torch.LongTensor,
        image_features: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """
        Fuse text token embeddings with image features.

        Image token positions in input_ids (marked with image_token_id) are replaced
        by the corresponding rows of image_features via masked_scatter.
        """
        image_token_id = self.config.image_token_id
        vocab_size = self.config.text_config.vocab_size

        # Clamp any out-of-vocabulary image token IDs before embedding lookup
        def true_fn(input_ids):
            llm_input_ids = input_ids.clone()
            llm_input_ids[input_ids == image_token_id] = 0
            return llm_input_ids

        def false_fn(input_ids):
            return input_ids

        llm_input_ids = torch.cond(
            image_token_id >= vocab_size,
            true_fn,
            false_fn,
            (input_ids,),
        )

        inputs_embeds = self.language_model.get_input_embeddings()(llm_input_ids)

        if image_features is not None:
            special_image_mask = (llm_input_ids == image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_features: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Standard forward (this method is swapped with get_image_features or
        get_fused_input_embeddings in user_script.py before ONNX export)."""
        if pixel_values is not None and image_grid_thw is not None:
            return self.get_image_features(pixel_values, image_grid_thw)
        return self.get_fused_input_embeddings(input_ids, image_features)
