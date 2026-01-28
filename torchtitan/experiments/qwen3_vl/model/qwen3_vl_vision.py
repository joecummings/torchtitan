# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL Vision Encoder implementation (images only).

Key components:
- Conv2D patch embedding
- 2D RoPE for vision (different from text MRoPE)
- DeepStack feature extraction at intermediate layers
- Patch merger with 2x2 spatial merge
"""

import einops as E
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
)
from torchtitan.protocols.model import AttentionMasksType

from .args import Qwen3VLVisionArgs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to vision queries and keys."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLVisionPatchEmbed(nn.Module):
    """
    Conv2D patch embedding for Qwen3 VL (images only).

    Projects image patches to embedding dimension.
    """

    def __init__(self, args: Qwen3VLVisionArgs) -> None:
        super().__init__()
        self.patch_size = args.patch_size
        self.in_channels = args.in_channels
        self.embed_dim = args.hidden_size

        self.proj = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Patchified pixel values of shape
                (total_patches, in_channels * patch_size * patch_size)

        Returns:
            Embedded patches of shape (total_patches, embed_dim)
        """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states

    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """
    2D Rotary Position Embedding for vision encoder.

    Uses theta=10000 (different from text model's theta).
    Applied to height/width grids.
    """

    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary frequencies for given sequence length."""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

    def init_weights(self):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.inv_freq.copy_(inv_freq)


class Qwen3VLVisionMLP(nn.Module):
    """Vision encoder MLP (GELU activation)."""

    def __init__(self, args: Qwen3VLVisionArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.intermediate_size
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_state), approximate="tanh"))

    def init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


class Qwen3VLVisionAttention(nn.Module):
    """
    Vision encoder attention with combined QKV projection.

    No GQA in vision encoder (all heads are full attention).
    """

    def __init__(self, args: Qwen3VLVisionArgs) -> None:
        super().__init__()
        self.dim = args.hidden_size
        self.num_heads = args.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        # Combined QKV projection
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

        self.inner_attention = FlexAttentionWrapper()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Shape (seq_len, hidden_size) - flattened sequence
            cu_seqlens: Cumulative sequence lengths for variable length attention
            position_embeddings: Tuple of (cos, sin) for rotary embeddings

        Returns:
            Output tensor of shape (seq_len, hidden_size)
        """
        seq_length = hidden_states.shape[0]

        # Combined QKV projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # (3, seq_len, num_heads, head_dim)
        query_states, key_states, value_states = qkv.unbind(0)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        # Reshape for attention: (1, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # Process chunks based on cu_seqlens for variable length sequences
        # For simplicity, we process as a single batch with masking
        if attention_masks is not None and isinstance(attention_masks, BlockMask):
            attn_output = self.inner_attention(
                query_states,
                key_states,
                value_states,
                block_mask=attention_masks,
                scale=self.scaling,
            )
        else:
            # Fallback to SDPA
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                scale=self.scaling,
            )

        attn_output = attn_output.squeeze(0).transpose(0, 1)  # (seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

    def init_weights(self):
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class Qwen3VLVisionBlock(nn.Module):
    """Vision encoder transformer block (pre-norm architecture)."""

    def __init__(self, args: Qwen3VLVisionArgs) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.attn = Qwen3VLVisionAttention(args)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.mlp = Qwen3VLVisionMLP(args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            attention_masks=attention_masks,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def init_weights(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.init_weights()
        self.mlp.init_weights()


class Qwen3VLVisionPatchMerger(nn.Module):
    """
    Patch merger with 2x2 spatial merge.

    Reduces spatial resolution by merging 2x2 patches into 1,
    then projects to text model dimension.
    """

    def __init__(
        self,
        args: Qwen3VLVisionArgs,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_merge_size = args.spatial_merge_size
        self.hidden_size = args.hidden_size * (args.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else args.hidden_size,
            eps=args.layer_norm_eps,
        )
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, args.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states of shape (seq_len, hidden_size)

        Returns:
            Merged and projected features of shape (seq_len // 4, out_hidden_size)
        """
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x = self.fc2(F.gelu(self.fc1(x)))
        return x

    def init_weights(self):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)


class Qwen3VLVisionEncoder(nn.Module):
    """
    Complete Qwen3 VL Vision Encoder with DeepStack (images only).

    Features:
    - Conv2D patch embedding
    - Learned 2D position embeddings with interpolation
    - 2D RoPE for attention
    - DeepStack: extracts features at intermediate layers [8, 16, 24]
    - Final patch merger for projection to text dimension
    """

    def __init__(self, args: Qwen3VLVisionArgs) -> None:
        super().__init__()
        self.args = args
        self.spatial_merge_size = args.spatial_merge_size
        self.patch_size = args.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        # Patch embedding (Conv2D for images)
        self.patch_embed = Qwen3VLVisionPatchEmbed(args)

        # Learned position embeddings
        self.pos_embed = nn.Embedding(args.num_position_embeddings, args.hidden_size)
        self.num_grid_per_side = int(args.num_position_embeddings ** 0.5)

        # Rotary position embedding for attention
        head_dim = args.hidden_size // args.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks
        self.layers = nn.ModuleDict(
            {str(idx): Qwen3VLVisionBlock(args) for idx in range(args.depth)}
        )

        # Final merger
        self.merger = Qwen3VLVisionPatchMerger(args, use_postshuffle_norm=False)

        # DeepStack mergers for intermediate features
        self.deepstack_visual_indexes = args.deepstack_visual_indexes
        self.deepstack_mergers = nn.ModuleDict(
            {
                str(idx): Qwen3VLVisionPatchMerger(args, use_postshuffle_norm=True)
                for idx in args.deepstack_visual_indexes
            }
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D rotary position embeddings for the vision features.

        For images, grid_thw has shape (num_images, 3) where [:, 0] = 1 (single frame).
        We compute 2D positions based on (height, width) dimensions only.
        """
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        # For images, t=1, so total tokens = sum(h * w) for each image
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for _, height, width in grid_thw:  # _ is t=1 for images
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # Compute full-resolution positions
            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Interpolate learned 2D position embeddings for variable image sizes.

        For images, grid_thw[:, 0] = 1 (single frame per image).
        Uses bilinear interpolation for position embeddings.
        """
        grid_hs, grid_ws = grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for h, w in zip(grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h.item())
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w.item())

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split(
            [h.item() * w.item() for h, w in zip(grid_hs, grid_ws)]
        )

        # Permute for spatial merge (for images, t=1)
        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, h, w in zip(patch_pos_embeds, grid_hs, grid_ws):
            pos_embed = (
                pos_embed.view(
                    h.item() // merge_size,
                    merge_size,
                    w.item() // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 2, 1, 3, 4)
                .flatten(0, 3)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """Generate attention masks for vision encoder."""
        grid_thw = extra_inputs["grid_thw"]

        # For vision encoder, we use causal attention within each image
        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        mask_mod = get_causal_mask_mod()

        return create_attention_mask(
            mask_mod,
            B=1,
            H=None,
            Q_LEN=total_tokens,
            KV_LEN=total_tokens,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Patchified pixel values
            grid_thw: Tensor of (num_images, 3) with temporal, height, width

        Returns:
            Tuple of:
                - merged_hidden_states: Final output projected to text dimension
                - deepstack_features: List of intermediate features from DeepStack layers
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Add position embeddings
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Compute cu_seqlens for variable length attention
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Forward through transformer layers with DeepStack
        deepstack_features = []
        for layer_idx, layer in self.layers.items():
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                attention_masks=attention_masks,
            )

            # Extract DeepStack features at specified layers
            layer_num = int(layer_idx)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_mergers[layer_idx](hidden_states)
                deepstack_features.append(deepstack_feature)

        # Final merger
        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states, deepstack_features

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize all weights."""
        self.patch_embed.init_weights()
        nn.init.normal_(self.pos_embed.weight)
        self.rotary_pos_emb.init_weights()

        for layer in self.layers.values():
            layer.init_weights()

        self.merger.init_weights()
        for merger in self.deepstack_mergers.values():
            merger.init_weights()
