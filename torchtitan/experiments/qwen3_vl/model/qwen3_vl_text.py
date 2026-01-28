# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL Text Model implementation.

Key features:
- MRoPE (Multi-dimensional RoPE) for vision-language
- Q/K normalization (Qwen3 specific)
- GQA (Grouped Query Attention)
- DeepStack integration points
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Qwen3VLModelArgs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3VLMRoPE(nn.Module):
    """
    Multi-dimensional Rotary Position Embedding for Qwen3 VL.

    Splits head_dim into 3 sections for temporal (t), height (h), and width (w).
    Uses interleaved layout: [THWTHWTHW...] instead of concatenated [TTT...HHH...WWW].

    Default section sizes: [24, 20, 20] for t, h, w (sum = 64 = head_dim/2)
    """

    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        mrope_section: list[int],
        theta: float = 500000.0,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.mrope_section = mrope_section
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Precompute inverse frequencies
        dim = head_dim  # Use full head_dim for frequency computation
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def compute_rope_cache(
        self,
        position_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cache for given position IDs.

        Args:
            position_ids: Shape (3, batch_size, seq_len) for t, h, w positions
                         or (batch_size, seq_len) for text-only

        Returns:
            Tuple of (cos, sin) tensors of shape (batch_size, seq_len, head_dim)
        """
        # Handle 2D position_ids (text-only case)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, -1, -1)

        # position_ids: (3, batch_size, seq_len)
        batch_size = position_ids.shape[1]
        seq_len = position_ids.shape[2]

        # Expand inv_freq: (3, batch_size, head_dim/2, 1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, batch_size, -1, 1)
            .to(device)
        )

        # position_ids_expanded: (3, batch_size, 1, seq_len)
        position_ids_expanded = position_ids[:, :, None, :].float().to(device)

        # freqs: (3, batch_size, head_dim/2, seq_len) -> transpose -> (3, batch_size, seq_len, head_dim/2)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)

        # Apply interleaved MRoPE
        freqs = self._apply_interleaved_mrope(freqs, self.mrope_section)

        # Duplicate for full head_dim
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos, sin

    def _apply_interleaved_mrope(
        self,
        freqs: torch.Tensor,
        mrope_section: list[int],
    ) -> torch.Tensor:
        """
        Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.

        Args:
            freqs: Shape (3, batch_size, seq_len, head_dim // 2)
            mrope_section: [t_size, h_size, w_size] frequency sections

        Returns:
            freqs_t: Shape (batch_size, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0].clone()  # Start with temporal frequencies

        # Interleave H and W into T's positions
        for dim_idx, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]

        return freqs_t

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for rotary embeddings.

        Args:
            x: Hidden states (used for dtype)
            position_ids: Position IDs, shape (3, batch_size, seq_len) or (batch_size, seq_len)

        Returns:
            Tuple of (cos, sin) tensors
        """
        cos, sin = self.compute_rope_cache(position_ids, x.device)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def init_weights(self):
        dim = self.head_dim
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.inv_freq.copy_(inv_freq)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to queries and keys.

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        cos: Cosine tensor of shape (batch, seq_len, head_dim)
        sin: Sine tensor of shape (batch, seq_len, head_dim)

    Returns:
        Tuple of rotated (xq, xk)
    """
    # Unsqueeze for broadcasting: (batch, seq_len, 1, head_dim)
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Qwen3VLTextAttention(nn.Module):
    """
    Qwen3 VL Text Attention with GQA and Q/K normalization.

    Key features:
    - Grouped Query Attention (GQA) with 8:1 ratio for 32B
    - Q/K RMSNorm applied AFTER projection, BEFORE RoPE
    - FlexAttention for efficient masking
    """

    q_norm: nn.RMSNorm | None
    k_norm: nn.RMSNorm | None

    def __init__(self, model_args: Qwen3VLModelArgs) -> None:
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attn_type = getattr(model_args, "attn_type", "flex")
        self.enable_gqa = self.n_heads > self.n_kv_heads

        # Q/K normalization (Qwen3 specific feature)
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Projections
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            model_args.dim, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        # Attention implementation
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        """
        Forward pass of attention.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cos: Cosine for RoPE of shape (batch, seq_len, head_dim)
            sin: Sine for RoPE of shape (batch, seq_len, head_dim)
            attention_masks: Attention mask

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Reshape to (batch, seq_len, n_heads, head_dim)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Apply Q/K normalization BEFORE RoPE (critical for Qwen3)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Compute attention
        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask), attention_masks
                output = (
                    self.inner_attention(
                        xq, xk, xv,
                        block_mask=attention_masks,
                        scale=self.scaling,
                        enable_gqa=self.enable_gqa,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv,
                    attention_masks,
                    scale=self.scaling,
                )
            case "sdpa":
                assert attention_masks is None
                output = (
                    self.inner_attention(
                        xq, xk, xv,
                        scale=self.scaling,
                        enable_gqa=self.enable_gqa,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class Qwen3VLTextMLP(nn.Module):
    """SwiGLU-style FFN for Qwen3 VL."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class Qwen3VLTextBlock(nn.Module):
    """
    Transformer decoder block for Qwen3 VL.

    Pre-norm architecture with optional DeepStack injection point.
    """

    def __init__(self, layer_id: int, model_args: Qwen3VLModelArgs) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Qwen3VLTextAttention(model_args)
        self.feed_forward = Qwen3VLTextMLP(
            dim=model_args.dim,
            hidden_dim=model_args.hidden_dim,
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            cos: Cosine for RoPE
            sin: Sine for RoPE
            attention_masks: Attention mask

        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        x = x + self.attention(
            self.attention_norm(x), cos, sin, attention_masks
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Qwen3VLTextModel(ModelProtocol):
    """
    Qwen3 VL Text Model (decoder).

    Features:
    - MRoPE for vision-language position encoding
    - GQA for efficient attention
    - Q/K normalization for training stability
    - DeepStack integration for multi-level vision features
    """

    def __init__(self, model_args: Qwen3VLModelArgs) -> None:
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.head_dim = model_args.head_dim

        # Token embeddings
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # MRoPE for position encoding
        self.mrope = Qwen3VLMRoPE(
            head_dim=model_args.head_dim,
            mrope_section=model_args.mrope_section,
            theta=model_args.mrope_theta,
            max_seq_len=model_args.max_seq_len,
        )

        # Transformer layers
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = Qwen3VLTextBlock(layer_id, model_args)

        # Final layer norm
        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Output projection
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        # DeepStack configuration
        self.num_deepstack_layers = len(model_args.encoder.deepstack_visual_indexes)

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize all weights."""
        buffer_device = buffer_device or torch.device("cpu")

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        self.mrope.init_weights()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device)

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim ** -0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """Generate FlexAttention masks."""
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods),
            B,
            None,
            input_batch.shape[1],
            input_batch.shape[1],
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """Generate attention masks based on attention type."""
        match self.model_args.attn_type:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.model_args.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal "
                        f"attention mask type, got {self.model_args.attn_mask_type}"
                    )
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        deepstack_features: list[torch.Tensor] | None = None,
        visual_pos_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through text model.

        Args:
            hidden_states: Token embeddings of shape (batch, seq_len, dim)
            position_ids: Position IDs of shape (3, batch, seq_len) or (batch, seq_len)
            attention_masks: Attention mask
            deepstack_features: List of vision features to inject at early layers
            visual_pos_mask: Boolean mask indicating visual token positions

        Returns:
            Output logits of shape (batch, seq_len, vocab_size)
        """
        # Compute MRoPE
        cos, sin = self.mrope(hidden_states, position_ids)

        # Forward through layers with DeepStack injection
        for layer_idx, layer in self.layers.items():
            hidden_states = layer(hidden_states, cos, sin, attention_masks)

            # Inject DeepStack features at early layers
            idx = int(layer_idx)
            if (
                deepstack_features is not None
                and visual_pos_mask is not None
                and idx < self.num_deepstack_layers
            ):
                hidden_states = self._inject_deepstack_features(
                    hidden_states,
                    visual_pos_mask,
                    deepstack_features[idx],
                )

        hidden_states = self.norm(hidden_states) if self.norm else hidden_states
        output = self.output(hidden_states) if self.output else hidden_states
        return output

    def _inject_deepstack_features(
        self,
        hidden_states: torch.Tensor,
        visual_pos_mask: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject DeepStack visual features into hidden states.

        Args:
            hidden_states: Shape (batch, seq_len, dim)
            visual_pos_mask: Boolean mask of shape (batch, seq_len)
            visual_embeds: Visual features to add

        Returns:
            Updated hidden states
        """
        visual_pos_mask = visual_pos_mask.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # Clone to avoid in-place modification issues with gradients
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_mask, :] = (
            hidden_states[visual_pos_mask, :] + visual_embeds
        )
        return hidden_states
