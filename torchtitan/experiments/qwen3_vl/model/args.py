# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Model arguments for Qwen3 VL (Vision-Language Model).
"""

from dataclasses import dataclass, field
from typing import Any

from torch import nn

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs


@dataclass
class SpecialTokens:
    """Special tokens used in Qwen3 VL for vision-language processing."""

    img_token: str
    img_id: int
    vision_start_token: str
    vision_start_id: int
    vision_end_token: str
    vision_end_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # Pytorch F.cross_entropy default

    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        """Create SpecialTokens from tokenizer's vocabulary."""
        # Qwen3 VL specific token IDs
        return cls(
            img_token="<|image_pad|>",
            img_id=151655,
            vision_start_token="<|vision_start|>",
            vision_start_id=151652,
            vision_end_token="<|vision_end|>",
            vision_end_id=151653,
            pad_token="<|endoftext|>",
            pad_id=151643,
        )


@dataclass
class Qwen3VLVisionArgs:
    """Vision encoder configuration for Qwen3 VL (images only)."""

    # Core architecture
    depth: int = 27  # Number of vision transformer layers
    hidden_size: int = 1152  # Vision embedding dimension
    intermediate_size: int = 4304  # Vision FFN intermediate size
    num_heads: int = 16  # Number of attention heads in vision encoder

    # Patch embedding
    in_channels: int = 3  # Input channels (RGB)
    patch_size: int = 16  # Spatial patch size
    spatial_merge_size: int = 2  # 2x2 spatial merge for patch merger

    # Output projection
    out_hidden_size: int = 3584  # Output dimension (projected to text model)

    # Position embeddings
    num_position_embeddings: int = 2304  # Max position embeddings (48x48 grid)

    # DeepStack: intermediate feature extraction layers
    deepstack_visual_indexes: list[int] = field(
        default_factory=lambda: [8, 16, 24]
    )

    # Normalization
    layer_norm_eps: float = 1e-6

    # Attention configuration
    attn_type: str = "flex"
    attn_mask_type: str = "causal"


@dataclass
class Qwen3VLTextArgs:
    """Text model configuration for Qwen3 VL (extends Qwen3 with MRoPE)."""

    # Core architecture
    dim: int = 5120  # Model dimension
    n_layers: int = 64  # Number of transformer layers
    n_heads: int = 64  # Number of attention heads
    n_kv_heads: int = 8  # Number of key-value heads for GQA
    head_dim: int = 128  # Dimension per head
    hidden_dim: int = 25600  # FFN intermediate dimension
    vocab_size: int = 151936  # Vocabulary size

    # Normalization
    norm_eps: float = 1e-6
    qk_norm: bool = True  # Q/K normalization (Qwen3 specific)

    # RoPE configuration
    rope_theta: float = 1000000.0  # Base frequency for RoPE

    # MRoPE configuration (Multi-dimensional RoPE for VL)
    mrope_section: list[int] = field(
        default_factory=lambda: [24, 20, 20]
    )  # t, h, w frequency sections (sum=64=head_dim/2)
    mrope_theta: float = 500000.0  # MRoPE uses different theta than text-only

    # Sequence configuration
    max_seq_len: int = 4096

    # Attention configuration
    attn_type: str = "flex"
    attn_mask_type: str = "block_causal"

    # Initialization
    depth_init: bool = True

    # Other
    eos_id: int = 151645


@dataclass
class Qwen3VLModelArgs(BaseModelArgs):
    """Combined model arguments for Qwen3 VL."""

    # Vision encoder config
    encoder: Qwen3VLVisionArgs = field(default_factory=Qwen3VLVisionArgs)

    # Text model config (flattened for compatibility with existing patterns)
    dim: int = 5120
    n_layers: int = 64
    n_heads: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 25600
    vocab_size: int = 151936
    norm_eps: float = 1e-6
    qk_norm: bool = True
    rope_theta: float = 1000000.0
    max_seq_len: int = 4096
    depth_init: bool = True

    # MRoPE configuration
    mrope_section: list[int] = field(
        default_factory=lambda: [24, 20, 20]
    )
    mrope_theta: float = 500000.0

    # Attention configuration
    attn_type: str = "flex"
    attn_mask_type: str = "block_causal"
    eos_id: int = 151645

    # Special token IDs for vision
    image_token_id: int = 151655
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653

    def update_from_config(self, job_config: JobConfig, **kwargs: Any) -> None:
        """Update model args from job config."""
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            from torchtitan.tools.logging import logger

            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        """Estimate number of parameters and FLOPs."""
        # Count parameters
        nparams = sum(p.numel() for p in model.parameters())

        # Estimate FLOPs for forward pass
        # Text model FLOPs (simplified estimate)
        text_flops = (
            2 * self.n_layers * seq_len * self.dim * (
                4 * self.dim +  # QKV projections
                2 * seq_len +   # Attention
                3 * self.hidden_dim  # FFN
            )
        )

        # Vision model FLOPs (simplified estimate)
        num_patches = (self.max_seq_len // 4)  # Rough estimate
        vision_flops = (
            2 * self.encoder.depth * num_patches * self.encoder.hidden_size * (
                4 * self.encoder.hidden_size +
                2 * num_patches +
                3 * self.encoder.intermediate_size
            )
        )

        return nparams, text_flops + vision_flops
