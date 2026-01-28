# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL (Vision-Language) Experiment for TorchTitan.

This experiment implements Qwen3 VL 32B, a multimodal vision-language model
with the following key features (images only):

- Conv2D patch embedding for images
- DeepStack: multi-level vision feature extraction at layers [8, 16, 24]
- MRoPE: Multi-dimensional RoPE for vision-language position encoding
- Q/K normalization for training stability
- GQA (Grouped Query Attention) with 8:1 ratio for 32B

Usage:
    CONFIG_FILE=torchtitan/experiments/qwen3_vl/train_configs/debug_model.toml
    torchtitan --job.config_file $CONFIG_FILE
"""

from dataclasses import fields
from typing import Any

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.protocols.train_spec import TrainSpec

# Reuse VLM dataloader (compatible with Qwen3 VL)
from ..vlm.datasets.mm_datasets import build_mm_dataloader
from .infra.parallelize import parallelize_qwen3_vl
from .model.args import Qwen3VLModelArgs, Qwen3VLVisionArgs
from .model.model import Qwen3VLTransformer
from .model.state_dict_adapter import Qwen3VLStateDictAdapter

__all__ = [
    "parallelize_qwen3_vl",
    "Qwen3VLModelArgs",
    "Qwen3VLTransformer",
    "qwen3_vl_args",
]


def _get_dict(obj) -> dict[str, Any]:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


# Model configurations for different sizes
qwen3_vl_args = {
    # Debug model for testing
    "debugmodel": Qwen3VLModelArgs(
        # Small text model
        dim=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        head_dim=32,
        hidden_dim=512,
        vocab_size=151936,
        qk_norm=True,
        max_seq_len=4096,
        # MRoPE config (scaled for smaller head_dim)
        mrope_section=[8, 4, 4],  # sum = 16 = head_dim/2
        mrope_theta=500000.0,
        # Small vision encoder
        encoder=Qwen3VLVisionArgs(
            depth=4,
            hidden_size=128,
            intermediate_size=256,
            num_heads=4,
            patch_size=16,
            spatial_merge_size=2,
            out_hidden_size=256,  # Match text model dim
            num_position_embeddings=576,  # 24x24 grid
            deepstack_visual_indexes=[1, 2, 3],  # Fewer layers
        ),
    ),
    # Full 32B configuration
    "32B": Qwen3VLModelArgs(
        # Text model (from torchtitan/models/qwen3/__init__.py:109-120)
        dim=5120,
        n_layers=64,
        n_heads=64,
        n_kv_heads=8,  # GQA 8:1 ratio
        head_dim=128,
        hidden_dim=25600,
        vocab_size=151936,
        qk_norm=True,
        rope_theta=1000000,
        max_seq_len=4096,
        # MRoPE configuration
        mrope_section=[24, 20, 20],  # sum = 64 = head_dim/2
        mrope_theta=500000.0,  # Different from text-only Qwen3
        # Vision encoder (from HF Qwen3-VL config)
        encoder=Qwen3VLVisionArgs(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            patch_size=16,
            spatial_merge_size=2,
            out_hidden_size=5120,  # Must match text model dim
            num_position_embeddings=2304,  # 48x48 grid
            deepstack_visual_indexes=[8, 16, 24],
        ),
    ),
    # 8B configuration for smaller-scale experiments
    "8B": Qwen3VLModelArgs(
        # Text model
        dim=4096,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,  # GQA 4:1 ratio
        head_dim=128,
        hidden_dim=12288,
        vocab_size=151936,
        qk_norm=True,
        rope_theta=1000000,
        max_seq_len=4096,
        # MRoPE configuration
        mrope_section=[24, 20, 20],
        mrope_theta=500000.0,
        # Vision encoder
        encoder=Qwen3VLVisionArgs(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            patch_size=16,
            spatial_merge_size=2,
            out_hidden_size=4096,  # Must match text model dim
            num_position_embeddings=2304,
            deepstack_visual_indexes=[8, 16, 24],
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    """
    Get the training specification for Qwen3 VL.

    Returns:
        TrainSpec with all necessary components for training Qwen3 VL.
    """
    return TrainSpec(
        model_cls=Qwen3VLTransformer,
        model_args=qwen3_vl_args,
        parallelize_fn=parallelize_qwen3_vl,
        pipelining_fn=None,  # Pipeline parallelism not yet supported for VLM
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3VLStateDictAdapter,
    )
