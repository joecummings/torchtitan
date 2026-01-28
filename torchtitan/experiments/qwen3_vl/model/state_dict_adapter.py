# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Qwen3 VL.

Converts between HuggingFace Qwen3-VL weights and TorchTitan format.
"""

import re
from typing import Any

from torch.distributed.tensor import DTensor
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import Qwen3VLModelArgs


class Qwen3VLStateDictAdapter(StateDictAdapter):
    """
    Adapter for converting between HF and TorchTitan state dicts for Qwen3 VL.

    Handles mappings for:
    - Vision encoder (patch_embed, blocks, merger, deepstack)
    - Text model (embeddings, attention with Q/K norm, MLP)
    """

    def __init__(self, model_args: Qwen3VLModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.from_hf_map = {
            # Vision Encoder - Patch Embedding
            "visual.patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
            "visual.patch_embed.proj.bias": "encoder.patch_embed.proj.bias",

            # Vision Encoder - Position Embedding
            "visual.pos_embed.weight": "encoder.pos_embed.weight",

            # Vision Encoder - Blocks (transformer layers)
            "visual.blocks.{}.norm1.weight": "encoder.layers.{}.norm1.weight",
            "visual.blocks.{}.norm1.bias": "encoder.layers.{}.norm1.bias",
            "visual.blocks.{}.attn.qkv.weight": "encoder.layers.{}.attn.qkv.weight",
            "visual.blocks.{}.attn.qkv.bias": "encoder.layers.{}.attn.qkv.bias",
            "visual.blocks.{}.attn.proj.weight": "encoder.layers.{}.attn.proj.weight",
            "visual.blocks.{}.attn.proj.bias": "encoder.layers.{}.attn.proj.bias",
            "visual.blocks.{}.norm2.weight": "encoder.layers.{}.norm2.weight",
            "visual.blocks.{}.norm2.bias": "encoder.layers.{}.norm2.bias",
            "visual.blocks.{}.mlp.linear_fc1.weight": "encoder.layers.{}.mlp.fc1.weight",
            "visual.blocks.{}.mlp.linear_fc1.bias": "encoder.layers.{}.mlp.fc1.bias",
            "visual.blocks.{}.mlp.linear_fc2.weight": "encoder.layers.{}.mlp.fc2.weight",
            "visual.blocks.{}.mlp.linear_fc2.bias": "encoder.layers.{}.mlp.fc2.bias",

            # Vision Encoder - Final Merger
            "visual.merger.norm.weight": "encoder.merger.norm.weight",
            "visual.merger.norm.bias": "encoder.merger.norm.bias",
            "visual.merger.linear_fc1.weight": "encoder.merger.fc1.weight",
            "visual.merger.linear_fc1.bias": "encoder.merger.fc1.bias",
            "visual.merger.linear_fc2.weight": "encoder.merger.fc2.weight",
            "visual.merger.linear_fc2.bias": "encoder.merger.fc2.bias",

            # Vision Encoder - DeepStack Mergers (indexed)
            "visual.merger_list.{}.norm.weight": "encoder.deepstack_mergers.{}.norm.weight",
            "visual.merger_list.{}.norm.bias": "encoder.deepstack_mergers.{}.norm.bias",
            "visual.merger_list.{}.linear_fc1.weight": "encoder.deepstack_mergers.{}.fc1.weight",
            "visual.merger_list.{}.linear_fc1.bias": "encoder.deepstack_mergers.{}.fc1.bias",
            "visual.merger_list.{}.linear_fc2.weight": "encoder.deepstack_mergers.{}.fc2.weight",
            "visual.merger_list.{}.linear_fc2.bias": "encoder.deepstack_mergers.{}.fc2.bias",

            # Text Model - Embeddings
            "model.embed_tokens.weight": "tok_embeddings.weight",

            # Text Model - Attention (with Q/K norm)
            "model.layers.{}.self_attn.q_proj.weight": "text_model.layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "text_model.layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "text_model.layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "text_model.layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "text_model.layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "text_model.layers.{}.attention.k_norm.weight",

            # Skip rotary_emb.inv_freq (computed dynamically)
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,

            # Text Model - MLP
            "model.layers.{}.mlp.gate_proj.weight": "text_model.layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "text_model.layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "text_model.layers.{}.feed_forward.w2.weight",

            # Text Model - Layer Norms
            "model.layers.{}.input_layernorm.weight": "text_model.layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "text_model.layers.{}.ffn_norm.weight",

            # Text Model - Final Norm and Output
            "model.norm.weight": "text_model.norm.weight",
            "lm_head.weight": "text_model.output.weight",
        }

        # Build reverse mapping
        self.to_hf_map = {}
        for hf_key, titan_key in self.from_hf_map.items():
            if titan_key is not None:
                self.to_hf_map[titan_key] = hf_key

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert TorchTitan state dict to HuggingFace format.

        Args:
            state_dict: TorchTitan state dict

        Returns:
            HuggingFace compatible state dict
        """
        hf_state_dict = {}

        for key, value in state_dict.items():
            # Handle DTensor
            if isinstance(value, DTensor):
                value = value.to_local()

            # Check for layer-indexed keys
            if any(
                pattern in key
                for pattern in ["layers.", "deepstack_mergers.", "merger_list."]
            ):
                # Extract layer numbers and find matching pattern
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key)

                if layer_num and abstract_key in self.to_hf_map:
                    new_key = self.to_hf_map[abstract_key].format(layer_num.group(0))
                    hf_state_dict[new_key] = value
            else:
                # Direct mapping
                if key in self.to_hf_map:
                    hf_state_dict[self.to_hf_map[key]] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert HuggingFace state dict to TorchTitan format.

        Args:
            hf_state_dict: HuggingFace state dict

        Returns:
            TorchTitan compatible state dict
        """
        state_dict = {}

        for key, value in hf_state_dict.items():
            # Check for layer-indexed keys (vision blocks or text layers)
            if any(
                pattern in key
                for pattern in ["blocks.", "model.layers.", "merger_list."]
            ):
                # Handle keys with one or more layer indices
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_nums = re.findall(r"\d+", key)

                if abstract_key in self.from_hf_map:
                    titan_key = self.from_hf_map[abstract_key]
                    if titan_key is None:
                        continue  # Skip (e.g., inv_freq)

                    # Replace placeholders with layer numbers
                    new_key = titan_key
                    for layer_num in layer_nums:
                        new_key = new_key.replace("{}", layer_num, 1)

                    state_dict[new_key] = value
            else:
                # Direct mapping
                if key in self.from_hf_map:
                    titan_key = self.from_hf_map[key]
                    if titan_key is not None:
                        state_dict[titan_key] = value

        return state_dict
