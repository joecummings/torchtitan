# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Main Qwen3 VL Transformer model.

Combines:
- Vision encoder with DeepStack
- Text decoder with MRoPE and Q/K normalization
- Vision-to-text projection
"""

import einops as E
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Qwen3VLModelArgs, SpecialTokens
from .qwen3_vl_text import Qwen3VLTextModel
from .qwen3_vl_vision import Qwen3VLVisionEncoder


def _scatter_img_tokens(
    h_BSD: torch.Tensor,
    tokens_BS: torch.Tensor,
    i_NLD: torch.Tensor,
    i_mask_NL: torch.Tensor,
    img_id: int,
) -> torch.Tensor:
    """
    Scatter image tokens into the LLM input embeddings.

    Args:
        h_BSD: Token embeddings of shape (batch, seq_len, dim)
        tokens_BS: Token IDs of shape (batch, seq_len)
        i_NLD: Image embeddings of shape (num_images, num_patches, dim)
        i_mask_NL: Boolean mask for valid image patches
        img_id: Token ID for image placeholder

    Returns:
        Updated token embeddings with image features scattered in
    """
    B, S, D = h_BSD.shape

    # Where are the image tokens in LLM input, make broadcastable with h_BSD
    img_mask_h_BSD = E.repeat(tokens_BS == img_id, "b s -> b s 1")

    # Only get valid (non-padded) tokens, result is flattened
    i_flatten = torch.masked_select(i_NLD, mask=i_mask_NL.unsqueeze(-1))

    assert i_flatten.numel() // D == img_mask_h_BSD.sum(), (
        f"Different number of visual embeddings {i_flatten.numel() // D} "
        f"with placeholder in input token embeddings {img_mask_h_BSD.sum()}"
    )

    h_BSD.masked_scatter_(mask=img_mask_h_BSD, source=i_flatten)
    return h_BSD


class Qwen3VLTransformer(ModelProtocol):
    """
    Qwen3 VL Transformer model.

    Architecture:
    1. Vision encoder processes images
    2. Patch merger projects vision features to text dimension
    3. DeepStack extracts multi-level vision features
    4. Text decoder with MRoPE processes combined sequence
    5. DeepStack features injected at early text decoder layers

    Forward flow:
    1. Encode images -> vision features + DeepStack features
    2. Compute MRoPE 3D positions from tokens + grid_thw
    3. Embed tokens -> h_BSD
    4. Scatter vision features into h_BSD at image positions
    5. Forward through text layers with DeepStack injection
    6. Output projection
    """

    def __init__(self, model_args: Qwen3VLModelArgs) -> None:
        super().__init__(model_args)
        self.model_args = model_args

        # Vision encoder
        self.encoder = Qwen3VLVisionEncoder(model_args.encoder)

        # Text model components (exposed for FSDP sharding)
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # Text decoder (without embeddings, handled separately)
        self.text_model = Qwen3VLTextModel(model_args)

        # Share embedding with text model
        self.text_model.tok_embeddings = None  # Don't duplicate

        # Output projection (shared reference)
        self.norm = self.text_model.norm
        self.output = self.text_model.output

        # Layers reference for FSDP
        self.layers = self.text_model.layers

        # Special tokens
        self.image_token_id = model_args.image_token_id

    def init_weights(self, buffer_device: torch.device | None = None):
        """Initialize all weights."""
        buffer_device = buffer_device or torch.device("cpu")

        # Initialize encoder
        if self.encoder is not None:
            self.encoder.init_weights(buffer_device)

        # Initialize embeddings
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        # Initialize text model (layers, norms, output)
        self.text_model.init_weights(buffer_device)

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        """
        Generate attention masks for both encoder and decoder.

        Returns nested dict with separate masks for vision and text.
        """
        # Get text model masks
        text_masks = self.text_model.get_attention_masks(
            input_batch, tokenizer, extra_inputs
        )

        # Get encoder masks if we have vision input
        encoder_masks = None
        if self.encoder is not None and extra_inputs is not None:
            if "grid_thw" in extra_inputs:
                encoder_masks = self.encoder.get_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )

        return {
            "text_masks": text_masks,
            "encoder_masks": encoder_masks,
        }

    def _compute_mrope_positions(
        self,
        tokens: torch.Tensor,
        grid_thw: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 3D MRoPE position IDs from token sequence.

        For vision tokens: assigns (t, h, w) positions based on grid_thw
        For text tokens: assigns (t, t, t) positions (all dimensions same)

        Args:
            tokens: Token IDs of shape (batch, seq_len)
            grid_thw: Grid dimensions of shape (num_images, 3)
            attention_mask: Optional attention mask

        Returns:
            Tuple of:
                - position_ids: Shape (3, batch, seq_len) for t, h, w
                - rope_deltas: Position offset for caching
        """
        spatial_merge_size = self.model_args.encoder.spatial_merge_size
        image_token_id = self.image_token_id
        vision_start_token_id = self.model_args.vision_start_token_id

        batch_size, seq_len = tokens.shape
        device = tokens.device

        mrope_position_deltas = []

        if grid_thw is not None and grid_thw.numel() > 0:
            position_ids = torch.ones(
                3, batch_size, seq_len,
                dtype=tokens.dtype,
                device=device,
            )

            if attention_mask is None:
                attention_mask = torch.ones_like(tokens)

            image_index = 0

            for i, input_ids in enumerate(tokens):
                input_ids_masked = input_ids[attention_mask[i] == 1]

                # Find vision segments
                vision_start_indices = torch.argwhere(
                    input_ids_masked == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids_masked[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum().item()

                input_tokens = input_ids_masked.tolist()
                llm_pos_ids_list = []
                st = 0

                for _ in range(image_nums):
                    if image_token_id in input_tokens:
                        ed = input_tokens.index(image_token_id, st)
                    else:
                        ed = len(input_tokens)

                    t, h, w = (
                        grid_thw[image_index][0].item(),
                        grid_thw[image_index][1].item(),
                        grid_thw[image_index][2].item(),
                    )
                    image_index += 1

                    llm_grid_t = t
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size
                    text_len = ed - st

                    # Text positions before image
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                    # Image positions (t, h, w grids)
                    t_index = (
                        torch.arange(llm_grid_t, device=device)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h, device=device)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w, device=device)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index])
                        + text_len
                        + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                # Remaining text after all images
                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                        + st_idx
                    )

                if llm_pos_ids_list:
                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                    position_ids[..., i, attention_mask[i] == 1] = llm_positions
                    mrope_position_deltas.append(
                        llm_positions.max() + 1 - len(tokens[i])
                    )
                else:
                    mrope_position_deltas.append(torch.tensor(0, device=device))

            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=device
            ).unsqueeze(1)

        else:
            # Text-only case: all dimensions get same position
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(seq_len, device=device)
                    .view(1, 1, -1)
                    .expand(3, batch_size, -1)
                )
                mrope_position_deltas = torch.zeros(
                    [batch_size, 1], device=device, dtype=tokens.dtype
                )

        return position_ids, mrope_position_deltas

    def forward(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: SpecialTokens,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through Qwen3 VL.

        Args:
            tokens: Token IDs of shape (batch, seq_len)
            pixel_values: Patchified image pixels
            grid_thw: Grid dimensions (temporal, height, width) for each image
            special_tokens: Special token configuration
            attention_masks: Nested dict with text_masks and encoder_masks

        Returns:
            Output logits of shape (batch, seq_len, vocab_size)
        """
        # Token embeddings
        h_BSD = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Process vision input
        visual_pos_mask = None
        deepstack_features = None

        if self.encoder is not None and pixel_values is not None:
            assert attention_masks is not None, (
                "encoder requires attention masks for FlexAttention"
            )

            # Compute valid pixel mask
            pixel_masks = E.reduce(grid_thw != -1, "n hw -> n", reduction="all")

            # Get encoder masks
            encoder_masks = attention_masks.get("encoder_masks")

            # Forward through vision encoder
            vision_output, deepstack_features = self.encoder(
                pixel_values, grid_thw, encoder_masks
            )

            # Scatter vision features into token embeddings
            h_BSD = _scatter_img_tokens(
                h_BSD,
                tokens,
                vision_output.unsqueeze(0) if vision_output.dim() == 2 else vision_output,
                pixel_masks.unsqueeze(-1).expand(-1, vision_output.shape[0] // pixel_masks.shape[0])
                if pixel_masks.dim() == 1
                else pixel_masks,
                special_tokens.img_id,
            )

            # Create visual position mask for DeepStack
            visual_pos_mask = (tokens == special_tokens.img_id)

        # Compute MRoPE positions
        position_ids, _ = self._compute_mrope_positions(tokens, grid_thw)

        # Get text attention masks
        text_masks = (
            attention_masks.get("text_masks") if attention_masks else None
        )

        # Forward through text model layers
        cos, sin = self.text_model.mrope(h_BSD, position_ids)

        for layer_idx, layer in self.text_model.layers.items():
            h_BSD = layer(h_BSD, cos, sin, text_masks)

            # Inject DeepStack features at early layers
            idx = int(layer_idx)
            if (
                deepstack_features is not None
                and visual_pos_mask is not None
                and idx < self.text_model.num_deepstack_layers
            ):
                h_BSD = self.text_model._inject_deepstack_features(
                    h_BSD,
                    visual_pos_mask,
                    deepstack_features[idx],
                )

        # Final normalization and output
        h_BSD = self.norm(h_BSD) if self.norm else h_BSD
        output = self.output(h_BSD) if self.output else h_BSD

        return output
