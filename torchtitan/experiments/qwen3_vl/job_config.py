# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Job configuration extensions for Qwen3 VL.

Adds VLM-specific configuration options for data processing.
"""

from dataclasses import dataclass


@dataclass
class Qwen3VLDataConfig:
    """
    Data configuration specific to Qwen3 VL training.

    Extends the base data config with vision-specific parameters.
    """

    # Maximum number of images per batch
    max_images_per_batch: int = 16

    # Maximum patches per image (controls max resolution)
    max_patches_per_image: int = 4096

    # Patch size (must match model config)
    patch_size: int = 16

    # Spatial merge size for patch merger
    spatial_merge_size: int = 2

    # Packing buffer size (0 to disable packing)
    packing_buffer_size: int = 0
