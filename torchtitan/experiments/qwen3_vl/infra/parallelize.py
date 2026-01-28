# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelization strategy for Qwen3 VL.

Applies:
1. Activation Checkpointing (on both encoder and text model)
2. torch.compile (per-block compilation)
3. FSDP (per-layer sharding)

Note: TP is not yet supported for VLM training.
"""

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac

from torchtitan.models.llama3.infra.parallelize import (
    _op_sac_save_list,
    apply_compile,
    apply_ddp,
    disable_fsdp_gradient_division,
)
from torchtitan.tools.logging import logger


def parallelize_qwen3_vl(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """
    Apply parallelization strategies to Qwen3 VL model.

    Applies in order:
    1. Activation Checkpointing
    2. torch.compile
    3. FSDP (or DDP)

    Args:
        model: The Qwen3 VL model to parallelize
        parallel_dims: Parallel dimensions configuration
        job_config: Job configuration

    Returns:
        Parallelized model
    """
    # Validate model has encoder
    assert hasattr(model, "encoder") and isinstance(model.encoder, nn.Module), (
        "Model must have an encoder attribute"
    )

    # Check sequence length divisibility
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # Check attention type for CP
    attn_type = getattr(model.model_args, "attn_type", "flex")
    if job_config.parallelism.context_parallel_degree > 1 and attn_type != "sdpa":
        raise NotImplementedError("CP support is only supported for SDPA.")

    # TP not yet supported for VLM
    if parallel_dims.tp_enabled:
        raise NotImplementedError("TP support for VLM training is still in progress.")

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # 1. Apply Activation Checkpointing
    if job_config.activation_checkpoint.mode != "none":
        # Apply to main model (text layers)
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
        )
        # Apply to encoder separately
        apply_ac(model.encoder, job_config.activation_checkpoint)
        logger.info("Applied Activation Checkpointing to model and encoder")

    # 2. Apply torch.compile (after AC, before FSDP)
    if job_config.compile.enable:
        apply_compile(model, job_config.compile)
        apply_compile(model.encoder, job_config.compile)
        logger.info("Applied torch.compile to model and encoder")

    # 3. Apply FSDP or DDP
    if parallel_dims.fsdp_enabled:
        # Get FSDP mesh
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )

        apply_fsdp_qwen3_vl(
            model,
            parallel_dims.get_mesh(names),
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh is not None and dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.compile.enable,
        )
        logger.info("Applied DDP to the model")

    return model


def apply_fsdp_qwen3_vl(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    """
    Apply FSDP2 to Qwen3 VL model with appropriate sharding.

    Sharding points:
    - model.tok_embeddings
    - model.encoder.layers[*]
    - model.encoder.deepstack_mergers[*]
    - model.text_model.layers[*]
    - [model.norm, model.output]
    - Root model

    Args:
        model: The model to shard
        dp_mesh: Device mesh for data parallelism
        param_dtype: Parameter dtype for mixed precision
        reduce_dtype: Reduction dtype for mixed precision
        pp_enabled: Whether pipeline parallelism is enabled
        cpu_offload: Whether to enable CPU offloading
        reshard_after_forward_policy: Resharding policy
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # Determine reshard_after_forward setting
    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid
            # per-microbatch all-gathers
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    # Shard token embeddings
    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # Shard encoder layers
    if hasattr(model, "encoder") and model.encoder is not None:
        # Shard patch embedding
        if hasattr(model.encoder, "patch_embed"):
            fully_shard(
                model.encoder.patch_embed,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        # Shard encoder transformer layers
        if hasattr(model.encoder, "layers"):
            for layer_id, transformer_block in model.encoder.layers.items():
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )

        # Shard DeepStack mergers
        if hasattr(model.encoder, "deepstack_mergers"):
            for merger_id, merger in model.encoder.deepstack_mergers.items():
                fully_shard(
                    merger,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )

        # Shard final merger
        if hasattr(model.encoder, "merger"):
            fully_shard(
                model.encoder.merger,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    # Shard text model layers
    if hasattr(model, "layers"):
        for layer_id, transformer_block in model.layers.items():
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    # Shard final norm and output together
    # Don't reshard these by default since FSDP would prefetch immediately
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    # Shard root module
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division
    disable_fsdp_gradient_division(model)
