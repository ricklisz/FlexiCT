#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from dinov3.eval.text.hf_text_tower import HFTextTower
from dinov3.eval.text.text_tower import TextTower

logger = logging.getLogger("dinov3")


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _build_device_mesh(process_group: dist.ProcessGroup | None) -> DeviceMesh:
    if process_group is None:
        return init_device_mesh(
            "cuda",
            mesh_shape=(dist.get_world_size(),),
            mesh_dim_names=("dp",),
        )
    return DeviceMesh.from_group(process_group, "cuda")


def _wrap_hf_text_model(text_model: HFTextTower, fsdp_config: dict) -> None:
    if text_model.backbone is None:
        text_model.build_backbone_from_config()
    if isinstance(text_model.backbone, nn.Module):
        text_model.backbone = fully_shard(
            text_model.backbone, **fsdp_config, reshard_after_forward=True
        )
    if isinstance(text_model.projection, nn.Linear):
        text_model.projection = fully_shard(
            text_model.projection, **fsdp_config, reshard_after_forward=True
        )
    fully_shard(text_model, **fsdp_config, reshard_after_forward=True)


def _wrap_text_tower(text_model: TextTower, fsdp_config: dict) -> None:
    if hasattr(text_model.backbone, "blocks"):
        for block_id, block in enumerate(text_model.backbone.blocks):
            text_model.backbone.blocks[block_id] = fully_shard(
                block, **fsdp_config, reshard_after_forward=True
            )
    if hasattr(text_model.head, "blocks"):
        for block_id, block in enumerate(text_model.head.blocks):
            text_model.head.blocks[block_id] = fully_shard(
                block, **fsdp_config, reshard_after_forward=True
            )
    if isinstance(text_model.head.linear_projection, nn.Linear):
        text_model.head.linear_projection = fully_shard(
            text_model.head.linear_projection, **fsdp_config, reshard_after_forward=True
        )
    fully_shard(text_model.backbone, **fsdp_config, reshard_after_forward=True)
    fully_shard(text_model.head, **fsdp_config, reshard_after_forward=True)
    fully_shard(text_model, **fsdp_config, reshard_after_forward=True)


def wrap_text_model_fsdp(
    text_model: nn.Module,
    cfg,
    process_group: dist.ProcessGroup | None = None,
) -> DeviceMesh:
    world_mesh = _build_device_mesh(process_group)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=DTYPE_MAP[cfg.compute_precision.param_dtype],
        reduce_dtype=DTYPE_MAP[cfg.compute_precision.reduce_dtype],
    )
    fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy}

    logger.info("DISTRIBUTED FSDP -- preparing text model for distributed training")
    if isinstance(text_model, HFTextTower):
        _wrap_hf_text_model(text_model, fsdp_config)
    elif isinstance(text_model, TextTower):
        _wrap_text_tower(text_model, fsdp_config)
    else:
        logger.warning(
            "Unknown text model type %s, sharding whole module",
            type(text_model),
        )
        fully_shard(text_model, **fsdp_config, reshard_after_forward=True)

    text_model.to_empty(device="cuda")
    return world_mesh
