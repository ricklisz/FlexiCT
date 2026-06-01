# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
from pathlib import Path
import logging
import torch
import torch.distributed as dist
# PyTorch DCP
import torch.distributed.checkpoint as dcp
import torch.distributed.checkpoint.filesystem as dcpfs
import torch.distributed.checkpoint.state_dict as dcpsd
from torch.distributed.checkpoint import default_planner

# DINOv3
from dinov3.configs import setup_config, setup_job
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.train.multidist_meta_arch import MultiDistillationMetaArch
from dinov3.checkpointer import find_latest_checkpoint

logger = logging.getLogger("dinov3.convert")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def materialize_from_meta_(module: torch.nn.Module, device="cpu"):
    """Allocate real storage for any meta params/buffers (no init_weights needed)."""
    for name, p in list(module.named_parameters(recurse=False)):
        if getattr(p, "is_meta", False):
            new = torch.nn.Parameter(torch.empty_like(p, device=device), requires_grad=p.requires_grad)
            module.register_parameter(name, new)
    for name, b in list(module.named_buffers(recurse=False)):
        if getattr(b, "is_meta", False):
            module.register_buffer(name, torch.empty_like(b, device=device), persistent=True)
    for child in module.children():
        materialize_from_meta_(child, device=device)


def dedtensorize_(module: torch.nn.Module):
    """Ensure no DTensor params/buffers remain."""
    try:
        from torch.distributed._tensor import DTensor
    except Exception:
        DTensor = ()  # no-op if module isn't available

    for name, p in list(module.named_parameters(recurse=False)):
        if isinstance(p, DTensor):
            local = p.full_tensor().contiguous()
            module.register_parameter(name, torch.nn.Parameter(local, requires_grad=p.requires_grad))
    for name, b in list(module.named_buffers(recurse=False)):
        if isinstance(b, DTensor):
            module.register_buffer(name, b.full_tensor().contiguous(), persistent=True)
    for child in module.children():
        dedtensorize_(child)


def maybe_full_tensor(x):
    try:
        from torch.distributed._tensor import DTensor
        if isinstance(x, DTensor):
            return x.full_tensor()
    except Exception:
        pass
    return x


def export_regular_checkpoint(out_path: Path, payload: dict):
    cleaned = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            cleaned[k] = {kk: maybe_full_tensor(vv) for kk, vv in v.items()}
        else:
            cleaned[k] = maybe_full_tensor(v)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cleaned, out_path)
    logger.info(f"✅ Wrote: {out_path.resolve()}")


def build_meta_arch_from_cfg(cfg):
    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")

    logger.info(f"Instantiating {meta_arch.__name__} (plain nn.Module; no init_weights, no FSDP)")
    model = meta_arch(cfg)                 # do NOT call prepare_for_distributed_training()
    # IMPORTANT: do NOT call model.init_weights(); it triggers DTensor/FSDP logic.
    materialize_from_meta_(model, "cpu")   # give storage to any meta params/buffers
    dedtensorize_(model)                   # safety: ensure no DTensors
    return model

def main():
    parser = argparse.ArgumentParser("Convert DCP shard dir → teacher-only .pth")
    parser.add_argument("--config-file", required=True, help="Training config path")
    parser.add_argument("--output-dir", required=True, help="Training output_dir (contains ckpt/)")
    parser.add_argument("--ckpt-dir", default="", help="Specific DCP dir (…/ckpt/21999). If empty, use latest.")
    parser.add_argument("--out", default="teacher_checkpoint.pth", help="Destination .pth")
    parser.add_argument("--strict", action="store_true", help="Strict DCP load (default: allow partial)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Config overrides")
    args = parser.parse_args()

    # Keep MODEL.WEIGHTS from auto-loading anything (belt-and-suspenders)
    setup_job(output_dir=args.output_dir, seed=0)
    cfg = setup_config(args, strict_cfg=False)
    if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "WEIGHTS"):
        try:
            cfg.MODEL.WEIGHTS = ""  # prevent init routines from pulling weights anywhere
        except Exception:
            pass
    logger.info("Loaded config.")

    # Build plain model (no init_weights)
    model = build_meta_arch_from_cfg(cfg)

    # Resolve DCP directory
    ckpt_root = Path(args.output_dir).expanduser() / "ckpt"
    if args.ckpt_dir:
        dcp_dir = Path(args.ckpt_dir).expanduser()
    else:
        latest = find_latest_checkpoint(ckpt_root)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints under {ckpt_root}")
        dcp_dir = latest
    if not dcp_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {dcp_dir}")
    logger.info(f"Loading DCP from: {dcp_dir}")

    # Logical state to load from DCP into our plain model
    to_load = {"iteration": None}
    to_load["model"] = dcpsd.get_model_state_dict(model)

    # Single-process DCP load
    dcp.load(
        to_load,
        storage_reader=dcpfs.FileSystemReader(dcp_dir),
        planner=default_planner.DefaultLoadPlanner(allow_partial_load=not args.strict),
        process_group=None,
    )
    iteration = to_load["iteration"]

    # Copy into model
    dcpsd.set_model_state_dict(model, to_load["model"])

    # Export teacher-only (EMA) weights
    teacher_sd = {k: maybe_full_tensor(v) for k, v in model.teacher.state_dict().items()}
    export_regular_checkpoint(Path(args.out).expanduser(), {"teacher": teacher_sd, "iteration": iteration})
    if dist.is_available() and dist.is_initialized():
        dist.barrier()                     # only call barrier if group exists
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
