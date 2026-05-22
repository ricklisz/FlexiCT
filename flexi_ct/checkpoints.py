"""Checkpoint path resolution for Flexi_CT model variants."""
from __future__ import annotations

import os
from collections.abc import Mapping
from os import PathLike

FLEXICT_2D_DEFAULT_CHECKPOINT: str | None = None
FLEXICT_3D_DEFAULT_CHECKPOINT: str | None = None
FLEXICT_VLM_DEFAULT_CHECKPOINT: str | None = None

_VARIANT_ENV = {
    "2d": "FLEXICT_2D_CHECKPOINT",
    "3d": "FLEXICT_3D_CHECKPOINT",
    "vlm": "FLEXICT_VLM_CHECKPOINT",
}

_VARIANT_DEFAULT = {
    "2d": FLEXICT_2D_DEFAULT_CHECKPOINT,
    "3d": FLEXICT_3D_DEFAULT_CHECKPOINT,
    "vlm": FLEXICT_VLM_DEFAULT_CHECKPOINT,
}

_RETRIEVAL_CKPT_TYPE_VARIANT = {
    "teacher": "3d",
    "vlm": "vlm",
}


def resolve_flexict_checkpoint(
    variant: str,
    explicit_checkpoint: str | PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve a Flexi_CT checkpoint path.

    Precedence is: explicit CLI/API path, FLEXICT_CHECKPOINT, variant-specific
    env var. There is no built-in private-host default.
    """
    variant_key = variant.lower()
    if variant_key not in _VARIANT_DEFAULT:
        raise ValueError(f"Unknown Flexi_CT checkpoint variant: {variant!r}")

    if explicit_checkpoint:
        return str(explicit_checkpoint)

    env_map = os.environ if env is None else env
    global_override = env_map.get("FLEXICT_CHECKPOINT")
    if global_override:
        return global_override

    variant_override = env_map.get(_VARIANT_ENV[variant_key])
    if variant_override:
        return variant_override

    default_checkpoint = _VARIANT_DEFAULT[variant_key]
    if default_checkpoint:
        return default_checkpoint

    raise ValueError(
        "Flexi_CT checkpoint path is required. Pass checkpoint_path/--checkpoint/--pretrain, "
        f"or set FLEXICT_CHECKPOINT or {_VARIANT_ENV[variant_key]}."
    )


def resolve_retrieval_checkpoint(
    ckpt_type: str,
    explicit_checkpoint: str | PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve retrieval-script checkpoint types to Flexi_CT model variants."""
    try:
        variant = _RETRIEVAL_CKPT_TYPE_VARIANT[ckpt_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown retrieval checkpoint type: {ckpt_type!r}") from exc
    return resolve_flexict_checkpoint(variant, explicit_checkpoint, env=env)
