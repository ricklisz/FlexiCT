#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NNUNET_SOURCE="${NNUNET_SOURCE:-${SCRIPT_DIR}/nnUNet}"
NNUNET_BIN="${NNUNET_BIN:-}"
export PYTHONPATH="${NNUNET_SOURCE}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "$NNUNET_BIN" ]]; then
    export PATH="${NNUNET_BIN}:${PATH}"
fi
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/flexict_mplconfig}"
mkdir -p "$MPLCONFIGDIR"
export nnUNet_raw="${nnUNet_raw:-${REPO_ROOT}/data/segmentation/raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${REPO_ROOT}/data/segmentation/preprocessed}"
export nnUNet_results="${nnUNet_results:-${REPO_ROOT}/results/segmentation/nnUNet_results}"

: "${DATASET_ID:?Set DATASET_ID, for example DATASET_ID=100}"
: "${CONFIG:?Set CONFIG, for example CONFIG=2d or CONFIG=3d_fullres}"
: "${FOLD:?Set FOLD, for example FOLD=0}"
: "${TRAINER:?Set TRAINER explicitly, for example TRAINER=flexict_patch8_Primus_v2_Trainer}"

PLANS="${PLANS:-nnUNetPlans}"
DEVICE="${DEVICE:-cuda}"

cmd=(nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS" -device "$DEVICE")

if [[ -n "${PRETRAINED_WEIGHTS:-}" ]]; then
    cmd+=(-pretrained_weights "$PRETRAINED_WEIGHTS")
fi

if [[ -n "${NUM_GPUS:-}" ]]; then
    cmd+=(-num_gpus "$NUM_GPUS")
fi

if [[ "${SAVE_NPZ:-0}" == "1" ]]; then
    cmd+=(--npz)
fi

if [[ "${CONTINUE_TRAINING:-0}" == "1" ]]; then
    cmd+=(--c)
fi

if [[ "${VALIDATE_ONLY:-0}" == "1" ]]; then
    cmd+=(--val)
fi

if [[ "${VALIDATE_BEST:-0}" == "1" ]]; then
    cmd+=(--val_best)
fi

if [[ "${DISABLE_CHECKPOINTING:-0}" == "1" ]]; then
    cmd+=(--disable_checkpointing)
fi

exec "${cmd[@]}" "$@"
