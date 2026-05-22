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

: "${DATASET_ID:?Set DATASET_ID, for example DATASET_ID=100 bash downstream/segmentation/run_plan_preprocess.sh}"

cmd=(nnUNetv2_plan_and_preprocess -d "$DATASET_ID")

if [[ -n "${CONFIG:-}" ]]; then
    read -r -a configs <<< "$CONFIG"
    cmd+=(-c "${configs[@]}")
fi

if [[ -n "${PLANNER:-}" ]]; then
    cmd+=(-pl "$PLANNER")
fi

if [[ -n "${FINGERPRINT_EXTRACTOR:-}" ]]; then
    cmd+=(-fpe "$FINGERPRINT_EXTRACTOR")
fi

if [[ -n "${NPFP:-}" ]]; then
    cmd+=(-npfp "$NPFP")
fi

if [[ -n "${NP:-}" ]]; then
    read -r -a preprocess_processes <<< "$NP"
    cmd+=(-np "${preprocess_processes[@]}")
fi

if [[ -n "${GPU_MEMORY_TARGET:-}" ]]; then
    cmd+=(-gpu_memory_target "$GPU_MEMORY_TARGET")
fi

if [[ -n "${PREPROCESSOR_NAME:-}" ]]; then
    cmd+=(-preprocessor_name "$PREPROCESSOR_NAME")
fi

if [[ -n "${OVERWRITE_TARGET_SPACING:-}" ]]; then
    read -r -a target_spacing <<< "$OVERWRITE_TARGET_SPACING"
    cmd+=(-overwrite_target_spacing "${target_spacing[@]}")
fi

if [[ -n "${OVERWRITE_PLANS_NAME:-}" ]]; then
    cmd+=(-overwrite_plans_name "$OVERWRITE_PLANS_NAME")
fi

if [[ "${VERIFY_DATASET_INTEGRITY:-0}" == "1" ]]; then
    cmd+=(--verify_dataset_integrity)
fi

if [[ "${NO_PREPROCESS:-0}" == "1" ]]; then
    cmd+=(--no_pp)
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
    cmd+=(--clean)
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
    cmd+=(--verbose)
fi

exec "${cmd[@]}" "$@"
