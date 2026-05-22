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

: "${INPUT:?Set INPUT to the nnU-Net input images folder}"
: "${OUTPUT:?Set OUTPUT to the prediction output folder}"

DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"

if [[ -n "${MODEL_FOLDER:-}" ]]; then
    use_model_folder=1
    cmd=(nnUNetv2_predict_from_modelfolder -i "$INPUT" -o "$OUTPUT" -m "$MODEL_FOLDER" -chk "$CHECKPOINT" -device "$DEVICE")
else
    use_model_folder=0
    : "${DATASET_ID:?Set DATASET_ID unless MODEL_FOLDER is set}"
    : "${CONFIG:?Set CONFIG unless MODEL_FOLDER is set}"
    : "${TRAINER:?Set TRAINER explicitly unless MODEL_FOLDER is set}"
    PLANS="${PLANS:-nnUNetPlans}"
    cmd=(nnUNetv2_predict -i "$INPUT" -o "$OUTPUT" -d "$DATASET_ID" -c "$CONFIG" -tr "$TRAINER" -p "$PLANS" -chk "$CHECKPOINT" -device "$DEVICE")
fi

if [[ -n "${FOLD:-}" ]]; then
    read -r -a folds <<< "$FOLD"
    cmd+=(-f "${folds[@]}")
fi

if [[ -n "${STEP_SIZE:-}" ]]; then
    cmd+=(-step_size "$STEP_SIZE")
fi

if [[ -n "${NPP:-}" ]]; then
    cmd+=(-npp "$NPP")
fi

if [[ -n "${NPS:-}" ]]; then
    cmd+=(-nps "$NPS")
fi

if [[ -n "${PREV_STAGE_PREDICTIONS:-}" ]]; then
    cmd+=(-prev_stage_predictions "$PREV_STAGE_PREDICTIONS")
fi

if [[ "$use_model_folder" == "0" && -n "${NUM_PARTS:-}" ]]; then
    cmd+=(-num_parts "$NUM_PARTS")
fi

if [[ "$use_model_folder" == "0" && -n "${PART_ID:-}" ]]; then
    cmd+=(-part_id "$PART_ID")
fi

if [[ "${DISABLE_TTA:-0}" == "1" ]]; then
    cmd+=(--disable_tta)
fi

if [[ "${SAVE_PROBABILITIES:-0}" == "1" ]]; then
    cmd+=(--save_probabilities)
fi

if [[ "${CONTINUE_PREDICTION:-0}" == "1" ]]; then
    cmd+=(--continue_prediction)
fi

if [[ "${VERBOSE:-0}" == "1" ]]; then
    cmd+=(--verbose)
fi

if [[ "${DISABLE_PROGRESS_BAR:-0}" == "1" ]]; then
    cmd+=(--disable_progress_bar)
fi

exec "${cmd[@]}" "$@"
