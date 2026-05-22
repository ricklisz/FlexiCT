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

: "${GT_FOLDER:?Set GT_FOLDER to the ground-truth segmentations folder}"
: "${PRED_FOLDER:?Set PRED_FOLDER to the prediction folder}"

if [[ -n "${DATASET_JSON:-}" && -n "${PLANS_JSON:-}" ]]; then
    use_simple=0
    cmd=(nnUNetv2_evaluate_folder "$GT_FOLDER" "$PRED_FOLDER" -djfile "$DATASET_JSON" -pfile "$PLANS_JSON")
elif [[ -n "${LABELS:-}" ]]; then
    use_simple=1
    read -r -a labels <<< "$LABELS"
    cmd=(nnUNetv2_evaluate_simple "$GT_FOLDER" "$PRED_FOLDER" -l "${labels[@]}")
else
    echo "Set DATASET_JSON and PLANS_JSON for nnUNetv2_evaluate_folder, or LABELS for nnUNetv2_evaluate_simple." >&2
    exit 2
fi

if [[ -n "${OUTPUT_JSON:-}" ]]; then
    cmd+=(-o "$OUTPUT_JSON")
fi

if [[ -n "${NUM_PROCESSES:-}" ]]; then
    cmd+=(-np "$NUM_PROCESSES")
fi

if [[ "$use_simple" == "1" && -n "${IGNORE_LABEL:-}" ]]; then
    cmd+=(-il "$IGNORE_LABEL")
fi

if [[ "${CHILL:-0}" == "1" ]]; then
    cmd+=(--chill)
fi

exec "${cmd[@]}" "$@"
