#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data/registration/Learn2Reg/AbdominalCT/AbdomenCTCT}"
CHECKPOINT="${CHECKPOINT:-${FLEXICT_CHECKPOINT:-${FLEXICT_2D_CHECKPOINT:-}}}"
OUT_DIR="${ROOT}/results/2d_registration/ct_ct_finalckpt_5fold"
EXP_NOTE="final_p8_ctct_original_params"
LOG_FILE="${OUT_DIR}/run_5fold.log"

mkdir -p "${OUT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT, FLEXICT_CHECKPOINT, or FLEXICT_2D_CHECKPOINT}"

echo "Root: ${ROOT}"
echo "Python: ${PYTHON}"
echo "Dataset: ${DATASET_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Experiment: ${EXP_NOTE}"
echo "Log: ${LOG_FILE}"

cd "${ROOT}"

"${PYTHON}" downstream/2d_registration/run_ours_5_fold.py \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --exp_note "${EXP_NOTE}" \
  --feature_size 80,70 \
  --batch_size 32 \
  --num_iter 500 \
  --smooth_weight 1 \
  --iter_smooth_num 2 \
  --iter_smooth_kernel 7 \
  --gap 1 \
  --fix_preprocess meddinov3_ct \
  --mov_preprocess meddinov3_ct \
  2>&1 | tee "${LOG_FILE}"

echo "Done. Results saved to ${OUT_DIR}"
