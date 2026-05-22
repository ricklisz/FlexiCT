#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data/registration/Learn2Reg/AbdomenMRCT/AbdomenMRCT}"
CHECKPOINT="${CHECKPOINT:-${FLEXICT_CHECKPOINT:-${FLEXICT_2D_CHECKPOINT:-}}}"
OUT_DIR="${ROOT}/results/2d_registration/ct_mr_finalckpt_original_params_pairs"
EXP_NOTE="final_p8_original_params"

mkdir -p "${OUT_DIR}"

: "${CHECKPOINT:?Set CHECKPOINT, FLEXICT_CHECKPOINT, or FLEXICT_2D_CHECKPOINT}"
export OUT_DIR EXP_NOTE

echo "Root: ${ROOT}"
echo "Python: ${PYTHON}"
echo "Dataset: ${DATASET_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Experiment: ${EXP_NOTE}"

cd "${ROOT}"

for pair_idx in 0 1 2 3 4 5 6 7 8; do
  log_file="${OUT_DIR}/run_pair_${pair_idx}.log"
  echo "Starting pair ${pair_idx}; logging to ${log_file}"
  "${PYTHON}" downstream/2d_registration/run_ours_5_fold.py \
    --dataset_dir "${DATASET_DIR}" \
    --output_dir "${OUT_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --exp_note "${EXP_NOTE}" \
    --feature_size 80,70 \
    --batch_size 32 \
    --num_iter 500 \
    --smooth_weight 2 \
    --iter_smooth_num 2 \
    --iter_smooth_kernel 7 \
    --gap 1 \
    --fix_preprocess mr \
    --mov_preprocess meddinov3_ct \
    --only_pair_index "${pair_idx}" \
    > "${log_file}" 2>&1
  echo "Finished pair ${pair_idx}"
done

"${PYTHON}" - <<'PY'
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["OUT_DIR"])
exp = os.environ["EXP_NOTE"]

dice_rows = []
hd_rows = []
ljd_vals = []
for pair_dir in sorted(root.glob("pair_*"), key=lambda p: int(p.name.split("_")[1])):
    dice_file = next(pair_dir.glob(f"DICE_fold*_{exp}.txt"))
    hd_file = next(pair_dir.glob(f"HD95_fold*_{exp}.txt"))
    ljd_file = next(pair_dir.glob(f"LogJacDetStd_fold*_{exp}.txt"))
    dice_rows.append(np.loadtxt(dice_file, ndmin=2))
    hd_rows.append(np.loadtxt(hd_file, ndmin=2))
    ljd_vals.append(np.loadtxt(ljd_file, ndmin=1))

all_dice = np.concatenate(dice_rows, axis=0)
all_hd = np.concatenate(hd_rows, axis=0)
all_ljd = np.concatenate(ljd_vals, axis=0)

np.savetxt(root / f"DICE_all_folds_{exp}.txt", all_dice, fmt="%.4f")
np.savetxt(root / f"HD95_all_folds_{exp}.txt", all_hd, fmt="%.4f")
np.savetxt(root / f"LogJacDetStd_all_folds_{exp}.txt", all_ljd, fmt="%.3f")

summary = np.array([
    np.nanmean(all_dice), np.nanstd(all_dice),
    np.mean(all_ljd), np.std(all_ljd),
    np.nanmean(all_hd), np.nanstd(all_hd),
])
np.savetxt(
    root / f"summary_all_folds_{exp}.txt",
    summary,
    fmt="%.4f",
    header="DICE_mean DICE_std LogJacDetStd_mean LogJacDetStd_std HD95_mean HD95_std",
)
print("summary", " ".join(f"{x:.4f}" for x in summary))
PY

echo "Done. Results saved to ${OUT_DIR}"
