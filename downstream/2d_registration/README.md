# 2D Registration Downstream Task

Use this guide to reproduce the Flexi_CT_2D Learn2Reg registration experiments.
Run commands from the repository root after installing the base environment and
downloading the `FlexiCT-2D` checkpoint.

## Data Download and Format

Download the pre-2022 Learn2Reg Abdomen MR-CT and/or Abdomen CT-CT data from
the official Learn2Reg datasets page:

```text
https://learn2reg.grand-challenge.org/Datasets/
```

Use the abdomen task downloads, unpack them, and point `DATASET_DIR` at the
task root:

```text
/path/to/Learn2reg/AbdomenMRCT/AbdomenMRCT
/path/to/Learn2reg/AbdominalCT/AbdomenCTCT
```

The runner assumes the preprocessed Learn2Reg NIfTI layout:

```text
DATASET_DIR/
  imagesTr/
  labelsTr/
  imagesTs/              # optional for this runner
  pairs_Val.csv
  pairs_Tr.csv           # optional
  pairs_Ts.csv           # optional
  pairs_Test.csv         # optional
  structures.csv
```

Each pair CSV row should be headerless and contain the moving filename first and
the fixed filename second. The referenced image files must exist in `imagesTr/`,
and matching segmentation files with the same filenames must exist in
`labelsTr/`; pairs with missing files are skipped. The code assumes the
preprocessed challenge files, so images and labels should already share the
registration grid used by the Learn2Reg task. `structures.csv` is read from its
first row and should contain the integer label IDs used for Dice and HD95.
Optional `masksTr/` files are only used when `--use_masks_tr` is passed.

## CT-MR Final Checkpoint Run

Use the training/runtime environment's `flexict` Python and run from the
repository root. Set the checkpoint and dataset path explicitly unless you have
staged the dataset at the wrapper's default `data/registration/...` path:

```bash
FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth \
DATASET_DIR=/path/to/Learn2reg/AbdomenMRCT/AbdomenMRCT \
bash downstream/2d_registration/run_ct_mr_final_params.sh
```

The wrapper loads `pairs_Val.csv`, `pairs_Tr.csv`, `pairs_Ts.csv`, and
`pairs_Test.csv` when present, validates image/label files under `imagesTr` and
`labelsTr`, then runs 5-fold cross validation with
`KFold(n_splits=5, shuffle=True, random_state=42)`.

It uses:

- checkpoint: `--checkpoint` when supplied, otherwise `FLEXICT_CHECKPOINT`,
  then `FLEXICT_2D_CHECKPOINT`; no checkpoint path is bundled
- dataset: `DATASET_DIR` or `data/registration/Learn2Reg/AbdomenMRCT/AbdomenMRCT`
- `feature_size=80,70`
- `gap=1`
- `num_iter=500`
- `smooth_weight=2`
- `iter_smooth_num=2`
- `iter_smooth_kernel=7`
- fixed preprocessing: `mr`
- moving preprocessing: `meddinov3_ct`

Outputs and logs are saved under:

```text
results/2d_registration/ct_mr_finalckpt_5fold
```

After all folds finish, the script writes aggregate files:

```text
DICE_all_folds_final_p8_original_params.txt
HD95_all_folds_final_p8_original_params.txt
LogJacDetStd_all_folds_final_p8_original_params.txt
summary_all_folds_final_p8_original_params.txt
```

## Direct Single-Pair Run

For debugging one case:

```bash
python downstream/2d_registration/run_ours_5_fold.py \
  --dataset_dir /path/to/Learn2reg/AbdomenMRCT/AbdomenMRCT \
  --output_dir results/2d_registration/debug_pair \
  --checkpoint /path/to/ct_2d_teacher.pth \
  --exp_note final_p8_original_params \
  --feature_size 80,70 \
  --batch_size 32 \
  --num_iter 500 \
  --smooth_weight 2 \
  --iter_smooth_num 2 \
  --iter_smooth_kernel 7 \
  --gap 1 \
  --fix_preprocess mr \
  --mov_preprocess meddinov3_ct \
  --only_pair_index 0
```

For a full memory-safe rerun that executes cases one at a time instead of true
5-fold grouping:

```bash
FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth \
DATASET_DIR=/path/to/Learn2reg/AbdomenMRCT/AbdomenMRCT \
bash downstream/2d_registration/run_ct_mr_final_params_pairs.sh
```

## CT-CT Final Checkpoint Run

For the AbdominalCT / CT-CT Learn2Reg task:

```bash
FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth \
DATASET_DIR=/path/to/Learn2reg/AbdominalCT/AbdomenCTCT \
bash downstream/2d_registration/run_ct_ct_final_params.sh
```

The wrapper also accepts `CHECKPOINT` or `FLEXICT_CHECKPOINT`; those take
precedence over `FLEXICT_2D_CHECKPOINT`. Its default CT-CT task root is:

```text
data/registration/Learn2Reg/AbdominalCT/AbdomenCTCT
```

The CT-CT wrapper also loads all standard pair CSVs when present, validates
against `imagesTr` and `labelsTr`, then runs 5-fold cross validation with the
same fixed seed.
It uses:

- `feature_size=80,70`
- `gap=1`
- `num_iter=500`
- `smooth_weight=1`
- `iter_smooth_num=2`
- `iter_smooth_kernel=7`
- fixed preprocessing: `meddinov3_ct`
- moving preprocessing: `meddinov3_ct`

Outputs and logs are saved under:

```text
results/2d_registration/ct_ct_finalckpt_5fold
```

For a memory-safe per-pair CT-CT fallback, including subset testing:

```bash
FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth \
DATASET_DIR=/path/to/Learn2reg/AbdominalCT/AbdomenCTCT \
PAIR_INDICES="0 1 2" bash downstream/2d_registration/run_ct_ct_final_params_pairs.sh
```
