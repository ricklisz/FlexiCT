# Segmentation

This directory contains FlexiCT segmentation workflows built on nnU-Net v2. The
wrapper scripts use the bundled nnU-Net source tree in `nnUNet/` so the FlexiCT
trainers are selected consistently. Start here for FlexiCT segmentation
reproduction; the READMEs inside `nnUNet/` are upstream nnU-Net reference notes.

## Setup

Install the repository dependencies from the root README, download the relevant
FlexiCT checkpoint, and prepare your data in the standard nnU-Net v2 format.

Install the bundled nnU-Net package in editable mode so the `nnUNetv2_*`
commands used by the wrapper scripts are available:

```bash
python -m pip install -e downstream/segmentation/nnUNet
```

Set the nnU-Net data locations before running the wrappers:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

Set the FlexiCT checkpoint for the model you plan to train:

```bash
export FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth
export FLEXICT_3D_CHECKPOINT=/path/to/ct_3d_teacher.pth
```

## Trainers

Choose the nnU-Net configuration with `CONFIG` and the FlexiCT trainer with
`TRAINER`. The FlexiCT trainer classes live in
`nnUNet/nnunetv2/training/nnUNetTrainer/flexictTrainer.py`.

| Trainer | Decoder | Runtime patch size |
|---|---|---:|
| `flexict_patch8_Primus_v2_Trainer` | Primus v2 | 8 |
| `flexict_patch16_Primus_v2_Trainer` | Primus v2 | 16 |
| `flexict_patch8_multiscale_Trainer` | Primus multiscale | 8 |
| `flexict_patch16_multiscale_Trainer` | Primus multiscale | 16 |
| `AutoPET_FlexiCTTrainer` | Primus v2, AutoPET augmentation | 8 |

`FlexiCTTrainer` remains available as a compatibility alias for the patch-8
Primus v2 trainer. For new runs, use `flexict_patch8_Primus_v2_Trainer` so the
trainer name is consistent with the other explicit variants.

Use `CONFIG=2d` for slice-based training and `CONFIG=3d_fullres` for volumetric
training. Patch-16 trainers load the released patch-8 FlexiCT checkpoint first,
then use runtime patch-size adaptation in the backbone.

Use `AutoPET_FlexiCTTrainer` for AutoPETII 2023 runs. It uses the same patch-8
Primus v2 FlexiCT network as `FlexiCTTrainer`, but applies AutoPET-specific
misalignment augmentation and the lower optimizer settings used by the explicit
FlexiCT trainer variants.

### Decoder differences

All FlexiCT segmentation trainers attach a Primus-style decoder to a FlexiCT
teacher backbone and use the same interaction layers `[3, 7, 11, 15]`.

Primus v2 concatenates those four intermediate feature maps, compresses the
concatenated channels back to the backbone embedding width with a 1x1
projection plus normalization, then applies the patch decoder. This keeps the
upsampling decoder width fixed at the backbone embedding dimension.

Primus multiscale concatenates the same intermediate feature maps but feeds the
full concatenated channel stack directly into the patch decoder, without the
Primus v2 projection bottleneck. This gives the decoder direct access to all
selected scales at a wider input channel width.

## Plan and preprocess

```bash
DATASET_ID=100 CONFIG=3d_fullres \
  bash downstream/segmentation/run_plan_preprocess.sh
```

Add `VERIFY_DATASET_INTEGRITY=1` if you want nnU-Net to validate the dataset
layout before preprocessing.

## Train

2D example:

```bash
DATASET_ID=100 CONFIG=2d FOLD=0 \
TRAINER=flexict_patch8_Primus_v2_Trainer \
  bash downstream/segmentation/run_train.sh
```

3D example:

```bash
DATASET_ID=100 CONFIG=3d_fullres FOLD=0 \
TRAINER=flexict_patch16_multiscale_Trainer \
  bash downstream/segmentation/run_train.sh
```

## Predict

```bash
INPUT=/path/to/imagesTs OUTPUT=/path/to/predictions \
DATASET_ID=100 CONFIG=3d_fullres \
TRAINER=flexict_patch16_multiscale_Trainer \
  bash downstream/segmentation/run_predict.sh
```

To predict from an exported nnU-Net model folder, set `MODEL_FOLDER` instead of
`DATASET_ID`, `CONFIG`, and `TRAINER`.

## Evaluate

Use the standard nnU-Net evaluator when `dataset.json` and `plans.json` are
available:

```bash
GT_FOLDER=/path/to/labelsTs PRED_FOLDER=/path/to/predictions \
DATASET_JSON=/path/to/dataset.json PLANS_JSON=/path/to/plans.json \
  bash downstream/segmentation/run_evaluate.sh
```

For a simple foreground-label evaluation:

```bash
GT_FOLDER=/path/to/labelsTs PRED_FOLDER=/path/to/predictions LABELS="1 2 3" \
  bash downstream/segmentation/run_evaluate.sh
```

## Reproducing reported FlexiCT results

After converting a dataset to nnU-Net v2 format and setting the environment
variables above, preprocess the same configuration used by the reported FlexiCT
run. The `3d_vit` configuration is a custom plan entry: it does not exist in the default
nnU-Net plans until you add it.

For the `3d_vit` rows, first create the regular residual-encoder plans without
preprocessing. Keep `PLANNER=nnUNetPlannerResEncL`; these runs use the
residual-encoder plans file even after the custom `3d_vit` entry is added:

```bash
for id in 003 006 007 140 221 990; do
  DATASET_ID="$id" PLANNER=nnUNetPlannerResEncL NO_PREPROCESS=1 \
    bash downstream/segmentation/run_plan_preprocess.sh
done
```

Then edit each generated `nnUNetResEncUNetLPlans.json`: add a new `3d_vit`
configuration with `inherits_from: 3d_fullres`, set `data_identifier` to
`nnUNetPlans_3d_vit`, and set the patch size and spacing from the table below.
After that, run the corresponding preprocess command.

For TotalSegmentator v2 and AMOS, use the default nnU-Net 2D config:

```bash
DATASET_ID=100 CONFIG=2d \
  bash downstream/segmentation/run_plan_preprocess.sh

DATASET_ID=219 CONFIG=2d \
  bash downstream/segmentation/run_plan_preprocess.sh
```

| Dataset | Preprocess command | Patch size | Spacing |
|---|---|---|---|
| Liver | `nnUNetv2_preprocess -d 003 -c 3d_vit -plans_name nnUNetResEncUNetLPlans` | `[224, 192, 192]` | `[0.767578125, 0.767578125, 0.767578125]` |
| Lung | `nnUNetv2_preprocess -d 006 -c 3d_vit -plans_name nnUNetResEncUNetLPlans` | `[192, 192, 192]` | `[0.72623832027, 0.78515625, 0.78515625]` |
| Pancreas | `nnUNetv2_preprocess -d 007 -c 3d_vit -plans_name nnUNetResEncUNetLPlans` | `[176, 224, 224]` | `[0.8027340173721313, 0.8027340173721313, 0.8027340173721313]` |
| TotalSegmentator v2 | `nnUNetv2_preprocess -d 100 -c 2d -plans_name nnUNetPlans` | `[256, 256]` | `[1.5, 1.5]` |
| KiTS23 | `nnUNetv2_preprocess -d 140 -c 3d_vit -plans_name nnUNetResEncUNetLPlans` | `[208, 224, 192]` | `[0.78125, 0.78125, 0.78125]` |
| AMOS2022 task 2 | `nnUNetv2_preprocess -d 219 -c 2d -plans_name nnUNetPlans` | `[640, 640]` | `[0.712890625, 0.712890625]` |
| AutoPETII 2023 | `nnUNetv2_preprocess -d 221 -c 3d_vit -plans_name nnUNetResEncUNetLPlans` | `[288, 192, 192]` | `[2.0364201068878174, 2.0364201068878174, 2.0364201068878174]` |
| WORD | `nnUNetv2_preprocess -d 990 -c 3d_vit -plans_name nnUNetResEncUNetLPlans` | `[288, 224, 224]` | `[0.9765625, 0.9765625, 0.9765625]` |
