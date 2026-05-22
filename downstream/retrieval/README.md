# Retrieval Reproduction

Run commands from the repository root after installing the base environment in
[`readme.md`](../../readme.md). The retrieval scripts evaluate frozen FlexiCT
features; they do not train or fine-tune the backbone.

## Checkpoints

The scripts default to `--ckpt_type vlm`. Pass a checkpoint explicitly or set
one of the environment variables:

```bash
export FLEXICT_VLM_CHECKPOINT=/path/to/ct_3d_vlm.pth
export FLEXICT_3D_CHECKPOINT=/path/to/ct_3d_teacher.pth
```

`--ckpt_type vlm` uses `FLEXICT_VLM_CHECKPOINT`; `--ckpt_type teacher` uses
`FLEXICT_3D_CHECKPOINT`. `--checkpoint /path/to/file.pth` overrides both.

## Data

The reproduction CSVs used by these scripts are bundled in
[`csv/`](csv/). They fix the case lists, labels, lesion coordinates, and
clinical fields used for the reported runs.

RADIO T-stage:

- Default annotations CSV: `csv/radio_annotations_fixed.csv`
- Default metadata CSV: `csv/radio_nsclc_radiogenomics.csv`
- Download helper files: `download_data/radio.sh` and
  `download_data/nsclc_radiogenomics.csv`
- The bundled annotations expect processed CT files at
  `data/radio/<study_uid>/image.nii.gz`, relative to the repository root.
  The RADIO retrieval scripts read these CT NIfTI files only.

Download RADIO imaging and segmentations from the official
[NSCLC-Radiogenomics TCIA collection](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/).
TCIA/NBIA Data Retriever works. For direct object storage download, install
`s5cmd` and run the bundled IDC manifest downloader:

```bash
mkdir -p tmp/radio_repro
bash downstream/retrieval/download_data/radio.sh tmp/radio_repro/dicom
```

The bundled helper script uses `nsclc_radiogenomics.csv` to download the RADIO
DICOM files and annotations. It rebuilds `radio_manifest.s5cmd` on each run, so
reruns do not append stale commands.

After download, convert each CT study listed in `csv/radio_annotations_fixed.csv`
to the NIfTI layout consumed by the retrieval scripts:

```bash
python downstream/retrieval/prepare_radio_data.py \
  --dicom_root tmp/radio_repro/dicom \
  --output_root tmp/radio_repro/radio_nifti \
  --output_annotations_csv tmp/radio_repro/radio_annotations_prepared.csv
```

Pass the prepared annotations CSV to the RADIO commands below. If you instead
write NIfTI files to `data/radio/<study_uid>/image.nii.gz`, the bundled
annotations CSV can be used directly.

C4KC-KiTS ISUP:

- Default clinical CSV: `csv/C4KC-KiTS_final.csv`
- Image directory: `--image_dir` or `C4KC_IMAGE_DIR`
  - Expected files: `{patient_id}_ct.nii.gz` and `{patient_id}_seg.nii.gz`
  - Segmentation labels: tumor/mass `2`; kidney fallback `1`

Download C4KC-KiTS imaging and segmentations from the official
[C4KC-KiTS TCIA collection](https://www.cancerimagingarchive.net/collection/c4kc-kits/),
using the TCIA/NBIA Data Retriever. The reproduction CSV is already bundled;
after download, convert or arrange the CT and segmentation files into the names
expected by the scripts:

```text
data/C4KC/<patient_id>_ct.nii.gz
data/C4KC/<patient_id>_seg.nii.gz
```

Use the `patient_id` values from `csv/C4KC-KiTS_final.csv`. If you store the
files elsewhere, pass that directory with `--image_dir` or set
`C4KC_IMAGE_DIR`.

Expected CSV sanity checks:

- RADIO: 144 valid T-stage cases from `csv/radio_annotations_fixed.csv`.
- C4KC-KiTS: 142 clear-cell RCC cases with ISUP grade before local file and
  segmentation-geometry checks.

## Rank-Fusion Retrieval

RADIO T-stage retrieval, using two lesion crop sizes and reciprocal rank
fusion:

```bash
python downstream/retrieval/ours_radio_retrieval.py \
  --ckpt_type vlm \
  --annotations_csv tmp/radio_repro/radio_annotations_prepared.csv \
  --roi_size_small 32 \
  --roi_size_large 64 \
  --rrf_k 60 \
  --seed 42 \
  --n_bootstrap 1000 \
  --output_dir results/retrieval/radio_T_stage/ours_rank_fusion
```

C4KC-KiTS ISUP retrieval, using two adaptive tumor crop scales and reciprocal
rank fusion:

```bash
python downstream/retrieval/ours_C4KC_retrieval.py \
  --ckpt_type vlm \
  --image_dir data/C4KC \
  --crop_scale_small 1.5 \
  --crop_scale_large 2.0 \
  --rrf_k 20 \
  --seed 42 \
  --n_bootstrap 1000 \
  --output_dir results/retrieval/C4KC_ISUP_grade/ours_adaptive_crop_rrf
```

Both retrieval scripts create `retrieval_results.json` with Recall@1/3/5,
Precision@1/3/5, mAP, random baselines, split metadata, and per-query top
retrievals. The query/support split is a seeded stratified 50/50 split.

## Linear-Probe Controls

RADIO T-stage frozen-feature linear probe:

```bash
python downstream/retrieval/ours_radio_linear.py \
  --ckpt_type vlm \
  --annotations_csv tmp/radio_repro/radio_annotations_prepared.csv \
  --roi_sizes 32 64 \
  --n_folds 5 \
  --n_repeats 2 \
  --seed 42 \
  --output_dir results/linear_probe/radio_T_stage/ours_sklearn_hier
```

C4KC-KiTS ISUP frozen-feature linear probe:

```bash
python downstream/retrieval/ours_C4KC_linear.py \
  --ckpt_type vlm \
  --image_dir data/C4KC \
  --crop_scale 1.5 \
  --n_folds 5 \
  --n_repeats 20 \
  --seed 42 \
  --output_dir results/retrieval/linear_probe/C4KC_ISUP_grade/ours_adaptive_crop
```

Linear probes create `linear_probe_results.json` with balanced accuracy, macro
F1, macro AUC, macro PR AUC, selected `C`, confusion matrices, and repeated-CV
confidence intervals.

The RADIO linear-probe script defaults to hierarchical intermediate-block
features. The retrieval scripts and C4KC-KiTS linear probe default to projected
VLM features unless `--use_hier` is passed.

## Multi-GPU

All four scripts support distributed feature extraction through `torchrun`; only
rank 0 writes the final JSON file.

```bash
torchrun --nproc_per_node=4 downstream/retrieval/ours_radio_retrieval.py ...
```
