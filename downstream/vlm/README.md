# Vision-language evaluation

This directory contains downstream evaluation scripts for `FlexiCT-3D-VLM`.
The scripts evaluate CT-report retrieval and zero-shot disease classification on
CT-RATE-style and Merlin-style datasets.

## Setup

Run commands from the repository root after installing the dependencies from the
root README. Download the `FlexiCT-3D-VLM` checkpoint and make sure the Qwen3
embedding tokenizer/config is available through Hugging Face or a local
`HF_HOME` cache.

Set the VLM checkpoint once per shell session:

```bash
export FLEXICT_VLM_CHECKPOINT=/path/to/ct_3d_vlm.pth
```

You can also pass the checkpoint directly with `--pretrain`.
For shared or offline systems, set the Hugging Face cache location before
running evaluation:

```bash
export HF_HOME=/path/to/huggingface_cache
```

## Data manifests

The shared dataset loader expects a CSV manifest with these columns:

| Column | Description |
|---|---|
| `file_path` | Path to a CT volume, usually `.nii` or `.nii.gz`. |
| `short_captions` | Structured short captions, if available. |
| `final_report` | Report text used for retrieval and prompt construction. |

Evaluation scripts also require task-specific label CSVs for zero-shot
classification.

## CT-RATE evaluation

Set the CT-RATE manifest and label files:

```bash
export CT_RATE_REPORTS_CSV=/path/to/validation_reports.csv
export CT_RATE_LABELS_CSV=/path/to/valid_predicted_labels.csv
```

Reproduce the reported CT-RATE results:

```bash
python downstream/vlm/ct_rate_eval.py \
  -work_dir results/ct_rate/ours_final \
  -batch_size 40
```

The script reports image-text retrieval and zero-shot classification metrics,
and writes CSV outputs under the selected work directory.

## Merlin evaluation

Set the Merlin report manifest and provide the zero-shot label CSV:

```bash
export MERLIN_REPORTS_CSV=/path/to/merlin_reports.csv

python downstream/vlm/merlin_eval.py \
  --merlin_labels_csv /path/to/merlin_labels.csv \
  -work_dir results/merlin_vlm \
  -batch_size 40
```

The script evaluates report retrieval and Merlin-style zero-shot disease
classification.

