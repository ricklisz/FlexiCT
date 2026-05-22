# 2D Classification Downstream Task

Use this guide to reproduce the Flexi_CT_2D frozen-feature classification
workflow. Run commands from the repository root after installing the base
environment in [`../../readme.md`](../../readme.md) and downloading the
`FlexiCT-2D` checkpoint.

## What this workflow runs

The scripts currently support the released `Flexi_CT_2D` wrapper on:

- `kits`
- `deep-lesion`
- `covidx`

The `config/` YAML files are included as reference settings from the original
classification experiments. The reproduction commands below use the Python
scripts in this directory and focus on the FlexiCT 2D runs.

Checkpoint resolution follows the shared downstream policy: pass
`--checkpoint` for a run-specific path, or set `FLEXICT_2D_CHECKPOINT` for the
default 2D teacher. `FLEXICT_CHECKPOINT` overrides all model-specific variables.

```bash
export FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth
```

Feature extraction downloads the CuriaBench datasets and processor through
Hugging Face on first use. For shared or offline systems, point `HF_HOME` to the
cache you want to use before running extraction:

```bash
export HF_HOME=/path/to/huggingface_cache
```

## Extract Features

```bash
python downstream/2d_classify/extract_features_for_sweep.py \
  --dataset all \
  --output_dir features/2d_classify \
  --batch_size 64
```

Outputs are saved as:

```text
features/2d_classify/flexi/{dataset}/{split}_features.npy
features/2d_classify/flexi/{dataset}/{split}_labels.npy
```

Case IDs are saved as `{split}_case_ids.npy` when the dataset has `series_id`.

## Run the Sweep

```bash
python downstream/2d_classify/data_efficiency_sweep.py \
  --features_dir features/2d_classify \
  --output_csv results/2d_classify/data_efficiency_results.csv
```

For quick debugging:

```bash
python downstream/2d_classify/data_efficiency_sweep.py \
  --dataset kits \
  --fractions 1.0 \
  --lrs 0.01 \
  --epochs 1 \
  --n_bootstrap 10
```
