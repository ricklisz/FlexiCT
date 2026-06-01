# FlexiCT Pretraining

This repository contains FlexiMedDINOv3/FlexiCT pretraining code adapted from DINOv3 for CT foundation model training. It follows the staged FlexiCT process: 2D axial CT pretraining, high-resolution 2D continuation, 3D anatomical pretraining, and report-guided semantic alignment.

## Repository Layout

- `dinov3/models/`: FlexiMedDINOv3 vision transformer and 2D/3D patch embedding.
- `dinov3/train/`: training entry points for 2D CT, 3D CT, and TIPS/OSL alignment.
- `dinov3/configs/train/`: phase-specific YAML configs.
- `dinov3/data/`: CT datasets, preprocessing, augmentations, masks, and collation.
- `dinov3/loss/`: DINO, iBOT, KoLeo, and Gram losses.
- `dinov3/eval/text/`: text encoder and CLIP-style alignment utilities.

## Installation

Use Linux, Python 3.11+, CUDA-capable GPUs, and CUDA 12.8 PyTorch 2.8 wheels. `requirements.txt` pins `torch==2.8.0+cu128` and `torchvision==0.23.0+cu128`; do not replace them with CPU wheels.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

If `python3.11 -m venv` fails because `ensurepip` is unavailable, create the environment with `virtualenv` instead:

```bash
virtualenv -p /usr/bin/python3.11 .venv
```

Verify CUDA PyTorch before training:

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
assert torch.__version__.endswith("+cu128")
assert torch.version.cuda == "12.8"
PY
```

Phase 3 uses a Hugging Face text encoder from `vlm.hf_model_name_or_path`. The configs default to FlashAttention 2; install it after PyTorch, or set `vlm.hf_use_flash_attention=false`.

```bash
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

## Data Preparation

### Phase 1: 2D CT Slices

The 2D script reads LMDB slice datasets via `LMDBSliceDatasetv2`. Each LMDB stores `<key>.npy`, optional `<key>.json`, and `__index__`; a separate `<dataset>_slices.index.txt` is also supported. Slice arrays may be `(H, W)`, channel-first, or channel-last with 1 or 3 channels.

Set LMDB roots before launch:

```bash
export FLEXICT_LMDB_ROOT_V3=/path/to/lmdb/ct_v3
export FLEXICT_LMDB_ROOT_V2=/path/to/lmdb/ct_v2
```

Expected layout:

```text
${FLEXICT_LMDB_ROOT_V3}/<dataset>.lmdb
${FLEXICT_LMDB_ROOT_V3}/<dataset>_slices.index.txt
${FLEXICT_LMDB_ROOT_V2}/<dataset>.lmdb
${FLEXICT_LMDB_ROOT_V2}/<dataset>_slices.index.txt
```

Edit dataset lists in `dinov3/train/train_ct_2D_adaptive_patch.py` for your local corpus. `dinov3/data/preprocess_totalseg.py` converts TotalSegmentator-style `imagesTr`/`labelsTr` NIfTI data into this LMDB format: it resamples volumes, labels axial slices by anatomy from `dataset.json`, and writes `<dataset>.lmdb`, `<dataset>_manifest.csv`, and `<dataset>_slices.index.txt`.

```bash
PYTHONPATH=$PWD python dinov3/data/preprocess_totalseg.py \
  --src-dir /path/to/TotalSegmentator/imagesTr \
  --dst-root /path/to/lmdb/ct_v2 \
  --dataset-name TotalSegmentatorv2 \
  --label-map-json /path/to/TotalSegmentator/dataset.json \
  --spacing 1.5 1.5 1.5 \
  --resize-to-256 \
  --sample-half
```

### Phase 1.1: High-Resolution 2D Data

Uses the same LMDB slice format as Phase 1. For `fleximeddinov3_b_ct_highres.yaml`, enable `build_multi_resolution_data_loader_from_cfg` in `train_ct_2D_adaptive_patch.py` when using list-valued crop sizes.

### Phase 2: 3D CT Volumes

The 3D script reads NIfTI volumes through `CT3D`. Place `paths_filtered4.json` in the launch working directory:

```json
{
  "paths": [
    "/path/to/volume_0001.nii.gz",
    "/path/to/volume_0002.nii.gz"
  ]
}
```

Volumes are loaded with SimpleITK, resampled to `(2.0, 2.0, 2.0)`, converted to `(1, D, H, W)`, CT-normalized, and augmented with 3D random crops.

### Phase 3: Report Alignment Data

The TIPS/OSL script reads paired CT volumes and reports from CSV files with `file_path`, `final_report`, and `short_captions`. `short_captions` is a JSON string keyed by anatomical section with `positive_findings` and `negative_findings`.

```bash
export FLEXICT_REPORT_CSVS=/path/to/reports/train_final_report.csv
```

Use comma-separated paths for multiple report manifests.

## Pretraining

Use `PYTHONPATH=$PWD`. `dinov3.run.submit` launches through submitit/Slurm; `torchrun --standalone` is for single-node runs. Set `--nproc_per_node` or `--ngpus` to the local GPU count.

### Phase 1: 2D Axial CT Pretraining

Trains FlexiMedDINOv3-Base on 2D axial CT slices with DINO, iBOT, and KoLeo. Config: `dinov3/configs/train/fleximeddinov3_b_ct_pretrain.yaml`. Key settings: `student.in_chans: 1`, `student.dinov3_weights: imagenet_model.pth`, global crop `256`, local crop `112`, runtime patch sizes 16 and 8.

Slurm/submitit:

```bash
PYTHONPATH=$PWD python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus 8 \
  --output-dir /path/to/outputs/flexict_2d \
  dinov3/train/train_ct_2D_adaptive_patch.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_pretrain.yaml
```

Single-node:

```bash
PYTHONPATH=$PWD torchrun \
  --standalone \
  --nproc_per_node=8 \
  dinov3/train/train_ct_2D_adaptive_patch.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_pretrain.yaml \
  --output-dir /path/to/outputs/flexict_2d
```

### Phase 1.1: High-Resolution 2D Continuation

Continues from Phase 1 and adds Gram teacher alignment at larger resolutions. Config: `dinov3/configs/train/fleximeddinov3_b_ct_highres.yaml`. Update `student.resume_from_teacher_chkpt`, `gram.ckpt`, and the multi-resolution loader before launch.

Slurm/submitit:

```bash
PYTHONPATH=$PWD python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus 8 \
  --output-dir /path/to/outputs/flexict_2d_highres \
  dinov3/train/train_ct_2D_adaptive_patch.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_highres.yaml
```

Single-node:

```bash
PYTHONPATH=$PWD torchrun \
  --standalone \
  --nproc_per_node=8 \
  dinov3/train/train_ct_2D_adaptive_patch.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_highres.yaml \
  --output-dir /path/to/outputs/flexict_2d_highres
```

### Phase 2: 3D Anatomical Pretraining

Initializes the 3D model from 2D weights, inflates the patch embedding, and trains on volumetric crops. Config: `dinov3/configs/train/fleximeddinov3_b_ct_3D_randcrop_pretrain.yaml`. Key settings: `student.dinov3_weights: 2D_model.pth`, global crop `[160, 160, 160]`, local crop `[80, 80, 80]`, RCC 3D masking, runtime patch sizes 16 and 8.

Slurm/submitit:

```bash
PYTHONPATH=$PWD python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus 8 \
  --output-dir /path/to/outputs/flexict_3d \
  dinov3/train/train_ct_3D_adaptive_patch.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_3D_randcrop_pretrain.yaml
```

Single-node:

```bash
PYTHONPATH=$PWD torchrun \
  --standalone \
  --nproc_per_node=8 \
  dinov3/train/train_ct_3D_adaptive_patch.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_3D_randcrop_pretrain.yaml \
  --output-dir /path/to/outputs/flexict_3d
```

If resuming from an already trained 3D checkpoint, review whether 2D-to-3D patch inflation should still run.

### Phase 3: Report-Guided Semantic Alignment

Trains CT vision features with report text using TIPS/OSL: one global crop, DINO local loss, iBOT, CLIP-style contrastive loss, and opposite sentence loss. Config: `dinov3/configs/train/tips/clip_osl.yaml`. Key settings: `student.resume_from_teacher_chkpt: 3D_model.pth`, `vlm.enabled: true`, `vlm.osl_enabled: true`, Qwen3 embedding text encoder, one global crop, and eight local crops.

Slurm/submitit:

```bash
PYTHONPATH=$PWD python -m dinov3.run.submit \
  --nodes 1 \
  --ngpus 8 \
  --output-dir /path/to/outputs/flexict_tips_osl \
  dinov3/train/train_tips_osl.py \
  --config-file dinov3/configs/train/tips/clip_osl.yaml
```

Single-node:

```bash
PYTHONPATH=$PWD torchrun \
  --standalone \
  --nproc_per_node=8 \
  dinov3/train/train_tips_osl.py \
  --config-file dinov3/configs/train/tips/clip_osl.yaml \
  --output-dir /path/to/outputs/flexict_tips_osl
```

## Checkpoint Conversion

Training saves distributed checkpoints under `<output-dir>/ckpt/<iteration>`. Convert a checkpoint to a regular teacher `.pth` file:

```bash
PYTHONPATH=$PWD python dinov3/train/convert_checkpoint.py \
  --config-file dinov3/configs/train/fleximeddinov3_b_ct_3D_randcrop_pretrain.yaml \
  --output-dir /path/to/outputs/flexict_3d \
  --ckpt-dir /path/to/outputs/flexict_3d/ckpt/<iteration> \
  --out /path/to/3D_model.pth
```

Use converted checkpoint paths in later phase configs.

## Notes

- Placeholder paths are sanitized and will not run until replaced.
- `train.dataset_path` is not the CT data source of truth; use the environment variables and script-level dataset lists above.
- Public files do not include the full dataset inventory used in the paper.
- High-resolution 2D continuation needs the multi-resolution loader for list-valued crop sizes.

## Development Checks

```bash
python -m py_compile dinov3/train/train_ct_2D_adaptive_patch.py \
  dinov3/train/train_ct_3D_adaptive_patch.py \
  dinov3/train/train_tips_osl.py

ruff check dinov3
mypy dinov3
```
