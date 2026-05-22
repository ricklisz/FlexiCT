# `flexi_ct` package

This directory contains the lightweight FlexiCT model package used by the demo
notebooks and downstream workflows.

## Public model wrappers

The main entry points are:

| Class | Input | Output |
|---|---|---|
| `Flexi_CT_2D` | 2D CT slices, `[B, 1, H, W]` | CLS token and patch tokens |
| `Flexi_CT_3D` | 3D CT volumes, `[B, 1, D, H, W]` | CLS token and patch tokens |
| `Flexi_CT_VLM` | 3D CT volumes and text prompts | image/text embeddings and similarity scores |

The released checkpoints use a base patch size of 8. Input spatial dimensions
should be divisible by the active patch size.

## Checkpoint resolution

Model constructors accept an explicit `checkpoint_path`. If omitted, checkpoint
paths are resolved through:

1. `FLEXICT_CHECKPOINT`
2. `FLEXICT_2D_CHECKPOINT`, `FLEXICT_3D_CHECKPOINT`, or
   `FLEXICT_VLM_CHECKPOINT`

There is no bundled default checkpoint path.

## Adapting patch size

There are two common cases.

### Use a released checkpoint with a different runtime patch size

Instantiate the model with the released base patch size, load the checkpoint,
then set the runtime patch size on the patch embedding module. The convolutional
patch kernel is resampled on the fly without changing checkpoint parameter
shapes.

```python
from flexi_ct import Flexi_CT_2D, Flexi_CT_3D, Flexi_CT_VLM

model_2d = Flexi_CT_2D(checkpoint_path="/path/to/ct_2d_teacher.pth")
model_2d.backbone.patch_embed_2D.set_patch_size(16)
model_2d.backbone.patch_size = 16

model_3d = Flexi_CT_3D(checkpoint_path="/path/to/ct_3d_teacher.pth")
model_3d.backbone.patch_embed_3D.set_patch_size(16)
model_3d.backbone.patch_size = 16

vlm = Flexi_CT_VLM(checkpoint_path="/path/to/ct_3d_vlm.pth")
vlm.model.vision_model.patch_embed_3D.set_patch_size(16)
vlm.model.vision_model.patch_size = 16
```

Use an integer for isotropic patches. `PatchEmbedND.set_patch_size(...)` also
accepts a tuple for anisotropic 3D patches, such as `(8, 16, 16)`, for forward
passes that consume flattened patch tokens. The `patch_size` attribute is used
by helper methods such as `get_intermediate_layers(..., reshape=True)`, which
currently assume an isotropic integer patch size. For anisotropic 3D patches,
avoid that helper reshape path or reshape tokens manually from the known output
grid.

### Build a new backbone with a different base patch size

For training or experiments without released checkpoint loading, pass
`patch_size` to the backbone factory:

```python
from flexi_ct.models import flexi_ct_backbone_base

backbone = flexi_ct_backbone_base(patch_size=16, in_chans=1)
```

Changing the base `patch_size` changes patch-embedding parameter shapes. Do not
edit the wrapper `_BACKBONE_KWARGS` and then expect the released checkpoints to
load with `strict=True`; use runtime patch-size adaptation instead.

## Layout

- `flexi_ct_2d.py`, `flexi_ct_3d.py`, `flexi_ct_vlm.py`: public wrappers.
- `models.py`: shared FlexiCT backbone, patch embedding, and VLM module.
- `checkpoints.py`: checkpoint-path resolution.
- `layers/`: transformer, attention, patch embedding, and RoPE components.
- `utils/`: small shared utilities.
