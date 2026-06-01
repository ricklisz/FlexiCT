#!/usr/bin/env python3
import os, io, csv, json, math, lmdb, numpy as np
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import tempfile, uuid
import argparse
import pandas as pd
from typing import Dict, List
import random
from collections import defaultdict

def _ids(label_map: Dict[str, int], names: List[str]) -> List[int]:
    return [label_map[n] for n in names if n in label_map]

def classify(
    seg_arr: np.ndarray,
    label_map: Dict[str, int],
):
    """
    seg_arr: (X, Y, Z) integer label volume aligned to CT image.
    label_map: {name -> id}
    Returns: list[str] of length Z, class per axial slice index.
    """

    # Define groups exactly as specified
    HEAD_NECK = _ids(
        label_map,
        [
            "common_carotid_artery_right",
            "common_carotid_artery_left",
            "brain",
            "skull",
            "thyroid_gland",
        ],
    )
    HEAD_NECK_AIRWAY = _ids(label_map, ["trachea", "esophagus"])
    CERV_IDS = _ids(label_map, [f"vertebrae_C{i}" for i in range(1,7)])

    LUNG_LOBES = _ids(
        label_map,
        [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ],
    )

    ABD_ORGANS = _ids(
        label_map,
        [
        "liver", "spleen", "stomach", "pancreas",
        "adrenal_gland_left", "adrenal_gland_right",
        "kidney_left", "kidney_right",
        "duodenum", "small_bowel", "colon",
        "gallbladder",
        ],
    )

    HIP = _ids(label_map, ["hip_left", "hip_right", "urinary_bladder", "prostate"])
    FEMUR = _ids(label_map, ["femur_left", "femur_right"])

    # Helper for presence on a slice
    def present(ids: List[int], sl: np.ndarray) -> bool:
        if not ids:
            return False
        return np.any(np.isin(sl, ids))

    sl = seg_arr
    has_femur = present(FEMUR, sl)
    has_hip = present(HIP, sl)
    has_lung = present(LUNG_LOBES, sl)
    has_abd = present(ABD_ORGANS, sl)
    # has_headneck = present(HEAD_NECK, sl) 
    has_headneck = (present(HEAD_NECK, sl) or (present(HEAD_NECK_AIRWAY, sl) and not has_lung) or present(CERV_IDS, sl))

    # Precedence per spec and to resolve overlaps deterministically
    if has_femur and not has_hip:
        label = "leg"
    elif has_hip:
        label = "pelvis"
    elif has_lung:
        label = "chest"
    elif (not has_lung) and (not has_hip) and has_abd:
        label = "abdominal"
    elif has_headneck:
        label = "head_and_neck"
    else:
        label = "none"
    return label

# ---------- your helpers (unchanged) ----------
def load_sitk_image(image_path):
    """
    Load image with SimpleITK; on 'non-orthonormal direction' error,
    read with nibabel and construct a SimpleITK image with a fixed header.
    """
    try:
        return sitk.ReadImage(image_path)
    except RuntimeError as e:
        if "No orthonormal definition found" not in str(e):
            raise

    # Fallback: nibabel -> SITK with corrected direction/spacing
    img = nib.load(image_path)

    # Prefer sform, else qform, else header-derived affine
    aff = img.get_sform()
    if aff is None or not np.any(aff):
        aff = img.get_qform()
    if aff is None or not np.any(aff):
        aff = img.affine  # nibabel’s best guess

    A = np.asarray(aff, dtype=np.float64)
    R = A[:3, :3]
    t = A[:3, 3]

    # spacing = column norms of R (voxel sizes)
    spacing = np.linalg.norm(R, axis=0)
    # guard against zeros
    spacing = np.where(spacing == 0, 1.0, spacing)

    # normalize columns to get (possibly non-orthogonal) direction
    Dinv = np.diag(1.0 / spacing)
    M = R @ Dinv  # should be rotation but may have shear/numerical issues

    # Polar decomposition to nearest rotation
    U, _, Vt = np.linalg.svd(M)
    direction = U @ Vt
    # Enforce right-handed (det=+1)
    if np.linalg.det(direction) < 0:
        U[:, -1] *= -1
        direction = U @ Vt

    # Build SITK image from array (note: SITK expects z,y,x ordering)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    sitk_img = sitk.GetImageFromArray(data)  # creates (z,y,x)

    # SITK expects spacing/origin/direction as (x,y,z)
    spacing_xyz = tuple(float(s) for s in spacing.tolist())
    origin_xyz = tuple(float(v) for v in t.tolist())
    direction_flat = tuple(direction.ravel(order="C").tolist())  # row-major 3x3 -> 9-tuple

    sitk_img.SetSpacing(spacing_xyz)
    sitk_img.SetOrigin(origin_xyz)
    sitk_img.SetDirection(direction_flat)

    return sitk_img

def resample_sitk_image(image_sitk, new_spacing_xy=(0.75, 0.75), is_label=False):
    """
    Resample only the axial plane (X,Y) to new_spacing_xy; keep Z unchanged.
    Reorients to LPS so axis 2 is axial.
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("LPS")
    image_sitk = orient_filter.Execute(image_sitk)

    original_spacing = image_sitk.GetSpacing()  # (sx, sy, sz)
    original_size    = image_sitk.GetSize()     # (nx, ny, nz)
    axial_axis = 2

    # new spacing: change X,Y; keep Z
    new_spacing = list(original_spacing)
    inplane_axes = [a for a in range(3) if a != axial_axis]  # [0,1]
    new_spacing[inplane_axes[0]] = float(new_spacing_xy[0])
    new_spacing[inplane_axes[1]] = float(new_spacing_xy[1])
    new_spacing = tuple(new_spacing)

    # size that preserves FoV
    new_size = [0, 0, 0]
    for ax in range(3):
        if ax == axial_axis:
            new_size[ax] = int(original_size[ax])      # Z unchanged
        else:
            scale = original_spacing[ax] / new_spacing[ax]
            new_size[ax] = max(1, int(round(original_size[ax] * scale)))
    new_size = tuple(new_size)

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(0)
    # dtype: use float32 for intensities; keep label dtype
    out_pixel_type = image_sitk.GetPixelID() if is_label else sitk.sitkFloat32
    resample.SetOutputPixelType(out_pixel_type)

    return resample.Execute(image_sitk)

def sitk_to_numpy_slices(img_sitk):
    """
    SimpleITK image -> NumPy array of shape (D, H, W) float32.
    (SITK returns array in (z, y, x); axial slices are arr[z, :, :])
    """
    arr = sitk.GetArrayFromImage(img_sitk)  # (D,H,W)
    return np.asarray(arr, dtype=np.float32, order='C')

def _is_bad_aspect(H, W, max_ar: float = 3.0):
    """
    Flag slices that are extremely skinny or tiny.
    Returns (bad: bool, ar: float)
    """
    H = int(H); W = int(W)
    ar = (max(H, W) / max(1.0, min(H, W)))
    if ar > max_ar:
        return True, float(ar)
    return False, float(ar)

def _resize_to_256(sl, pad_to_square = False, pad_value = -1000.0):
    """
    Resize a 2D slice to 256x256.
    - mode="stretch": direct resize to (256,256) (changes aspect).
    - mode="letterbox": preserve aspect, pad with pad_value.
    Returns (resized: np.ndarray, info: dict)
    """
    assert sl.ndim == 2, "Expected (H, W) slice"
    H, W = sl.shape
    
    if pad_to_square:
        size = max(H, W)
        pad_h = (size - H) // 2
        pad_w = (size - W) // 2
        sl = np.pad(sl, ((pad_h, size - H - pad_h), (pad_w, size - W - pad_w)),
                    constant_values=pad_value)
        H, W = sl.shape  # update shape after padding

    target = 384
    # Fast path for already square 256
    if H == target and W == target:
        return sl.astype(np.float32, copy=False)

    img2d = sitk.GetImageFromArray(sl.astype(np.float32, copy=False))
    out = sitk.Resample(
        img2d,
        (target, target),
        sitk.Transform(),
        sitk.sitkLinear,
        img2d.GetOrigin(),
        (W / target, H / target),
        img2d.GetDirection(),
        pad_value,
        sitk.sitkFloat32,
    )
    arr = sitk.GetArrayFromImage(out)
    return np.asarray(arr, dtype=np.float32, order="C")

def _resize_to_384(sl, target_size=None, pad_value=-1000.0, interpolator=sitk.sitkLinear):
    """
    Center pad or crop to `target_size` (default: max(H, W)), then resample to 384x384.
    sl: 2D numpy array (H, W)
    pad_value: value to use when padding
    interpolator: e.g., sitk.sitkNearestNeighbor, sitk.sitkLinear (default), sitk.sitkBSpline
    """
    assert sl.ndim == 2, "Expected (H, W) slice"
    H, W = sl.shape
    T = target_size or max(H, W)

    # ---- center crop (if larger than T) ----
    h_start = max((H - T) // 2, 0)
    w_start = max((W - T) // 2, 0)
    sl = sl[h_start:h_start + min(H, T), w_start:w_start + min(W, T)]

    # ---- center pad (if smaller than T) ----
    pad_h = max(T - sl.shape[0], 0)
    pad_w = max(T - sl.shape[1], 0)
    if pad_h or pad_w:
        ph = (pad_h // 2, pad_h - pad_h // 2)
        pw = (pad_w // 2, pad_w - pad_w // 2)
        sl = np.pad(sl, (ph, pw), mode="constant", constant_values=pad_value)

    # fast path
    sl = sl.astype(np.float32, copy=False)
    if sl.shape == (384, 384):
        return sl

    # ---- resample to 384x384 ----
    img = sitk.GetImageFromArray(sl)  # spacing defaults to (1,1)
    out = sitk.Resample(
        img,
        (384, 384),
        sitk.Transform(),          # identity transform
        interpolator,              # *** use a real interpolator here ***
        img.GetOrigin(),
        (T / 384.0, T / 384.0),    # output spacing so that T -> 384
        img.GetDirection(),
        pad_value,
        sitk.sitkFloat32,
    )
    return np.asarray(sitk.GetArrayFromImage(out), dtype=np.float32, order="C")

def resample_sitk_image_xyz(image_sitk, new_spacing_xyz=(0.75, 0.75, 1.5), is_label=False):
    """
    Resample the volume to (dx, dy, dz) spacing.
    Reorients to LPS so axis 2 is axial. Uses Linear (labels: Nearest).
    """
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("LPS")
    image_sitk = orient_filter.Execute(image_sitk)

    original_spacing = image_sitk.GetSpacing()  # (sx, sy, sz)
    original_size    = image_sitk.GetSize()     # (nx, ny, nz)

    new_spacing = tuple(float(s) for s in new_spacing_xyz)
    new_size = []
    for ax in range(3):
        scale = original_spacing[ax] / new_spacing[ax]
        new_size.append(max(1, int(round(original_size[ax] * scale))))
    new_size = tuple(new_size)

    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image_sitk.GetDirection())
    resample.SetOutputOrigin(image_sitk.GetOrigin())
    resample.SetInterpolator(interpolator)
    resample.SetDefaultPixelValue(0)
    out_pixel_type = image_sitk.GetPixelID() if is_label else sitk.sitkFloat32
    resample.SetOutputPixelType(out_pixel_type)

    return resample.Execute(image_sitk)

def build_dataset_from_list(
    src_dir,
    dst_root,
    dataset_name="Dataset_from_list",
    label_map_json=None,
    new_spacing_xyz=(0.75, 0.75, 1.5),
    also_write_flat=False,
    map_size=int(100 * (1024**3)),  # ~100 GB default; adjust as needed
    resize_to_256=False,            # keep False if you want original in-plane size after resample
    sample_half=False,  
):
    """
    Same behavior as build_dataset_v2 but takes an explicit list of .nii.gz paths.
    """
    src_dir = Path(src_dir)
    nifti_paths = sorted(src_dir.glob("*.nii.gz"))
    if not nifti_paths:
        raise SystemExit(f"No numpy files in {src_dir}")

    out_root = Path(dst_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lmdb_path = out_root / f"{dataset_name}.lmdb"
    manifest_path = out_root / f"{dataset_name}_manifest.csv"
    index_txt = out_root / f"{dataset_name}_slices.index.txt"
    flat_dir = out_root / dataset_name
    if also_write_flat:
        flat_dir.mkdir(parents=True, exist_ok=True)

    print(f"LMDB map_size ~= {map_size/1e9:.2f} GB")
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=False, lock=True, readahead=False, max_dbs=1)

    counter = 0
    keys = []
    label_map_path = Path(label_map_json) if label_map_json else src_dir.parent / "dataset.json"
    with open(label_map_path, 'r') as file:
        label_maps = json.load(file)['labels']

    with env.begin(write=True) as txn, open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "key","logical_filename","source_volume","slice_idx","view",
            "orig_spacing_x","orig_spacing_y","orig_spacing_z",
            "new_spacing_x","new_spacing_y","new_spacing_z",
            "height_px","width_px"
        ])

        for p in tqdm(nifti_paths, desc="Processing selected volumes"):
            img = load_sitk_image(str(p))
            orig_sx, orig_sy, orig_sz = img.GetSpacing()
            img_res = resample_sitk_image_xyz(img, new_spacing_xyz=new_spacing_xyz, is_label=False)
            arr = sitk_to_numpy_slices(img_res)
            seg = load_sitk_image(str(p).replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz'))
            seg_res = resample_sitk_image_xyz(seg, new_spacing_xyz=new_spacing_xyz, is_label=True)
            seg_arr = sitk_to_numpy_slices(seg_res)
            D, H, W = arr.shape
            sx, sy, sz = img_res.GetSpacing()
            assert (D, H, W) == seg_arr.shape

            if sample_half:
                start = random.randint(0, 1)  # random parity per volume
                slice_indices = range(start, D, 2)
            else:
                slice_indices = range(D)

            for k in slice_indices:
                sl = arr[k, :, :]
                seg_sl = seg_arr[k, :, :].astype(np.int32, copy=False)
                label = classify(seg_sl, label_maps)

                if label == 'none':
                    continue

                if resize_to_256:
                    sl = _resize_to_384(sl, target_size = 228)

                buf = io.BytesIO()
                np.save(buf, sl.astype(np.float16 if resize_to_256 else np.float32, copy=False), allow_pickle=False)
                key = f"slice_{counter:06d}"
                txn.put(f"{key}.npy".encode(), buf.getvalue())

                meta = {
                    "source": str(p),
                    "axis": "axial",
                    "slice_idx": int(k),
                    "dtype": "float32" if not resize_to_256 else "float16",
                    "shape": [int(sl.shape[0]), int(sl.shape[1])],
                    "spacing_mm": [float(sx), float(sy), float(sz)],
                    "direction": None,
                    "origin": None,
                    "view": label
                }
                txn.put(f"{key}.json".encode(), json.dumps(meta).encode())
                keys.append(key)

                writer.writerow([
                    key, f"{key}.npy", str(p), k, label,
                    f"{orig_sx:.6f}", f"{orig_sy:.6f}", f"{orig_sz:.6f}",
                    f"{new_spacing_xyz[0]:.6f}", f"{new_spacing_xyz[1]:.6f}", f"{new_spacing_xyz[2]:.6f}",
                    sl.shape[0], sl.shape[1]
                ])

                if also_write_flat:
                    np.save(flat_dir / f"{key}.npy", sl, allow_pickle=False)

                counter += 1

        txn.put(b"__index__", "\n".join(keys).encode())

    with open(index_txt, "w") as f:
        f.write("\n".join(keys))

    print(f"Done. Wrote {counter} slices.")
    print(f"LMDB:     {lmdb_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Index:    {index_txt}")
    if also_write_flat:
        print(f"Flat .npy directory: {flat_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an LMDB slice dataset from TotalSegmentator-style NIfTI files.")
    parser.add_argument("--src-dir", required=True, help="Directory containing image .nii.gz files")
    parser.add_argument("--dst-root", required=True, help="Output directory for LMDB and metadata")
    parser.add_argument("--dataset-name", default="TotalSegmentatorv2", help="Output dataset prefix")
    parser.add_argument("--label-map-json", default=None, help="Path to dataset.json; defaults to <src-dir>/../dataset.json")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.5, 1.5, 1.5), metavar=("X", "Y", "Z"))
    parser.add_argument("--also-write-flat", action="store_true", help="Also write per-slice .npy files")
    parser.add_argument("--map-size-gb", type=int, default=1000, help="LMDB map size in GB")
    parser.add_argument("--resize-to-256", action="store_true", help="Resize slices before writing")
    parser.add_argument("--sample-half", action="store_true", help="Sample every other axial slice with random parity")
    args = parser.parse_args()

    build_dataset_from_list(
        src_dir=args.src_dir,
        dst_root=args.dst_root,
        dataset_name=args.dataset_name,
        label_map_json=args.label_map_json,
        new_spacing_xyz=tuple(args.spacing),
        also_write_flat=args.also_write_flat,
        map_size=int(args.map_size_gb * (1024**3)),
        resize_to_256=args.resize_to_256,
        sample_half=args.sample_half,
    )
