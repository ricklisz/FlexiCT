import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.ndimage import (
    _ni_support,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from monai.metrics import compute_hausdorff_distance, compute_surface_dice  
import json
from skimage.morphology import skeletonize, skeletonize_3d


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxels of binary objects in `result`
    and their nearest partner surface voxels in `reference`.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # emptiness checks
    if 0 == np.count_nonzero(result):
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if 0 == np.count_nonzero(reference):
        raise RuntimeError("The second supplied array does not contain any binary object.")

    # extract 1-voxel-wide borders
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # distance transform is computed in the inverted (background) domain
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds

def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    Average Symmetric Surface Distance (ASSD).
    """
    assd_vals = np.concatenate([
        __surface_distances(result, reference, voxelspacing, connectivity),
        __surface_distances(reference, result, voxelspacing, connectivity),
    ])
    return assd_vals.mean()

def resample(img, new_spacing=(1.5, 1.5, 1.5), interpolator=sitk.sitkNearestNeighbor):
    """
    Resample a SimpleITK image to isotropic spacing.

    Parameters
    ----------
    img : sitk.Image
        Input image.
    new_spacing : tuple of float
        Desired spacing (sx, sy, sz) in mm.
    interpolator : sitk interpolator
        sitk.sitkNearestNeighbor for labels, sitk.sitkLinear for images.

    Returns
    -------
    resampled_img : sitk.Image
        Resampled image with isotropic spacing.
    """
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputDirection(img.GetDirection())
    resample.SetInterpolator(interpolator)
    return resample.Execute(img)

def _surface_distance_arrays_mm(pred: np.ndarray, ref: np.ndarray, spacing_xyz) -> tuple[np.ndarray, np.ndarray]:
    """
    Return symmetric surface distances in mm:
      d_pred_to_ref (for pred surface points) and d_ref_to_pred (for ref surface points).
    spacing_xyz must be (x, y, z) as provided by SimpleITK.
    """
    # Empty handling up front
    if pred.sum() == 0 and ref.sum() == 0:
        return np.array([0.0]), np.array([0.0])
    if pred.sum() == 0 or ref.sum() == 0:
        return np.array([np.inf]), np.array([np.inf])

    # Convert to SITK with correct spacing
    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint8))
    ref_sitk  = sitk.GetImageFromArray(ref.astype(np.uint8))
    pred_sitk.SetSpacing(spacing_xyz)
    ref_sitk.SetSpacing(spacing_xyz)

    # Extract 3D surfaces (fully-connected)
    contour = sitk.LabelContourImageFilter()
    contour.SetFullyConnected(True)
    pred_surf = sitk.GetArrayFromImage(contour.Execute(pred_sitk)) > 0
    ref_surf  = sitk.GetArrayFromImage(contour.Execute(ref_sitk)) > 0

    # Fallback: tiny objects can sometimes produce empty contour; treat any foreground voxel as surface
    if not pred_surf.any() and pred.any():
        pred_surf = pred.astype(bool)
    if not ref_surf.any() and ref.any():
        ref_surf = ref.astype(bool)

    # Distance transforms in mm (to the *boundary*; signed but we take abs)
    dt_ref  = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(ref_sitk,  insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    dt_pred = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(pred_sitk, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))

    d_pred_to_ref = np.abs(dt_ref[pred_surf])
    d_ref_to_pred = np.abs(dt_pred[ref_surf])

    # If either side somehow ends empty, treat as infinite disagreement
    if d_pred_to_ref.size == 0 or d_ref_to_pred.size == 0:
        return np.array([np.inf]), np.array([np.inf])

    return d_pred_to_ref, d_ref_to_pred


def __centerline(mask):
    """
    Compute a 1-voxel-thick centerline (skeleton) of a binary vessel mask.

    Uses 2D skeletonize for 2D, skeletonize_3d for 3D.
    """
    mask = np.atleast_1d(mask.astype(np.bool_))

    if mask.ndim == 2:
        # 2D skeleton
        centerline = skeletonize(mask)
    elif mask.ndim == 3:
        # 3D skeleton
        centerline = skeletonize_3d(mask)
    else:
        raise ValueError("Centerline computation only implemented for 2D or 3D arrays.")

    return centerline.astype(np.bool_)


def __centerline_distances(result, reference, voxelspacing=None):
    """
    Distances from the centerline voxels of `result` to the closest
    centerline voxels of `reference`.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # emptiness checks on original masks
    if 0 == np.count_nonzero(result):
        raise RuntimeError("The first supplied array does not contain any binary object.")
    if 0 == np.count_nonzero(reference):
        raise RuntimeError("The second supplied array does not contain any binary object.")

    # extract centerlines
    result_cl = __centerline(result)
    reference_cl = __centerline(reference)

    if 0 == np.count_nonzero(result_cl):
        raise RuntimeError("Centerline of the first array is empty.")
    if 0 == np.count_nonzero(reference_cl):
        raise RuntimeError("Centerline of the second array is empty.")

    # distance transform in the inverted (background) domain of reference centerline
    dt = distance_transform_edt(~reference_cl, sampling=voxelspacing)
    cds = dt[result_cl]
    return cds


def centerline_distance(result, reference, voxelspacing=None):
    cd_vals = np.concatenate([
        __centerline_distances(result, reference, voxelspacing),
        __centerline_distances(reference, result, voxelspacing),
    ])
    return cd_vals.mean()


def compute_dice(pred, ref):
    """Compute the Dice coefficient for two binary numpy arrays."""
    intersection = np.logical_and(pred, ref).sum()
    denom = pred.sum() + ref.sum()
    if denom == 0:
        # Both empty => define Dice=1.0, or handle as you prefer
        return 1.0
    return 2.0 * intersection / denom

def compute_volume_difference(pred, ref, spacing):
    """
    Compute the volume difference (in mL, for instance) between two binary arrays,
    given spacing = (sx, sy, sz).
    """
    voxel_volume = np.prod(spacing)  # mm^3 per voxel
    vol_pred = pred.sum() * voxel_volume
    vol_ref  = ref.sum()  * voxel_volume
    # convert mm^3 to mL (1 mL = 1000 mm^3)
    return (vol_pred - vol_ref) / 1000.0

def compute_centroid_distance(pred, ref, spacing):
    """
    Compute the Euclidean distance between centroids (in physical space).
    """
    coords_pred = np.argwhere(pred)
    coords_ref  = np.argwhere(ref)
    
    if coords_pred.size == 0 and coords_ref.size == 0:
        # Both empty => distance=0 or handle as you prefer
        return np.nan
    if coords_pred.size == 0 or coords_ref.size == 0:
        # One is empty and the other not => define distance=NaN (or large sentinel)
        return np.nan
    
    centroid_pred = coords_pred.mean(axis=0)  # (z, y, x)
    centroid_ref  = coords_ref.mean(axis=0)   # (z, y, x)
    
    # Convert index space to physical space
    # Make sure to align the order of spacing with the order of the coordinates
    diff = (centroid_pred - centroid_ref) * np.array(spacing[::-1])
    return np.sqrt(np.sum(diff**2))

def compute_hd95(pred: np.ndarray, ref: np.ndarray, spacing_xyz) -> float:
    d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
    all_d = np.concatenate([d1, d2])
    # If both arrays are inf (one empty, one non-empty), define as inf (or large sentinel)
    if np.isinf(all_d).all():
        return np.nan
    return float(np.percentile(all_d, 95))

def compute_average_surface_distance(pred: np.ndarray, ref: np.ndarray, spacing_xyz) -> float:
    d1, d2 = _surface_distance_arrays_mm(pred, ref, spacing_xyz)
    all_d = np.concatenate([d1, d2])
    if np.isinf(all_d).all():
        return np.nan
    return float(0.5 * (d1.mean() + d2.mean()))

def compute_nsd(pred: np.ndarray, ref: np.ndarray, spacing_xyz, tau_mm: float) -> float:
    """
    Normalized Surface Distance (aka Surface Dice) at tolerance tau_mm.
    Returns a value in [0, 1].

    NSD = ( |S_pred within τ of Ref| + |S_ref within τ of Pred| ) / ( |S_pred| + |S_ref| )
    """
    # Handle empties explicitly (common convention; be explicit in your paper/report)
    if pred.sum() == 0 and ref.sum() == 0:
        return 1.0  # both empty => perfect agreement by convention
    if pred.sum() == 0 or ref.sum() == 0:
        return 0.0  # one empty => total disagreement

    d_pred_to_ref, d_ref_to_pred = _surface_distance_arrays_mm(pred, ref, spacing_xyz)

    # If something went wrong and we only have infs, treat as 0 overlap
    if np.isinf(d_pred_to_ref).all() and np.isinf(d_ref_to_pred).all():
        return 0.0

    within_pred = np.sum(d_pred_to_ref <= tau_mm)
    within_ref  = np.sum(d_ref_to_pred <= tau_mm)
    total_pred  = d_pred_to_ref.size
    total_ref   = d_ref_to_pred.size

    return float((within_pred + within_ref) / (total_pred + total_ref))

def _compute_metrics_for_file(fname, preds_dir, labels_dir, class_list, use_gpu=False):
    """Runs on a worker process; returns {'filename': ..., 'dice_class1': ..., ...}."""
    # Device choice per worker
    if use_gpu:
        # optional: bind each worker to the same GPU; you can also shard GPUs by env CUDA_VISIBLE_DEVICES
        device = torch.device("cuda")
        torch.set_num_threads(1)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(1)

    pred_path = os.path.join(preds_dir, fname)
    label_path = os.path.join(labels_dir, fname)

    # Load volumes
    pred_img = sitk.ReadImage(pred_path)
    label_img = sitk.ReadImage(label_path)
    pred_array = sitk.GetArrayFromImage(pred_img)   # [z, y, x]
    label_array = sitk.GetArrayFromImage(label_img) # [z, y, x]
    spacing = label_img.GetSpacing()                # (sx, sy, sz)

    if pred_array.shape != label_array.shape:
        raise ValueError(f"Shape mismatch between prediction and label for {fname}.")

    metrics_per_scan = {"filename": fname}
    num_classes = len(class_list)

    for c in range(num_classes):
        label_c = (label_array == class_list[c]['total_id'])
        pred_c = np.isin(pred_array, class_list[c]['cardiac_id'])
        if label_c.sum() == 0:
            dice_c = np.nan
            hd95_c = np.nan
            nsd_c  = np.nan
            assd_c = np.nan
        else:
            # Dice (numpy)
            dice_c = compute_dice(pred_c, label_c)
            # MONAI HD95 / NSD (torch) — run on CPU by default for safe parallelism
            pred_c_oh  = torch.as_tensor(pred_c, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(device)
            label_c_oh = torch.as_tensor(label_c, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(device)

            hd95_c = compute_hausdorff_distance(
                y_pred=pred_c_oh, y=label_c_oh,
                include_background=False, percentile=95, spacing=(spacing[2], spacing[1], spacing[0])
            ).cpu().numpy()[0][0]

            nsd_c = compute_surface_dice(
                y_pred=pred_c_oh, y=label_c_oh,
                class_thresholds=[1.0],
                include_background=False, spacing=(spacing[2], spacing[1], spacing[0])
            ).cpu().numpy()[0][0]

            try:
                assd_c = assd(
                    result=pred_c, reference=label_c,
                    voxelspacing=(spacing[2], spacing[1], spacing[0]),
                    connectivity=1
                )
            except RuntimeError:
                assd_c = np.nan

        metrics_per_scan[f"dice_class{c}"] = dice_c
        metrics_per_scan[f"hd95_class{c}"] = hd95_c
        metrics_per_scan[f"nsd_class{c}"]  = nsd_c
        metrics_per_scan[f"assd_class{c}"] = assd_c

    return metrics_per_scan


def compute_segmentation_metrics(
    preds_dir,
    labels_dir,
    class_list,
    file_suffix=".nii.gz",
    max_workers=None,
    use_gpu=False
):
    """
    Parallelized version of your compute_segmentation_metrics (per-file parallelism).
    By default runs MONAI metrics on CPU in workers (use_gpu=False) for safety.
    """
    # Discover overlap
    pred_files  = sorted([f for f in os.listdir(preds_dir) if f.endswith(file_suffix)])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(file_suffix)])
    common_files = sorted(list(set(pred_files).intersection(set(label_files))))

    if max_workers is None:
        # Good default that avoids oversubscription; tweak as needed
        max_workers = max(1, os.cpu_count() // 2)

    worker = partial(_compute_metrics_for_file,
                     preds_dir=preds_dir,
                     labels_dir=labels_dir,
                     class_list=class_list,
                     use_gpu=use_gpu)

    rows = []
    # Tip: If you like a progress bar, wrap this in tqdm(as_completed(...), total=len(common_files))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, fname): fname for fname in common_files}
        for fut in as_completed(futures):
            rows.append(fut.result())

    df_metrics = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)

    # Aggregate — identical to your original
    metric_cols = [c for c in df_metrics.columns if c != "filename"]
    means = df_metrics[metric_cols].mean(numeric_only=True, skipna=True)
    stds  = df_metrics[metric_cols].std(numeric_only=True, skipna=True)

    summary = {}
    num_classes = len(class_list)
    for c in range(num_classes):
        class_summary = {}
        for metric_key in ["dice", "hd95", "nsd", "assd"]:
            col_name = f"{metric_key}_class{c}"
            class_summary[metric_key] = {"mean": means[col_name], "std": stds[col_name]}
        summary[f"class_{class_list[c]['name']}"] = class_summary

    return df_metrics, summary

def inference(dataset, trainer):
    preds_dir = f"/path/to/project-data/segmentation_data/{dataset}/imagesTs_{trainer}"
    csv_filename = preds_dir + f"/{trainer}_segmentation_metrics.csv"

    labels_dir = f"/path/to/project-data/segmentation_data/{dataset}/labelsTs"
    # Load dataset.json which contains the labels mapping (including background)
    class_list = [
        {"name": "aorta", 'total_id': 7, 'cardiac_id': [1,2,5]},
        {"name": "inferior_vena_cava", 'total_id': 8, 'cardiac_id': [6]},
        {"name": "pulmonary_artery", 'total_id': 49, 'cardiac_id': [13]},
        {"name": "heart_atrium_left", 'total_id': 45, 'cardiac_id': [7]},
        {"name": "heart_ventricle_left", 'total_id': 46, 'cardiac_id': [11]},
        {"name": "heart_atrium_right", 'total_id': 47, 'cardiac_id': [16]},
        {"name": "heart_ventricle_right","total_id": 48, "cardiac_id": [18]}, 
    ]
    print(class_list)
    df, summary_dict = compute_segmentation_metrics(preds_dir, labels_dir, class_list)

    aggregated = {metric: {"mean": [], "std": []} for metric in ["dice", "hd95", "nsd", "assd"]}
    for class_name, metrics in summary_dict.items():
        for metric_key in aggregated.keys():
            mean_val = metrics[metric_key]["mean"]
            std_val = metrics[metric_key]["std"]
            # Append only if the values are not nan
            if not np.isnan(mean_val):
                aggregated[metric_key]["mean"].append(mean_val)
            if not np.isnan(std_val):
                aggregated[metric_key]["std"].append(std_val)
    
    classwise_average = {}
    for metric_key, values in aggregated.items():
        avg_mean = np.mean(values["mean"]) if values["mean"] else np.nan
        avg_std = np.mean(values["std"]) if values["std"] else np.nan
        classwise_average[metric_key] = {"mean": avg_mean, "std": avg_std}

    # Add the aggregated class-wise averages to the renamed summary dict
    summary_dict["classwise_average"] = classwise_average

    # Print the summarized metrics
    print("\nSummary (Mean & Std) per class (NaN entries are ignored in the calculation):")
    for class_label, metrics in summary_dict.items():
        print(f"  {class_label}:")
        for metric_key, stats in metrics.items():
            print(f"    {metric_key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    df.to_csv(csv_filename, index=False)
    print(f"\nDataFrame saved to {csv_filename}")

if __name__ == "__main__":
    dataset = "Dataset501_Total"
    trainer = "prev_meddinov3_base_Simple_Mask2Former_Trainer__nnUNetPlans__2d_896"
    inference(dataset, trainer)
