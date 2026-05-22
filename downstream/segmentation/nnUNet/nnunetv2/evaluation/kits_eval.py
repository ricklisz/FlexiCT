import os.path
from multiprocessing import Pool
from typing import Tuple, Union, List

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
from pathlib import Path
import os
from time import time

TRAINING_DIR = Path(__file__).parent.parent.parent / "dataset"
TESTING_DIR = (
    Path(os.environ["KITS23_TEST_DIR"]).resolve(strict=True)
    if "KITS23_TEST_DIR" in os.environ.keys() else None
)
LEGACY_SRC_DIR = (
    Path(os.environ["KITS21_SERVER_DATA"]).resolve(strict=True)
    if "KITS21_SERVER_DATA" in os.environ.keys() else None
)
SRC_DIR = (
    Path(os.environ["KITS23_SERVER_DATA"]).resolve(strict=True)
    if "KITS23_SERVER_DATA" in os.environ.keys() else None
)
KITS21_PATH = (
    Path(os.environ["KITS21_PATH"]).resolve(strict=True)
    if "KITS21_PATH" in os.environ.keys() else None
)
CACHE_FILE = Path(__file__).parent.parent / "annotation" / "cache.json"


def construct_HEC_from_segmentation(segmentation: np.ndarray, label: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Takes a segmentation as input (integer map with values indicating what class a voxel belongs to) and returns a
    boolean array based on where the selected label/HEC is. If label is a tuple, all pixels belonging to any of the
    listed classes will be set to True in the results. The rest remains False.
    """
    if not isinstance(label, (tuple, list)):
        return segmentation == label
    else:
        if len(label) == 1:
            return segmentation == label[0]
        else:
            mask = np.zeros(segmentation.shape, dtype=bool)
            for l in label:
                mask[segmentation == l] = True
            return mask
        
KITS_HEC_LABEL_MAPPING = {
    'kidney_and_mass': (1, 2, 3),
    'mass': (2, 3),
    'tumor': (2, ),
}

KITS_LABEL_TO_HEC_MAPPING = {j: i for i, j in KITS_HEC_LABEL_MAPPING.items()}

HEC_NAME_LIST = list(KITS_HEC_LABEL_MAPPING.keys())

# just for you as a reference. This tells you which metric is at what index.
# This is not used anywhere
METRIC_NAME_LIST = ["Dice", "SD"]

LABEL_AGGREGATION_ORDER = (1, 3, 2)
# this means that we first place the kidney, then the cyst and finally the
# tumor. The order matters! If parts of a later label (example tumor) overlap
# with a prior label (kidney or cyst) the prior label is overwritten

KITS_LABEL_NAMES = {
    1: "kidney",
    2: "tumor",
    3: "cyst"
}

# values are determined by kits21/evaluation/compute_tolerances.py
HEC_SD_TOLERANCES_MM = {
    'kidney_and_mass': 1.0330772532390826,
    'mass': 1.1328796488598762,
    'tumor': 1.1498198361434828,
}

# this determines which reference file we use for evaluation
GT_SEGM_FNAME = 'segmentation.nii.gz'

# how many groups of sampled segmentations?
NUMBER_OF_GROUPS = 5

def dice(prediction: np.ndarray, reference: np.ndarray):
    """
    Both predicion and reference have to be bool (!) numpy arrays. True is interpreted as foreground, False is background
    """
    intersection = np.count_nonzero(prediction & reference)
    numel_pred = np.count_nonzero(prediction)
    numel_ref = np.count_nonzero(reference)
    if numel_ref == 0 and numel_pred == 0:
        return np.nan
    else:
        return 2 * intersection / (numel_ref + numel_pred)
    
def compute_metrics_for_label(segmentation_predicted: np.ndarray, segmentation_reference: np.ndarray,
                              label: Union[int, Tuple[int, ...]], spacing: Tuple[float, float, float],
                              sd_tolerance_mm: float = None) \
        -> Tuple[float, float]:
    """
    :param segmentation_predicted: segmentation map (np.ndarray) with int values representing the predicted segmentation
    :param segmentation_reference:  segmentation map (np.ndarray) with int values representing the gt segmentation
    :param label: can be int or tuple of ints. If tuple of ints, a HEC is constructed from the labels in the tuple.
    :param spacing: important to know for volume and surface distance computation
    :param sd_tolerance_mm
    :return:
    """
    assert all([i == j] for i, j in zip(segmentation_predicted.shape, segmentation_reference.shape)), \
        "predicted and gt segmentation must have the same shape"

    # make label always a tuple. Needed for inferring sd_tolerance_mm if not given
    label = (label,) if not isinstance(label, (tuple, list)) else label

    # build a bool mask from the segmentation_predicted, segmentation_reference and provided label(s)
    mask_pred = construct_HEC_from_segmentation(segmentation_predicted, label)
    mask_gt = construct_HEC_from_segmentation(segmentation_reference, label)
    gt_empty = np.count_nonzero(mask_gt) == 0
    pred_empty = np.count_nonzero(mask_pred) == 0

    if sd_tolerance_mm is None:
        sd_tolerance_mm = HEC_SD_TOLERANCES_MM[KITS_LABEL_TO_HEC_MAPPING[label]]

    if gt_empty and pred_empty:
        sd = 1
        dc = 1
    elif gt_empty or pred_empty:
        sd = 0
        dc = 0
    else:
        dc = dice(mask_pred, mask_gt)
        dist = compute_surface_distances(mask_gt, mask_pred, spacing)
        sd = compute_surface_dice_at_tolerance(dist, tolerance_mm=sd_tolerance_mm)

    return dc, sd


def compute_metrics_for_case(fname_pred: str, fname_ref: str) -> np.ndarray:
    """
    Takes two .nii.gz segmentation maps and computes the KiTS metrics for all HECs. The return value of this function
    is an array of size num_HECs x num_metrics.
    The order of metrics in the tuple follows the order on the KiTS website (https://kits23.kits-challenge.org/):
    -> Dice (1 is best)
    -> Surface Dice (1 is best)
    :param fname_pred: filename of the predicted segmentation
    :param fname_ref: filename of the ground truth segmentation
    :return: np.ndarray of shape 3x2 (labels x metrics). Labels are HECs in the order given by HEC_NAME_LIST
    """
    img_pred = sitk.ReadImage(fname_pred)
    img_ref = sitk.ReadImage(fname_ref)

    # we need to invert the spacing because SimpleITK is weird
    spacing_pred = list(img_pred.GetSpacing())[::-1]
    spacing_ref = list(img_ref.GetSpacing())[::-1]

    if not all([i == j] for i, j in zip(spacing_pred, spacing_ref)):
        # no need to make this an error. We can evaluate successfullt as long as the shapes match.
        print("WARNING: predicted and reference segmentation do not have the same spacing!")

    img_pred_npy = sitk.GetArrayFromImage(img_pred)
    img_gt_npy = sitk.GetArrayFromImage(img_ref)

    metrics = np.zeros((len(HEC_NAME_LIST), 2), dtype=float)
    for i, hec in enumerate(HEC_NAME_LIST):
        metrics[i] = compute_metrics_for_label(img_pred_npy, img_gt_npy, KITS_HEC_LABEL_MAPPING[hec],
                                               tuple(spacing_pred), sd_tolerance_mm=HEC_SD_TOLERANCES_MM[hec])
    return metrics


def evaluate_predictions(
    folder_with_predictions: str,
    gt_dir: str,
    num_processes: int = 8,
    write_csv_file: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    folder_with_predictions: contains case_XXXXX.nii.gz (subset ok)
    gt_dir: contains GT case_XXXXX.nii.gz (full dataset ok)
    """

    start = time()
    p = Pool(num_processes)

    pred_files = subfiles(folder_with_predictions, suffix=".nii.gz", join=True)
    gt_files = subfiles(gt_dir, suffix=".nii.gz", join=True)

    # map: case_id -> filepath
    pred_map = {os.path.basename(f)[:-7]: f for f in pred_files}  # strips ".nii.gz"
    gt_map = {os.path.basename(f)[:-7]: f for f in gt_files}

    common_caseids = sorted(set(pred_map.keys()) & set(gt_map.keys()))
    missing_gt = sorted(set(pred_map.keys()) - set(gt_map.keys()))
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} predictions have no GT match (will be skipped). "
              f"Example: {missing_gt[:5]}")

    if len(common_caseids) == 0:
        raise RuntimeError("No overlapping case IDs found between predictions and gt_dir.")

    params = [(pred_map[c], gt_map[c]) for c in common_caseids]

    metrics_list = p.starmap(compute_metrics_for_case, params)
    metrics = np.vstack([m[None] for m in metrics_list])

    p.close()
    p.join()

    end = time()
    print(f"Evaluated {len(common_caseids)} cases. Took {np.round(end - start, 2)} s. "
          f"Num_processes: {num_processes}")

    if write_csv_file:
        out_csv = join(folder_with_predictions, "evaluation.csv")
        with open(out_csv, "w") as f:
            f.write("caseID,"
                    "Dice_kidney_and_mass,Dice_mass,Dice_tumor,"
                    "SD_kidney_and_mass,SD_mass,SD_tumor\n")
            for i, c in enumerate(common_caseids):
                f.write("%s,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f\n" % (
                    c,
                    metrics[i, 0, 0], metrics[i, 1, 0], metrics[i, 2, 0],
                    metrics[i, 0, 1], metrics[i, 1, 1], metrics[i, 2, 1],
                ))

            mean_metrics = metrics.mean(0)
            f.write("average,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f,%0.8f\n" % (
                mean_metrics[0, 0], mean_metrics[1, 0], mean_metrics[2, 0],
                mean_metrics[0, 1], mean_metrics[1, 1], mean_metrics[2, 1],
            ))

        print(f"Wrote: {out_csv}")

    # return also the list of evaluated prediction files (aligned with metrics rows)
    evaluated_pred_files = [pred_map[c] for c in common_caseids]
    return metrics, evaluated_pred_files



def sort_by_worst_Dice(evaluation_csv_file: str, n_worst: int = 20):
    loaded = np.loadtxt(evaluation_csv_file, dtype=str, skiprows=1, delimiter=',')
    casenames = loaded[:, 0]
    metrics = loaded[:, 1:].astype(float)
    dice_scores = metrics[:, :3]
    for i, hec in enumerate(HEC_NAME_LIST):
        print(hec)
        argsorted = np.argsort(dice_scores[:, i])
        for a in argsorted[:n_worst]:
            print(casenames[a], dice_scores[a, i])
        print()
        
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Runs KiTS evaluation on a folder of predictions (subset allowed) "
                    "against a flat GT folder of case_XXXXX.nii.gz."
    )
    parser.add_argument("folder_with_predictions", type=str,
                        help="Folder containing predicted segmentations named case_XXXXX.nii.gz")
    parser.add_argument("--gt_dir", required=True, type=str,
                        help="Folder containing GT segmentations named case_XXXXX.nii.gz")
    parser.add_argument("-num_processes", required=False, default=12, type=int,
                        help="Number of CPU cores to use. Default: 12")

    args = parser.parse_args()
    evaluate_predictions(args.folder_with_predictions, args.gt_dir, args.num_processes)


if __name__ == '__main__':
    main()  
    
# python /path/to/project-data/home/nnUNet/nnunetv2/evaluation/kits_eval.py /path/to/project-data/results/nnUNet_results/Dataset140_KiTS23/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation --gt_dir /path/to/project-data/nnUNet_preprocessed/Dataset140_KiTS23/gt_segmentations -num_processes 12
