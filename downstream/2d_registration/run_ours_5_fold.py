import csv
import argparse
import json
import os
import sys
import time
from pathlib import Path

import einops
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist
from skimage.transform import resize
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flexi_ct import Flexi_CT_2D  # noqa: E402
from flexi_ct.checkpoints import resolve_flexict_checkpoint  # noqa: E402

# local imports
import utils.img_operations as img_op
from utils.img_operations import (
    extract_lung_mask,
    MR_normalize,
    pca_lowrank_transform,
    remove_uniform_intensity_slices,
    to_lungCT_window,
)
from utils.convexAdam_3D import convex_adam_3d_param
"""
FILE NOTE: 5-fold cross-validation registration runner for Flexi_CT_2D.
"""


DEFAULT_OUTPUT_DIR = ROOT / "results" / "2d_registration" / "ours_5_fold"
DEFAULT_SPLIT_CSVS = ("pairs_Val.csv", "pairs_Tr.csv", "pairs_Ts.csv", "pairs_Test.csv")


def _set_patch_size(model: torch.nn.Module, patch_size: int) -> None:
    for module in model.modules():
        if hasattr(module, "set_patch_size"):
            module.set_patch_size(patch_size)
    if hasattr(model, "patch_size"):
        model.patch_size = patch_size


def _parse_feature_size(value: str) -> tuple[int, int]:
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("feature size must be formatted as H,W or HxW")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("feature size values must be integers") from exc


class dinoReg:

    def __init__(self, device_id=0, batch_size = 32, lr=1, smooth_weight=10, num_iter=1000, feat_size=(80,80), patch_size=8, ckpt_path='', configs=None):
        self.configs = configs if configs is not None else {}
        self.device_id = device_id
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.patch_size = patch_size
        self.ckpt_path = ckpt_path
        self.gap = self.configs.get('gap', 3)
        self.model = self.load_model()
        self.embed_dim = self.model.backbone.embed_dim * 4
        self.img_size = (self.patch_size * feat_size[0], self.patch_size * feat_size[1])
        self.num_iter = num_iter

        self.batch_size = batch_size
        self.reg_featureDim = 24
        self.lr = lr
        self.smooth_weight = smooth_weight
        print('learning rate', self.lr)

        self.feature_height = self.img_size[0] // self.patch_size
        self.feature_width = self.img_size[1] // self.patch_size

        # Store configs and initialize eigenvalue_array
        self.eigenvalue_array = []
        self.pca_matrix = None  # Can be set externally if useSavedPCA is True
        self.pca_save_path = None  # Path to save PCA matrix if computed

    def zero_to_one(self, array):
        mean = float(np.mean(array))
        std = float(np.std(array))
        if std < 1e-6:
            return array - mean
        return (array - mean) / std

    def ct_normalize(self, img):
        return np.clip(img, a_min=-1000, a_max=1000)

    def _apply_intensity_preprocess(self, arr, mode):
        """
        Apply intensity preprocessing based on mode.

        Args:
            arr: 3D numpy array (H, W, D)
            mode: str, one of 'lung', 'abdomen', 'soft_tissue', 'mr'
                - 'lung': CT lung window (wl=-600, ww=1500)
                - 'abdomen': CT abdominal window (wl=40, ww=400)
                - 'soft_tissue': CT soft tissue window (wl=50, ww=400)
                - 'mr': MR percentile normalization

        Returns:
            Preprocessed array normalized to [0, 1]
        """
        if mode == 'lung':
            return to_lungCT_window(arr, wl=-600, ww=1500)
        elif mode == 'abdomen':
            return to_lungCT_window(arr, wl=40, ww=400)
        elif mode == 'soft_tissue':
            return to_lungCT_window(arr, wl=50, ww=400)
        elif mode == 'mr':
            return MR_normalize(arr)
        elif mode == 'meddinov3_ct':
            return self.zero_to_one(self.ct_normalize(arr))
        elif mode == 'meddinov3_mr':
            return self.zero_to_one(arr)
        else:
            raise ValueError(f"Unknown preprocessing mode: {mode}. Use 'lung', 'abdomen', 'soft_tissue', or 'mr'.")

    def extract_dinov2_feature(self, input_array):

        assert len(input_array.shape) == 3  # 2D image

        """flipping the input if needed"""
        # input_array = np.swapaxes(input_array, 0,1)

        input_rgb_array = input_array[np.newaxis, :, :, :]

        input_tensor = torch.as_tensor(np.transpose(input_rgb_array, [0, 3, 1, 2]), dtype=torch.float32)
        feature_array = self._extract_patch_tokens(input_tensor)
        del input_tensor

        return feature_array

    def extract_dinov2_feature_batch(self, input_batch):
        """
        Extract DINO features for a batch of 2D slices in a single forward pass.

        Args:
            input_batch: numpy array of shape [B, H, W, 1] where B is batch size

        Returns:
            feature_array: numpy array of shape [B, N, C] where N is number of patches, C is embed_dim
        """
        assert len(input_batch.shape) == 4  # B x H x W x 1

        input_tensor = torch.as_tensor(np.transpose(input_batch, [0, 3, 1, 2]), dtype=torch.float32)
        feature_array = self._extract_patch_tokens(input_tensor)

        del input_tensor
        return feature_array

    def _extract_patch_tokens(self, input_tensor):
        """Return concatenated patch tokens from the last four Flexi_CT_2D layers."""
        input_tensor = input_tensor.to(device=self.device)

        with torch.inference_mode():
            layer_outputs = self.model.backbone.get_intermediate_layers(
                input_tensor,
                n=4,  # last 4 layers
                norm=True
            )
            feature_array = torch.cat(layer_outputs, dim=-1).detach().cpu().numpy()
        return feature_array


    def case_inference(self, mov_arr, fix_arr, orig_img_shape, aff_mov,
                       mask_fixed=None, mask_moving=None, case_id='noID', disp_init=None, grid_sp_adam=1,
                       DINOReg_useMask=True, save_feature=False, output_dir=None):

        assert len(mov_arr.shape) == 3

        """prepcocessing and feature extraction"""
        mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape, mask_fixed_arr, mask_moving_arr = self.case_preprocess(mov_arr, fix_arr)


        print('preprocessed moving and fixed image, shape', mov_arr.shape, fix_arr.shape)
        gap = self.gap #3

        mov_feature = self.encode_3D_gap(mov_arr, gap=gap)
        print('encoded moving image')
        fix_feature = self.encode_3D_gap(fix_arr, gap=gap)
        print('encoded fixed image')

        feat_sliceNum = self.slice_num


        """PCA reduce dimension"""
        #only features inside the mask
        if DINOReg_useMask:
            # reshape to model output

            mask_fixed_arr = resize(mask_fixed_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                anti_aliasing=True)
            mask_moving_arr = resize(mask_moving_arr, (self.feature_height, self.feature_width, feat_sliceNum),
                                 anti_aliasing=True)
            mask_fixed_arr = np.where(mask_fixed_arr > 0.99, 1.0, 0)
            mask_moving_arr = np.where(mask_moving_arr > 0.99, 1.0, 0)

            mask_moving_arr = mask_moving_arr.flatten().astype(bool)
            mask_fixed_arr = mask_fixed_arr.flatten().astype(bool)
            mov_feature = mov_feature[mask_moving_arr, :]
            fix_feature = fix_feature[mask_fixed_arr, :]

        print('Starting PCA to reduce dimension')
        all_features = np.concatenate([mov_feature,fix_feature], axis=0)
        print('all features shape', all_features.shape, 'mask sum', mask_moving_arr.sum(), mask_fixed_arr.sum())
        pca_start_time = time.time()
        if self.configs.get('useSavedPCA', False) and self.pca_matrix is not None:
            # Use existing PCA matrix
            reduced_patches = np.dot(all_features, self.pca_matrix)
            eigenvalues = np.zeros(24)
        else:
            # Compute PCA and optionally save the matrix
            reduced_patches, eigenvalues, V = pca_lowrank_transform(all_features, self.reg_featureDim, mat=True)
            reduced_patches = reduced_patches.numpy() if hasattr(reduced_patches, 'numpy') else reduced_patches
            eigenvalues = eigenvalues.numpy() if hasattr(eigenvalues, 'numpy') else eigenvalues
            V = V.numpy() if hasattr(V, 'numpy') else V
            # Save PCA matrix if useSavedPCA is True (file didn't exist)
            if self.configs.get('useSavedPCA', False) and self.pca_save_path is not None:
                np.save(self.pca_save_path, V[:, :self.reg_featureDim])
                self.pca_matrix = V[:, :self.reg_featureDim]
                print(f'Saved PCA matrix to {self.pca_save_path}')
        print('PCA finished in {}, splitting features'.format(time.time()-pca_start_time))

        if DINOReg_useMask:
            mov_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
            fix_pca = np.zeros((self.feature_height * self.feature_width * feat_sliceNum, self.reg_featureDim), dtype='float32')
            mov_pca[mask_moving_arr, :] = reduced_patches[:mask_moving_arr.sum(), :]
            fix_pca[mask_fixed_arr, :] = reduced_patches[mask_moving_arr.sum():, :]
            mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
        else:

            mov_pca = reduced_patches[:feat_sliceNum * self.feature_height * self.feature_width, :]
            fix_pca = reduced_patches[feat_sliceNum * self.feature_height * self.feature_width:, :]
            mov_pca = mov_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])
            fix_pca = fix_pca.reshape([self.feature_height, self.feature_width, feat_sliceNum, -1])

        self.eigenvalue_array.append(eigenvalues[:24])




        print('reshaping to original image shape')
        mov_pca_rescaled = resize(mov_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                   anti_aliasing=True)
        fix_pca_rescaled = resize(fix_pca, (orig_chunked_shape[0], orig_chunked_shape[1], orig_chunked_shape[2], self.reg_featureDim),
                                   anti_aliasing=True)


        #plug in the slices to keep, the rest are 0
        mov_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')
        fix_fullImg_pca_rescaled = np.zeros((orig_img_shape[0], orig_img_shape[1], orig_img_shape[2], self.reg_featureDim),
                                          dtype='float32')

        mov_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = mov_pca_rescaled
        fix_fullImg_pca_rescaled[:, :, slices_to_keep_indices, :] = fix_pca_rescaled

        if save_feature and output_dir is not None:
            os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
            np.save(os.path.join(output_dir, 'features', case_id + '_mov_feat.npy'), mov_fullImg_pca_rescaled)
            np.save(os.path.join(output_dir, 'features', case_id + '_fix_feat.npy'), fix_fullImg_pca_rescaled)

        """ConvexAdam optimization"""
        print('starting ConvexAdam optimization')

        disp = convex_adam_3d_param(fix_fullImg_pca_rescaled, mov_fullImg_pca_rescaled, loss_func = "SSD", grid_sp_adam=grid_sp_adam,
                                               lambda_weight=self.configs['smooth_weight'], selected_niter=self.configs['num_iter'], lr=self.configs['lr'], disp_init=disp_init,
                                                iter_smooth_kernel = self.configs['iter_smooth_kernel'],
                                                iter_smooth_num = self.configs['iter_smooth_num'], end_smooth_kernel=1,final_upsample=self.configs['final_upsample'])


        """apply displacement field to moving image or landmarks"""

        return disp

    def case_preprocess(self, mov_arr, fix_arr):
        assert len(mov_arr.shape) == 3
        assert len(fix_arr.shape) == 3

        pad_indices = []
        filtered_image_data, slices_to_keep_indices = remove_uniform_intensity_slices(fix_arr)
        pad_indices.append(slices_to_keep_indices)
        fix_arr = filtered_image_data
        mov_arr = mov_arr[:, :, slices_to_keep_indices]

        orig_chunked_shape = fix_arr.shape

        # Apply intensity preprocessing based on config
        # Options: 'lung', 'abdomen', 'soft_tissue', 'mr'
        fix_preprocess_mode = self.configs.get('fix_preprocess', 'soft_tissue')
        mov_preprocess_mode = self.configs.get('mov_preprocess', 'soft_tissue')

        fix_arr = self._apply_intensity_preprocess(fix_arr, fix_preprocess_mode)
        mov_arr = self._apply_intensity_preprocess(mov_arr, mov_preprocess_mode)

        filtered_z = fix_arr.shape[2]

        mask_fixed = np.zeros_like(fix_arr)
        mask_moving = np.zeros_like(mov_arr)
        for slice_idx in range(fix_arr.shape[2]):
            mask_fixed[:, :, slice_idx] = extract_lung_mask(fix_arr[:, :, slice_idx], threshold_value=0.005)
            mask_moving[:, :, slice_idx] = extract_lung_mask(mov_arr[:, :, slice_idx], threshold_value=0.005)


        return mov_arr, fix_arr, slices_to_keep_indices, orig_chunked_shape , mask_fixed, mask_moving

    def load_model(self):
        """Load the demo Flexi_CT_2D wrapper and expose its backbone features."""
        checkpoint = resolve_flexict_checkpoint("2d", self.ckpt_path)
        print(f"Loading Flexi_CT_2D weights from {checkpoint}")
        model = Flexi_CT_2D(checkpoint_path=checkpoint, device=self.device)
        _set_patch_size(model.backbone, self.patch_size)
        model.eval()
        return model

    def encode_3D_gap(self, input_arr, gap=3):


        imageH, imageW, slice_num = input_arr.shape

        """new uniform resize"""
        feature_height = self.feature_height
        feature_width = self.feature_width
        self.slice_num = slice_num


        input_arr = resize(input_arr, (feature_height*self.patch_size, feature_width*self.patch_size, slice_num), anti_aliasing=True)

        print(self.patch_size)
        print(feature_height, feature_width, slice_num)
        print('resized input shape', input_arr.shape)

        # 3D image into 2D model, stack each slices feature
        img_feature = np.zeros([feature_height * feature_width, slice_num, self.embed_dim], dtype=np.float32)
        encoding_slice_idx = np.arange(0, slice_num-1, gap).tolist()
        encoding_slice_idx.append(slice_num-1)

        # Batch encoding: collect all slices to encode and process in batches
        num_slices_to_encode = len(encoding_slice_idx)
        batch_size = self.batch_size  # Configurable batch size for DINO forward passes

        # Collect all slices into a list
        slices_to_encode = []
        for slice_id in encoding_slice_idx:
            input_slice = input_arr[:, :, slice_id, np.newaxis]  # H x W x 1
            slices_to_encode.append(input_slice)

        # Stack into batch tensor: [N, H, W, 1]
        slices_batch = np.stack(slices_to_encode, axis=0)  # N x H x W x 1

        # Process in batches to avoid OOM
        all_features = []
        for batch_start in range(0, num_slices_to_encode, batch_size):
            batch_end = min(batch_start + batch_size, num_slices_to_encode)
            batch_slices = slices_batch[batch_start:batch_end]  # B x H x W x 1

            # Extract features for batch
            batch_features = self.extract_dinov2_feature_batch(batch_slices)  # B x N x C
            all_features.append(batch_features)
            print(f"\rEncoded slices {batch_start+1}-{batch_end}/{num_slices_to_encode}", end="")

        print()  # New line after progress

        # Concatenate all batch features
        all_features = np.concatenate(all_features, axis=0)  # num_slices_to_encode x N x C

        # Assign encoded features to their slice positions
        for idx, slice_id in enumerate(encoding_slice_idx):
            img_feature[:, slice_id, :] = all_features[idx]

        # Interpolate features for skipped slices
        prev_slice = 0
        for slice_id in encoding_slice_idx:
            if slice_id > 0 and slice_id < slice_num-1:
                for i in range(1, gap):
                    slice_id_gap = slice_id - i
                    if slice_id_gap >= 0:
                        # Linear interpolation between prev_slice and slice_id
                        weight_current = (gap - i) / gap
                        weight_prev = i / gap
                        img_feature[:, slice_id_gap, :] = (
                            img_feature[:, slice_id, :] * weight_current +
                            img_feature[:, prev_slice, :] * weight_prev
                        )
            elif slice_id == slice_num-1:
                last_gap = slice_num - encoding_slice_idx[-2]
                for i in range(1, last_gap):
                    slice_id_gap = slice_num - i
                    weight_current = (last_gap - i) / last_gap
                    weight_prev = i / last_gap
                    img_feature[:, slice_id_gap, :] = (
                        img_feature[:, slice_id, :] * weight_current +
                        img_feature[:, prev_slice, :] * weight_prev
                    )
            prev_slice = slice_id

        img_feature = img_feature.reshape([feature_height * feature_width * slice_num, self.embed_dim])


        return img_feature


    def extract_slice_feature(self, input_arr_orig, mask=True):

        """input single slice 2d, output the feature of that slice"""

        input_arr = resize(input_arr_orig, (self.feature_height*self.patch_size, self.feature_width*self.patch_size), anti_aliasing=True)
        if mask:
            input_arr_masksize = resize(input_arr_orig, (self.feature_height, self.feature_width), anti_aliasing=True)
            pca_mask = extract_lung_mask(input_arr_masksize).flatten().astype(bool)

        input_slice = input_arr[:, :, np.newaxis]
        featrure = self.extract_dinov2_feature(input_slice)

        featrure = einops.rearrange(featrure, '1 n c -> n c')

        if mask:
            return featrure, pca_mask
        return featrure, np.ones(featrure.shape[0], dtype=bool)


def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet


def compute_95_hausdorff_distance(seg1, seg2):
    # Assuming seg1 and seg2 are binary segmentation masks
    u_indices = np.array(np.where(seg1)).T
    v_indices = np.array(np.where(seg2)).T

    # Compute all pairwise distances between the two sets of points
    distances = cdist(u_indices, v_indices, 'euclidean')

    # Flatten the distance matrix and sort the distances
    sorted_distances = np.sort(distances, axis=None)

    # Find the 95th percentile distance
    hd_95 = np.percentile(sorted_distances, 95)
    return hd_95


def compute_label_wise_95hd(seg1, seg2, labels):
    hd95_results = []
    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = seg1 == label
        seg2_label = seg2 == label

        # Compute 95% HD for the current label
        hd95 = compute_95_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd95)

    return hd95_results

def score_case(seg_fixed, seg_moving, disp_field, fixed_arr, case, label_list, spacing=1, roi_mask=None):

    print('disp shape in score_case',disp_field.shape)

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)

    dice_coefficient = img_op.apply_deformation_and_compute_dice(
        seg_fixed, seg_moving, disp_field, fixed_arr, case, num_classes=label_list, roi_mask=roi_mask
    )
    dice_coefficient = np.array(dice_coefficient)

    hd95 = img_op.warp_compute_label_wise_95hd(seg_fixed, seg_moving, label_list, disp_field, roi_mask=roi_mask)
    hd95= np.array(hd95)

    return {'DICE': dice_coefficient,
            'LogJacDetStd': log_jac_det[2:-2, 2:-2, 2:-2].std(),
            'HD95': hd95}


def load_csv_pairs(csv_path):
    """Load pairs from a CSV file, skipping header rows that contain non-path data."""
    pair_list = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                pair_list.append(row)
    return pair_list


def validate_pairs(pair_list, dataset_dir, image_key, label_key):
    """
    Validate that all image and label files referenced in pair_list exist.
    Prints warnings for missing files and returns a filtered list of valid pairs.
    """
    valid_pairs = []
    missing_count = 0
    for pair in pair_list:
        moving_fn = pair[0]
        fixed_fn = pair[1]
        files_to_check = [
            os.path.join(dataset_dir, image_key, moving_fn),
            os.path.join(dataset_dir, image_key, fixed_fn),
            os.path.join(dataset_dir, label_key, moving_fn),
            os.path.join(dataset_dir, label_key, fixed_fn),
        ]
        all_exist = True
        for fpath in files_to_check:
            if not os.path.exists(fpath):
                print(f'WARNING: Missing file: {fpath}')
                all_exist = False
                missing_count += 1
        if all_exist:
            valid_pairs.append(pair)
        else:
            print(f'  -> Skipping pair: {moving_fn} <-> {fixed_fn}')
    print(f'Validation complete: {len(valid_pairs)}/{len(pair_list)} pairs are valid '
          f'({missing_count} missing files).')
    return valid_pairs


def run_fold(fold_idx, test_pairs, dino_reg_model, configs, dataset_dir, image_key, label_key,
             mask_key, use_masksTr, output_dir_fold, exp_note, label_list, save_feature=False):
    """
    Run inference and evaluation for a single fold's test pairs.
    Saves per-fold metric files to output_dir_fold.
    """
    os.makedirs(output_dir_fold, exist_ok=True)

    quantify = True
    DICE_list = []
    LogJacDetStd_list = []
    hd_list = []

    for i, pair in enumerate(test_pairs):
        print(f'[Fold {fold_idx}] case {i}/{len(test_pairs)}')

        moving_fn = pair[0]
        fixed_fn = pair[1]

        fixed_basename = os.path.basename(fixed_fn).split('.')[0]
        moving_basename = os.path.basename(moving_fn).split('.')[0]

        img_fixed = nib.load(os.path.join(dataset_dir, image_key, fixed_fn))
        img_moving = nib.load(os.path.join(dataset_dir, image_key, moving_fn))

        arr_fixed = img_fixed.get_fdata()
        arr_moving = img_moving.get_fdata()

        aff_mov = img_moving.affine

        seg_fixed = nib.load(os.path.join(dataset_dir, label_key, fixed_fn)).get_fdata()
        seg_moving = nib.load(os.path.join(dataset_dir, label_key, moving_fn)).get_fdata()
        roi_eval = None
        if use_masksTr:
            roi_fixed = nib.load(os.path.join(dataset_dir, mask_key, fixed_fn)).get_fdata() > 0.5
            roi_eval = roi_fixed

        disp_init = None

        disp = dino_reg_model.case_inference(
            arr_moving, arr_fixed, arr_moving.shape, aff_mov,
            case_id=fixed_basename,
            disp_init=disp_init,
            grid_sp_adam=configs['fm_downsample'],
            DINOReg_useMask=configs['DINOReg_useMask'],
            save_feature=save_feature,
            output_dir=output_dir_fold
        )

        # Save displacement field
        disp_img = nib.Nifti1Image(disp, aff_mov)
        nib.save(disp_img, os.path.join(output_dir_fold, '{}_to_{}_disp_{}.nii.gz'.format(
            moving_basename, fixed_basename, exp_note)))

        disp = np.moveaxis(disp, 3, 0)

        # Warp moving image
        D, H, W = arr_moving.shape
        identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
        warped_image = map_coordinates(arr_moving, identity + disp, order=0)
        warped_image_nib = nib.Nifti1Image(warped_image, aff_mov)
        nib.save(warped_image_nib, os.path.join(output_dir_fold, '{}_to_{}_warped_{}.nii.gz'.format(
            moving_basename, fixed_basename, exp_note)))

        if quantify:
            arr_fixed_norm = to_lungCT_window(arr_fixed)
            result = score_case(seg_fixed, seg_moving, disp, arr_fixed_norm, i, label_list, roi_mask=roi_eval)

            DICE_list.append(result['DICE'])
            LogJacDetStd_list.append(result['LogJacDetStd'])
            hd_list.append(result['HD95'])
            print(f'[Fold {fold_idx}] DICE:', result['DICE'])
            print(f'[Fold {fold_idx}] LogJacDetStd', result['LogJacDetStd'])
            print(f'[Fold {fold_idx}] HD95', result['HD95'])

    if quantify and len(DICE_list) > 0:
        temp_dice = np.asarray(DICE_list)
        print(f'[Fold {fold_idx}] temp_dice shape', temp_dice.shape)

        mean_value = np.nanmean(temp_dice)
        print(f'[Fold {fold_idx}] DICE mean', np.nanmean(temp_dice), 'DICE std', np.nanstd(temp_dice))
        print(f'[Fold {fold_idx}] LogJacDetStd mean', np.mean(np.asarray(LogJacDetStd_list)),
              'LogJacDetStd std', np.std(np.asarray(LogJacDetStd_list)))

        np.savetxt(
            os.path.join(output_dir_fold, 'summary_fold{}_{}.txt'.format(fold_idx, exp_note)),
            np.array([np.nanmean(temp_dice), np.nanstd(temp_dice),
                      np.mean(np.asarray(LogJacDetStd_list)), np.std(np.asarray(LogJacDetStd_list))]),
            fmt='%.4f'
        )
        np.savetxt(
            os.path.join(output_dir_fold, 'DICE_fold{}_{}.txt'.format(fold_idx, exp_note)),
            temp_dice,
            fmt='%.4f'
        )
        np.savetxt(
            os.path.join(output_dir_fold, 'LogJacDetStd_fold{}_{}.txt'.format(fold_idx, exp_note)),
            np.asarray(LogJacDetStd_list),
            fmt='%.3f'
        )
        np.savetxt(
            os.path.join(output_dir_fold, 'HD95_fold{}_{}.txt'.format(fold_idx, exp_note)),
            np.asarray(hd_list),
            fmt='%.4f'
        )
        print(f'[Fold {fold_idx}] Metrics saved to {output_dir_fold}')

    return {
        'DICE': DICE_list,
        'LogJacDetStd': LogJacDetStd_list,
        'HD95': hd_list,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True, help="Learn2Reg-style dataset root.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Explicit Flexi_CT_2D checkpoint. If omitted, uses FLEXICT_CHECKPOINT, "
            "then FLEXICT_2D_CHECKPOINT. No private-host default is bundled."
        ),
    )
    parser.add_argument("--exp_note", default="flexi_ct_2d_p8")
    parser.add_argument("--image_key", default="imagesTr")
    parser.add_argument("--label_key", default="labelsTr")
    parser.add_argument("--mask_key", default="masksTr")
    parser.add_argument("--use_masks_tr", action="store_true", help="Use masksTr for ROI-limited evaluation.")
    parser.add_argument("--split_csvs", nargs="+", default=list(DEFAULT_SPLIT_CSVS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only_fold", type=int, default=None,
                        help="Run only this 1-indexed fold, useful for isolated long jobs.")
    parser.add_argument("--only_pair_index", type=int, default=None,
                        help="Run only this 0-indexed valid pair after CSV loading and validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--feature_size", type=_parse_feature_size, default=(80, 70))
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--smooth_weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=6.0)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--fm_downsample", type=int, default=1)
    parser.add_argument("--iter_smooth_num", type=int, default=2)
    parser.add_argument("--iter_smooth_kernel", type=int, default=7)
    parser.add_argument("--final_upsample", type=int, default=1)
    parser.add_argument("--fix_preprocess", default="meddinov3_ct",
                        choices=["lung", "abdomen", "soft_tissue", "mr", "meddinov3_ct", "meddinov3_mr"])
    parser.add_argument("--mov_preprocess", default="meddinov3_ct",
                        choices=["lung", "abdomen", "soft_tissue", "mr", "meddinov3_ct", "meddinov3_mr"])
    parser.add_argument("--use_saved_pca", action="store_true")
    parser.add_argument("--no_dino_reg_mask", action="store_true")
    parser.add_argument("--save_feature", action="store_true")
    return parser


def _load_all_pairs(dataset_dir, split_csvs):
    all_pairs = []
    for split_csv in split_csvs:
        csv_path = os.path.join(dataset_dir, split_csv)
        if not os.path.exists(csv_path):
            print(f'WARNING: {csv_path} not found, skipping.')
            continue
        split_pairs = load_csv_pairs(csv_path)
        print(f'Loaded {len(split_pairs)} pairs from {split_csv}')
        all_pairs.extend(split_pairs)
    return all_pairs


def main():
    args = build_arg_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because ConvexAdam uses CUDA tensors internally.")

    time_start = time.time()
    dataset_dir = args.dataset_dir
    output_dir_0 = args.output_dir
    exp_note = args.exp_note
    os.makedirs(output_dir_0, exist_ok=True)

    configs = {
        'patch_size': args.patch_size,
        'ckpt_path': args.checkpoint,
        'batch_size': args.batch_size,
        'smooth_weight': args.smooth_weight,
        'lr': args.lr,
        'num_iter': args.num_iter,
        'fm_downsample': args.fm_downsample,
        'feature_size': args.feature_size,
        'gap': args.gap,
        'useSavedPCA': args.use_saved_pca,
        'DINOReg_useMask': not args.no_dino_reg_mask,
        'convex': False,
        'ztrans': False,
        'iter_smooth_num': args.iter_smooth_num,
        'iter_smooth_kernel': args.iter_smooth_kernel,
        'final_upsample': args.final_upsample,
        'mask': 'slice fill stack',
        'fix_preprocess': args.fix_preprocess,
        'mov_preprocess': args.mov_preprocess,
        'device_id': args.device_id,
    }

    with open(os.path.join(output_dir_0, 'configs.json'), 'w') as f:
        json.dump(configs, f, indent=2)

    all_pairs = _load_all_pairs(dataset_dir, args.split_csvs)
    print(f'Total pairs loaded: {len(all_pairs)}')

    all_pairs = validate_pairs(all_pairs, dataset_dir, args.image_key, args.label_key)
    print(f'Valid pairs after file validation: {len(all_pairs)}')

    if len(all_pairs) == 0:
        print('ERROR: No valid pairs found. Exiting.')
        sys.exit(1)
    if args.only_pair_index is None and len(all_pairs) < args.folds:
        raise ValueError(f"--folds={args.folds} requires at least {args.folds} valid pairs, got {len(all_pairs)}.")

    structures_csv = os.path.join(dataset_dir, 'structures.csv')
    if not os.path.exists(structures_csv):
        raise FileNotFoundError(f"structures.csv not found: {structures_csv}")
    df = pd.read_csv(structures_csv, header=None)
    label_list = df.iloc[0, :].tolist()

    if args.only_pair_index is not None:
        if args.only_pair_index < 0 or args.only_pair_index >= len(all_pairs):
            raise ValueError(
                f"--only_pair_index={args.only_pair_index} is out of range for "
                f"{len(all_pairs)} valid pairs."
            )
        dino_reg_model = dinoReg(
            device_id=args.device_id,
            batch_size=configs['batch_size'],
            lr=configs['lr'],
            smooth_weight=configs['smooth_weight'],
            num_iter=configs['num_iter'],
            feat_size=configs['feature_size'],
            ckpt_path=configs['ckpt_path'],
            patch_size=configs['patch_size'],
            configs=configs
        )
        pair_output_dir = os.path.join(output_dir_0, f'pair_{args.only_pair_index + 1}')
        run_fold(
            fold_idx=args.only_pair_index + 1,
            test_pairs=[all_pairs[args.only_pair_index]],
            dino_reg_model=dino_reg_model,
            configs=configs,
            dataset_dir=dataset_dir,
            image_key=args.image_key,
            label_key=args.label_key,
            mask_key=args.mask_key,
            use_masksTr=args.use_masks_tr,
            output_dir_fold=pair_output_dir,
            exp_note=exp_note,
            label_list=label_list,
            save_feature=args.save_feature,
        )
        print('Total time elapsed', time.time() - time_start, 'exp_note', exp_note)
        print(pair_output_dir)
        return

    all_pairs_arr = np.array(all_pairs, dtype=object)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    dino_reg_model = dinoReg(
        device_id=args.device_id,
        batch_size=configs['batch_size'],
        lr=configs['lr'],
        smooth_weight=configs['smooth_weight'],
        num_iter=configs['num_iter'],
        feat_size=configs['feature_size'],
        ckpt_path=configs['ckpt_path'],
        patch_size=configs['patch_size'],
        configs=configs
    )

    all_fold_results = []

    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(all_pairs_arr)):
        if args.only_fold is not None and fold_idx + 1 != args.only_fold:
            continue

        print(f'\n{"="*60}')
        print(f'Starting Fold {fold_idx + 1}/{args.folds}')
        print(f'  Train pairs: {len(train_indices)}, Test pairs: {len(test_indices)}')
        print(f'{"="*60}')

        test_pairs = all_pairs_arr[test_indices].tolist()
        output_dir_fold = os.path.join(output_dir_0, f'fold_{fold_idx + 1}')
        os.makedirs(output_dir_fold, exist_ok=True)

        pca_path = os.path.join(output_dir_fold, 'pca_matrix.npy')
        if configs['useSavedPCA'] and os.path.exists(pca_path):
            PCA_matrix = np.load(pca_path)
            print(f'[Fold {fold_idx + 1}] Loaded PCA matrix from {pca_path}')
        else:
            PCA_matrix = None
            if configs['useSavedPCA']:
                print(f'[Fold {fold_idx + 1}] PCA matrix not found, will compute and save it')

        dino_reg_model.pca_matrix = PCA_matrix
        dino_reg_model.pca_save_path = pca_path if configs['useSavedPCA'] else None
        dino_reg_model.eigenvalue_array = []

        fold_results = run_fold(
            fold_idx=fold_idx + 1,
            test_pairs=test_pairs,
            dino_reg_model=dino_reg_model,
            configs=configs,
            dataset_dir=dataset_dir,
            image_key=args.image_key,
            label_key=args.label_key,
            mask_key=args.mask_key,
            use_masksTr=args.use_masks_tr,
            output_dir_fold=output_dir_fold,
            exp_note=exp_note,
            label_list=label_list,
            save_feature=args.save_feature,
        )
        all_fold_results.append(fold_results)

    print(f'\n{"="*60}')
    print('Cross-validation complete. Aggregating results across all folds...')
    print(f'{"="*60}')

    all_dice = []
    all_ljd = []
    all_hd95 = []

    for fold_results in all_fold_results:
        if fold_results['DICE']:
            all_dice.extend(fold_results['DICE'])
        if fold_results['LogJacDetStd']:
            all_ljd.extend(fold_results['LogJacDetStd'])
        if fold_results['HD95']:
            all_hd95.extend(fold_results['HD95'])

    if all_dice:
        all_dice_arr = np.asarray(all_dice)
        all_ljd_arr = np.asarray(all_ljd)
        all_hd95_arr = np.asarray(all_hd95)

        print(f'Overall DICE mean: {np.nanmean(all_dice_arr):.4f}, std: {np.nanstd(all_dice_arr):.4f}')
        print(f'Overall LogJacDetStd mean: {np.mean(all_ljd_arr):.4f}, std: {np.std(all_ljd_arr):.4f}')
        print(f'Overall HD95 mean: {np.nanmean(all_hd95_arr):.4f}, std: {np.nanstd(all_hd95_arr):.4f}')

        np.savetxt(
            os.path.join(output_dir_0, 'summary_all_folds_{}.txt'.format(exp_note)),
            np.array([
                np.nanmean(all_dice_arr), np.nanstd(all_dice_arr),
                np.mean(all_ljd_arr), np.std(all_ljd_arr),
                np.nanmean(all_hd95_arr), np.nanstd(all_hd95_arr),
            ]),
            fmt='%.4f',
            header='DICE_mean DICE_std LogJacDetStd_mean LogJacDetStd_std HD95_mean HD95_std'
        )
        np.savetxt(
            os.path.join(output_dir_0, 'DICE_all_folds_{}.txt'.format(exp_note)),
            all_dice_arr,
            fmt='%.4f'
        )
        np.savetxt(
            os.path.join(output_dir_0, 'LogJacDetStd_all_folds_{}.txt'.format(exp_note)),
            all_ljd_arr,
            fmt='%.3f'
        )
        np.savetxt(
            os.path.join(output_dir_0, 'HD95_all_folds_{}.txt'.format(exp_note)),
            all_hd95_arr,
            fmt='%.4f'
        )

    print('Total time elapsed', time.time() - time_start, 'exp_note', exp_note)
    print(output_dir_0)


if __name__ == '__main__':
    main()
