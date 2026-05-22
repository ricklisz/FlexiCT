import sys

import numpy as np
import torch
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
from skimage import morphology
from skimage.measure import label, regionprops
from scipy import ndimage


#take out slices that are uniform
def remove_uniform_intensity_slices(image_data):
    slices_to_keep_indices = [i for i in range(image_data.shape[2])
                              if not np.max(image_data[:,:,i]) == np.min(image_data[:,:,i])]

    # Extract slices to keep
    filtered_image_data = image_data[:,:,slices_to_keep_indices]

    return filtered_image_data, slices_to_keep_indices

def reconstruct_image(filtered_image_data, slices_to_keep_indices, original_shape, default_intensity=0):

    # Initialize an empty array with the original shape
    reconstructed_image = np.full(original_shape, default_intensity, dtype=filtered_image_data.dtype)

    # Insert the filtered slices back into their original positions
    for i, slice_index in enumerate(slices_to_keep_indices):
        reconstructed_image[:,:,slice_index] = filtered_image_data[:,:,i]

    return reconstructed_image

def to_lungCT_window(image_data, wl=-600, ww=1500):
    img = np.clip(image_data, wl - ww / 2, wl + ww / 2)
    # normalzie
    img = (img - (wl - ww / 2)) / ww
    return img

def MR_normalize(image_data, quantile=97):
    #normalize to [0,max], max is 95th percentile
    max = np.percentile(image_data, quantile)
    image_data = np.clip(image_data, image_data.min(), max)
    image_data = (image_data - image_data.min()) / (max - image_data.min())
    return image_data

def clip_and_normalize_image(image, lower_percentile=0.5, upper_percentile=95.0):
    """
    Clip image intensity values to a specified range and normalize to [0, 1].

    Parameters:
    - image: A numpy array representing the image.
    - lower_percentile: Lower percentile bound for clipping.
    - upper_percentile: Upper percentile bound for clipping.

    Returns:
    - A numpy array of the same shape as `image`, with intensity values normalized to [0, 1].
    """
    # Compute the percentile values for clipping
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)

    # Clip the image intensities
    clipped_image = np.clip(image, lower_bound, upper_bound)

    # Normalize the clipped image to [0, 1]
    # Avoid division by zero in case lower_bound == upper_bound
    if upper_bound > lower_bound:
        normalized_image = (clipped_image - lower_bound) / (upper_bound - lower_bound)
    else:
        normalized_image = np.zeros_like(image)

    return normalized_image


def pca_lowrank_transform(all_features, n_components, mat=False):
    # Convert input features to a PyTorch tensor if not already
    all_features_tensor = torch.tensor(all_features, dtype=torch.float32)

    # Perform PCA using torch.pca_lowrank
    U, S, V = torch.pca_lowrank(all_features_tensor, q=n_components)

    # Compute the reduced representation by projecting the data onto the principal components
    # Note: The original data is projected onto the principal components to get the reduced data
    reduced_data = torch.matmul(all_features_tensor, V[:, :n_components])

    # Step 1: Square the singular values to get the eigenvalues
    eigenvalues = S.pow(2)
    # Step 2: Calculate the total variance
    total_variance = eigenvalues.sum()
    # Step 3: Normalize each eigenvalue to get the proportion of variance
    proportion_variance_explained = eigenvalues / total_variance

    if mat:
        return reduced_data, proportion_variance_explained, V
    return reduced_data, proportion_variance_explained


def compute_dice_coefficient(mask1, mask2):
    """
    Compute the Dice Similarity Coefficient between two binary masks.
    """
    intersection = np.sum(mask1 * mask2)
    size1 = np.sum(mask1)
    size2 = np.sum(mask2)
    return (2.0 * intersection + 1e-5 ) / (size1 + size2 + 1e-5) 

def warp_arr(arr, disp):

    D, H, W = arr.shape
    identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    warped_arr = map_coordinates(arr, identity + disp, order=0, mode='nearest')
    # warped_arr = np.where(warped_arr > 0.5, 1, 0)
    return warped_arr

def apply_deformation_and_compute_dice(fixed_arr, moving_arr, disp, affine, case, num_classes, roi_mask=None):
    """
    Apply deformation to the moving mask and compute the DICE coefficient with the fixed mask.
    :param fixed_arr: Fixed binary segmentation array
    :param moving_arr: Moving binary segmentation array to be deformed
    :param disp: Deformation field
    :param output_dir: Directory to save the warped image
    :param fixed_image: Nifti1Image object of fixed image for affine information
    :return: Dice Similarity Coefficient
    """


    warped_image = warp_arr(moving_arr, disp)
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
        fixed_arr = np.where(roi_mask, fixed_arr, 0)
        warped_image = np.where(roi_mask, warped_image, 0)
    dice = []

    if isinstance(num_classes, list):
        for i in num_classes:
            if np.sum(fixed_arr == i) == 0 or np.sum(warped_image == i) == 0:
                dice.append(np.nan)
            else:
                dice.append(compute_dice_coefficient(fixed_arr == i, warped_image == i))

    elif isinstance(num_classes, int):
        for i in range(1, num_classes + 1):
            if np.sum(fixed_arr == i) == 0 and np.sum(warped_image == i) == 0:
                dice.append(np.nan)
            else:
                dice.append(compute_dice_coefficient(fixed_arr == i, warped_image == i))
    else:
        print("ERROR: Argument is neither a list nor an integer.")
        sys.exit()

    # Save the warped image
    # warped_nifti = nib.Nifti1Image(warped_image, affine)
    # label_out_dir = path.join(output_dir, exp_note, 'label')
    # out_fn = path.join(label_out_dir, 'labelWarped_{}.nii.gz'.format(case))
    # os.makedirs(path.dirname(out_fn), exist_ok=True)
    # nib.save(warped_nifti, out_fn)


    # Compute the DICE coefficient
    return dice


def compute_hausdorff_distance(seg1, seg2):
    # Assuming seg1 and seg2 are binary segmentation masks
    # Extract boundary points or use entire segmentation, depending on your application
    # For simplicity, let's use np.where to find indices of the segmented objects
    u_indices = np.array(np.where(seg1)).T
    v_indices = np.array(np.where(seg2)).T

    # Compute the directed Hausdorff distances and take the maximum to get the true Hausdorff distance
    hd1 = directed_hausdorff(u_indices, v_indices)[0]
    hd2 = directed_hausdorff(v_indices, u_indices)[0]

    hd = max(hd1, hd2)
    return hd


def compute_fast_95hd(seg1, seg2):
    # Extract the points where each segmentation mask is True
    points_seg1 = np.argwhere(seg1)
    points_seg2 = np.argwhere(seg2)

    # Check if either set of points is empty
    if points_seg1.size == 0 or points_seg2.size == 0:
        # Handle the case where there are no points to compare
        return np.nan  # Or another appropriate value or handling mechanism

    # Use NearestNeighbors to find the nearest distances
    nn = NearestNeighbors(n_neighbors=1)

    # Fit on seg2 points and find distances to seg1 points
    nn.fit(points_seg2)
    distances_1, _ = nn.kneighbors(points_seg1)
    distances_1 = distances_1.ravel()  # Ensure 1-dimensional

    # Fit on seg1 points and find distances to seg2 points
    nn.fit(points_seg1)
    distances_2, _ = nn.kneighbors(points_seg2)
    distances_2 = distances_2.ravel()  # Ensure 1-dimensional

    # Combine the distances and compute the 95th percentile
    all_distances = np.hstack((distances_1, distances_2))
    hd_95 = np.percentile(all_distances, 95)

    return hd_95


def warp_compute_label_wise_95hd(fix_seg, mov_seg, labels, disp, roi_mask=None):
    hd95_results = []

    warped = warp_arr(mov_seg, disp)
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)

    for label in labels:
        # Isolate current label in both segmentations
        seg1_label = fix_seg == label
        seg2_label = warped == label
        if roi_mask is not None:
            seg1_label = seg1_label & roi_mask
            seg2_label = seg2_label & roi_mask

        # Compute 95% HD for the current label
        hd95 = compute_fast_95hd(seg1_label, seg2_label)
        # hd95 = compute_hausdorff_distance(seg1_label, seg2_label)
        hd95_results.append(hd95)

    return hd95_results

def get_largest_roi(image):
    # Thresholding
    binary_mask = np.where(image > 0.05, 1.0, 0)

    # Morphological opening
    opened_mask = morphology.opening(binary_mask)

    # Label connected components
    labeled_mask = label(opened_mask)

    # Find largest connected component
    largest_component = None
    largest_area = 0
    for region in regionprops(labeled_mask):
        if region.area > largest_area:
            largest_area = region.area
            largest_component = region

    # Create binary mask of the largest connected component
    largest_roi = np.zeros_like(image)
    if largest_component is not None:
        largest_roi[labeled_mask == largest_component.label] = 1

    return largest_roi

def extract_lung_mask(image, threshold_value=0.01):
    """
    Extracts a binary mask covering the foreground of a lung CT slice.
    
    Args:
        image (np.array): The input lung CT slice (2D array).
        threshold_value (float): Threshold value for binarization. Default is 0.
        
    Returns:
        mask (np.array): Binary mask covering the lungs in the image.
    """
    # Step 1: Thresholding to separate foreground (lungs) from background
    thresh = np.where(image > threshold_value, 1, 0).astype(np.uint8)
    # print('foreground pixels',thresh.sum())
    if np.sum(thresh) <= 1000:
        return thresh

    # Step 2: Label connected components and keep the largest component
    labeled_array, num_features = ndimage.label(thresh)

    bincount_values = np.bincount(labeled_array.flat)[1:]
    if bincount_values.size > 0:
        largest_component_label = np.argmax(bincount_values) + 1
    else:
        largest_component_label = 0  # incase there are no foreground pixels, empty slice

    largest_component_mask = (labeled_array == largest_component_label).astype(np.uint8)

    # Step 3: morphological opening to remove small noise
    # kernel = np.ones((5, 5), dtype=np.uint8)

    # Step 3: Fill lung holes using binary fill holes
    filled_mask = ndimage.binary_fill_holes(largest_component_mask).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    opened_mask = ndimage.binary_opening(filled_mask, structure=kernel)
    # return opened_mask



    # Get the largest connected component after opening
    labeled_mask = label(opened_mask)

    # Find largest connected component
    largest_component = None
    largest_area = 0
    for region in regionprops(labeled_mask):
        if region.area > largest_area:
            largest_area = region.area
            largest_component = region
    # Create binary mask of the largest connected component
    largest_roi = np.zeros_like(image)
    if largest_component is not None:
        largest_roi[labeled_mask == largest_component.label] = 1

    return largest_roi

    
    
