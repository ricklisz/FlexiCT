import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import math
import nibabel as nib
from os import path


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win
        # win = [27] * ndims if self.win is None else self.win #takes forever

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


def GPTNCC(img1, img2, window_size=9):
    """
    Compute the local cross-correlation between two 5D images.

    Parameters:
    - img1: a 5D PyTorch tensor of shape (B, C, H, W, D)
    - img2: a 5D PyTorch tensor of shape (B, C, H, W, D)
    - window_size: an integer defining the cubic window size (n x n x n)

    Returns:
    - cc_map: a 5D PyTorch tensor of local cross-correlation values
    """
    # Ensure the input images are on the same device and of the same type
    img2 = img2.to(img1.device).type_as(img1)

    # Pad images to handle borders
    pad_size = window_size // 2
    img1 = F.pad(img1, (pad_size,) * 6)
    img2 = F.pad(img2, (pad_size,) * 6)

    B, C, H, W, D = img1.shape

    # Create the window volume for the local sums
    window_volume = window_size ** 3

    # Compute the sum of squares
    I2 = img1 * img1
    J2 = img2 * img2
    IJ = img1 * img2

    # Convolution with a sum filter to compute local sums
    sum_filter = torch.ones((C, 1, window_size, window_size, window_size)).to(img1.device)
    I_sum = F.conv3d(img1, sum_filter, padding=0, groups=C)
    J_sum = F.conv3d(img2, sum_filter, padding=0, groups=C)
    I2_sum = F.conv3d(I2, sum_filter, padding=0, groups=C)
    J2_sum = F.conv3d(J2, sum_filter, padding=0, groups=C)
    IJ_sum = F.conv3d(IJ, sum_filter, padding=0, groups=C)

    # Compute means
    u_I = I_sum / window_volume
    u_J = J_sum / window_volume

    # Compute cross correlation
    cross = IJ_sum - u_I * J_sum - u_J * I_sum + u_I * u_J * window_volume
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * window_volume
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * window_volume

    cc_map = cross * cross / (I_var * J_var + 1e-5)

    # Normalize and remove padding
    # cc_map = cc_map[:, :, pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

    return -torch.mean(cc_map)

def GPTNCC_w0(img1, img2, weights, window_size=9):
    """

    this version contains C hyperparameter, which is the number of channels, regularized by L1 for sparsity
    """
    # Ensure the input images are on the same device and of the same type
    img2 = img2.to(img1.device).type_as(img1)

    # Pad images to handle borders
    pad_size = window_size // 2
    img1 = F.pad(img1, (pad_size,) * 6)
    img2 = F.pad(img2, (pad_size,) * 6)

    B, C, H, W, D = img1.shape

    # Create the window volume for the local sums
    window_volume = window_size ** 3

    # Compute the sum of squares
    I2 = img1 * img1
    J2 = img2 * img2
    IJ = img1 * img2

    # Convolution with a sum filter to compute local sums
    sum_filter = torch.ones((C, 1, window_size, window_size, window_size)).to(img1.device)
    #make weights sum to 1
    # weights = weights / torch.sum(weights)
    # sum_filter = sum_filter * weights.view(C, 1, 1, 1, 1)
    sum_filter = sum_filter * torch.abs(weights.view(C, 1, 1, 1, 1))


    I_sum = F.conv3d(img1, sum_filter, padding=0, groups=C)
    J_sum = F.conv3d(img2, sum_filter, padding=0, groups=C)
    I2_sum = F.conv3d(I2, sum_filter, padding=0, groups=C)
    J2_sum = F.conv3d(J2, sum_filter, padding=0, groups=C)
    IJ_sum = F.conv3d(IJ, sum_filter, padding=0, groups=C)

    # Compute means
    u_I = I_sum / window_volume
    u_J = J_sum / window_volume

    # Compute cross correlation
    cross = IJ_sum - u_I * J_sum - u_J * I_sum + u_I * u_J * window_volume
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * window_volume
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * window_volume

    cc_map = cross * cross / (I_var * J_var + 1e-5)

    # Normalize and remove padding
    # cc_map = cc_map[:, :, pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

    return -torch.mean(cc_map)



class nccTrue:
    """
    Local (over window) normalized cross correlation loss for 5D inputs.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):
        # Assumes inputs are of shape [B, C, H, W, D]
        assert y_true.size() == y_pred.size(), "Input tensors must have the same shape"

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims


        # normalize
        Ii = Ii - torch.mean(Ii, dim=1, keepdim=True)
        Ji = Ji - torch.mean(Ji, dim=1, keepdim=True)
        #dot product on channel is
        IJ = torch.sum(Ii * Ji, dim=1, keepdim=True)

        # print('IJ shape',IJ.shape, IJ.max(), IJ.min(), IJ.mean(), IJ.std())

        # magnitude1 = torch.sqrt(torch.sum(Ii ** 2, dim=1))
        # magnitude2 = torch.sqrt(torch.sum(Ji ** 2, dim=1))

        magnitude1 = torch.sum(Ii ** 2, dim=1)
        magnitude2 = torch.sum(Ji ** 2, dim=1)

        # print('magnitude1 shape',magnitude1.shape, magnitude1.max(), magnitude1.min(), magnitude1.mean(), magnitude1.std())

        ncc = IJ / (magnitude1 * magnitude2 + 1e-5)

        # print('ncc shape',ncc.shape, ncc.max(), ncc.min(), ncc.mean(), ncc.std())

        # ncc_np = ncc.cpu().detach().numpy()[0,0,:,:,:]
        # aff = np.eye(4); aff[0,0] = -1; aff[1,1] = -1
        # ncc_img = nib.Nifti1Image(ncc_np, aff)
        # timestring = datetime.now().strftime("%Y%m%d-%H%M%S")
        # out_dir = '/fast/songx/tempFiles/lungReg/vis/ncc'
        # nib.save(ncc_img, path.join(out_dir,'ncc_{}.nii.gz'.format(timestring)))

        # Return the mean negative NCC as loss
        return -torch.mean(ncc)


class NCC_neighbor:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
        B, C, H, W, D = Ii.size()


        # set window size
        win = [9] * ndims if self.win is None else self.win
        # win = [27] * ndims if self.win is None else self.win #takes forever

        # compute filters
        # sum_filt = torch.ones([1, 1, *win]).to("cuda")
        sum_filt_C = torch.ones([1, C, *win]).to("cuda")
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = torch.mean(Ii * Ii, dim=1, keepdim=True)
        J2 = torch.mean(Ji * Ji, dim=1, keepdim=True)
        IJ = torch.mean(Ii * Ji, dim=1, keepdim=True)


        I_sum = conv_fn(Ii, sum_filt_C, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt_C, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class MutualInformationLoss:
    """
    Mutual Information loss for comparing similarity between two images.
    """

    def __init__(self, bins=64, bin_range=(0, 1), sigma=0.05):
        self.bins = bins
        self.bin_range = bin_range
        self.sigma = sigma

    def differentiable_2d_histogram(self, image1, image2):
        """
        Calculate a differentiable 2D histogram for two images.
        """
        min_val, max_val = self.bin_range
        image1 = torch.clamp(image1, min_val, max_val)
        image2 = torch.clamp(image2, min_val, max_val)

        bin_centers = torch.linspace(min_val, max_val, self.bins).view(1, -1).to(image1.device)

        flat_image1 = image1.flatten().unsqueeze(1)
        flat_image2 = image2.flatten().unsqueeze(1)

        hist_1d_image1 = torch.exp(-0.5 * ((flat_image1 - bin_centers) / self.sigma) ** 2)
        hist_1d_image1 /= hist_1d_image1.sum(dim=1, keepdim=True)

        hist_1d_image2 = torch.exp(-0.5 * ((flat_image2 - bin_centers) / self.sigma) ** 2)
        hist_1d_image2 /= hist_1d_image2.sum(dim=1, keepdim=True)

        hist_2d = torch.mm(hist_1d_image1.t(), hist_1d_image2)
        hist_2d /= hist_2d.sum()

        return hist_2d

    def mutual_information(self, hist_2d):
        """
        Calculate mutual information from a 2D histogram.
        """
        p_x = hist_2d.sum(dim=1, keepdim=True)
        p_y = hist_2d.sum(dim=0, keepdim=True)

        h_xy = -torch.sum(hist_2d[hist_2d > 0] * torch.log(hist_2d[hist_2d > 0]))
        h_x = -torch.sum(p_x[p_x > 0] * torch.log(p_x[p_x > 0]))
        h_y = -torch.sum(p_y[p_y > 0] * torch.log(p_y[p_y > 0]))

        mi = h_x + h_y - h_xy
        return mi

    def loss(self, image1, image2):
        """
        Compute the mutual information loss between two images.
        """
        hist_2d = self.differentiable_2d_histogram(image1, image2)
        return -self.mutual_information(hist_2d)  # Negative MI for loss minimization

class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()