import dis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
from scipy.ndimage import distance_transform_edt as edt
from .convex_adam_utils import *
import time
import argparse
import nibabel as nib
import os
import sys
import einops
import warnings
from . import losses

warnings.filterwarnings("ignore")



# coupled convex optimisation with adam instance optimisation
def convex_adam_3d(feature_fixed,
                feature_moving,
                lambda_weight=1.25,
                grid_sp=6,
                disp_hw=4,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
                loss_func='NCC',
                lr=1,
                disp_init=None,
                feat_smooth=True,
                   adamW=False):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    mask_fixed = None
    mask_moving = None

    H, W, D = features_fix.shape[2:]
    print('ConvexAdam optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    # compute features and downsample (using average pooling)
    # with torch.no_grad():
    #
    #     features_fix, features_mov = extract_features(img_fixed=features_fix,
    #                                                   img_moving=features_mov,
    #                                                   mind_r=mind_r,
    #                                                   mind_d=mind_d,
    #                                                   use_mask=use_mask,
    #                                                   mask_fixed=mask_fixed,
    #                                                   mask_moving=mask_moving)
    #     # print(features_mov.shape)
    #
    #     features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
    #     features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
    n_ch = features_fix.shape[1]

    if disp_init is None:
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)


        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(
            0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)


        else:
            disp_hr = disp_soft

    else:
        disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():
            if feat_smooth:
                patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
                patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)
            else:
                patch_features_fix = features_fix
                patch_features_mov = features_mov

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        if adamW:
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
                padding=1).permute(0, 2, 3, 4, 1)
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
                # sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")


        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements

def convex_adam_3d_interSmooth(feature_fixed,
                feature_moving,
                mind_r=1,
                mind_d=2,
                lambda_weight=1.25,
                grid_sp=6,
                disp_hw=4,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
                loss_func='NCC',
                lr=1,
                disp_init=None):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    mask_fixed = None
    mask_moving = None

    H, W, D = features_fix.shape[2:]
    print('ConvexAdam optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    n_ch = features_fix.shape[1]

    if disp_init is None:
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)


        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(
            0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)


        else:
            disp_hr = disp_soft

    else:
        disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
                padding=1).permute(0, 2, 3, 4, 1)
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
                # sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")

            if iter % 100 == 0 and iter > 0:
                # Smooth the displacement field parameters in net
                with torch.no_grad():  # Ensure smoothing is not tracked by autograd
                    # Assuming net[0] is the layer you want to smooth
                    original_weights = net[0].weight.data
                    # print(original_weights.shape)
                    # Applying a simple averaging filter for demonstration; adjust as needed
                    # Here, we use a very naive form of "smoothing" by averaging adjacent elements
                    # This is purely illustrative and not a recommended practice for real use cases
                    kernel_smooth = 3
                    padding_smooth = kernel_smooth // 2
                    smooth_iter = 4
                    #triple smoothing
                    for smooth_iter_idx in range(smooth_iter):
                        original_weights = F.avg_pool3d(original_weights, kernel_size=kernel_smooth, stride=1, padding=padding_smooth,
                                                        count_include_pad=False)

                    # smoothed_weights = F.avg_pool3d(original_weights, kernel_size=3, stride=1, padding=1,
                    #                                 count_include_pad=False)
                    #
                    # Replace the original weights with the smoothed weights
                    net[0].weight.data = original_weights


        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements

def convex_adam_3d_param(feature_fixed,
                feature_moving,
                lambda_weight=1.25,
                selected_niter=80,
                grid_sp_adam=4, #feature map downsampling
                loss_func='NCC',
                lr=1,
                disp_init=None,
                iter_smooth_kernel=3,
                iter_smooth_num = 3,
                end_smooth_kernel = 3,
                final_upsample = 2
                ):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    mask_fixed = None
    mask_moving = None

    H, W, D = features_fix.shape[2:]
    print('optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    n_ch = features_fix.shape[1]



    if disp_init is None:
        print('disp_init is None, using Convex optimization')
        grid_sp = 6
        disp_hw = 4
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)


        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(
            0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        ic=True
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)
        else:
            disp_hr = disp_soft
    else:    
        disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            # disp_sample = F.avg_pool3d(
            #     F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
            #     padding=1).permute(0, 2, 3, 4, 1)
            

            iter_smooth_padding = (iter_smooth_kernel -1) // 2

            disp_sample = net[0].weight
            for smooth_num in range (iter_smooth_num):
                disp_sample = F.avg_pool3d(disp_sample, iter_smooth_kernel, stride=1, padding=iter_smooth_padding)
            disp_sample = disp_sample.permute(0, 2, 3, 4, 1)

            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                # sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")



        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam * final_upsample, size=(H*final_upsample, W*final_upsample, D*final_upsample), 
                                mode='trilinear', align_corners=False)

        #keep same smoothing when outputing result
        if end_smooth_kernel > 1:
            disp_hr = F.avg_pool3d(disp_hr, end_smooth_kernel, stride=1, padding=end_smooth_padding)
        
        

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements

def convex_adam_3d_param_dataSmooth(feature_fixed,
                feature_moving,
                lambda_weight=1.25,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=4, #feature map downsampling
                loss_func='NCC',
                lr=1,
                disp_init=None,
                iter_smooth_kernel=3,
                iter_smooth_num = 3
                ):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    mask_fixed = None
    mask_moving = None

    H, W, D = features_fix.shape[2:]
    print('ConvexAdam optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    n_ch = features_fix.shape[1]


    disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            # disp_sample = F.avg_pool3d(
            #     F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
            #     padding=1).permute(0, 2, 3, 4, 1)
            

            iter_smooth_padding = (iter_smooth_kernel - 1) // 2
            for smooth_num in range (iter_smooth_num):
                net[0].weight.data = F.avg_pool3d(net[0].weight.data, iter_smooth_kernel, stride=1, padding=iter_smooth_padding)
            disp_sample = net[0].weight
            disp_sample = disp_sample.permute(0, 2, 3, 4, 1)

            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) 
                # sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")



        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

        if selected_smooth > 1:
            kernel_smooth = selected_smooth
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements

def convex_adam_3d_catMind(feature_fixed,
                feature_moving,
                img_fixed,
                img_moving,
                mind_r=1,
                mind_d=2,
                lambda_weight=1.25,
                grid_sp=6,
                disp_hw=4,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
                loss_func='SSD',
                lr=1,
                disp_init=None):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    mask_fixed = None
    mask_moving = None

    H, W, D = features_fix.shape[2:]
    print('ConvexAdam optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    # compute features and downsample (using average pooling)
    # with torch.no_grad():
    #
    #     features_fix, features_mov = extract_features(img_fixed=features_fix,
    #                                                   img_moving=features_mov,
    #                                                   mind_r=mind_r,
    #                                                   mind_d=mind_d,
    #                                                   use_mask=use_mask,
    #                                                   mask_fixed=mask_fixed,
    #                                                   mask_moving=mask_moving)
    #     # print(features_mov.shape)
    #
    #     features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
    #     features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
    n_ch = features_fix.shape[1]

    if disp_init is None:
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)


        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(
            0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)


        else:
            disp_hr = disp_soft

    else:
        disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
                padding=1).permute(0, 2, 3, 4, 1)
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
                # sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")


        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements
def convex_adam_3d_w0(feature_fixed,
                feature_moving,
                mind_r=1,
                mind_d=2,
                lambda_weight=1.25,
                grid_sp=6,
                disp_hw=4,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
                loss_func='NCC',
                lr=1,
                disp_init=None,
                lambda_l1=1):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    #set timezone to est, new york time
    import pytz
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    new_york = pytz.timezone('America/New_York')
    now_utc = datetime.now(pytz.utc)
    now_new_york = now_utc.astimezone(new_york)
    #time should be est time zone
    writer = SummaryWriter('results/{}'.format(now_new_york.strftime("%Y%m%d-%H%M%S")))

    mask_fixed = None
    mask_moving = None

    C, H, W, D = features_fix.shape[1:]
    print('ConvexAdam optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # weight_params = torch.nn.Parameter(torch.ones(C-1, device=device, requires_grad=True)) #have to be initialized on device
    weight_params = torch.nn.Parameter(torch.ones(C-1, device=device, requires_grad=True)/C) #have to be initialized on device

    #convert lambda to tensor
    lambda_l1 = torch.tensor(lambda_l1).cuda()

    # compute features and downsample (using average pooling)
    # with torch.no_grad():
    #
    #     features_fix, features_mov = extract_features(img_fixed=features_fix,
    #                                                   img_moving=features_mov,
    #                                                   mind_r=mind_r,
    #                                                   mind_d=mind_d,
    #                                                   use_mask=use_mask,
    #                                                   mask_fixed=mask_fixed,
    #                                                   mask_moving=mask_moving)
    #     # print(features_mov.shape)
    #
    #     features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
    #     features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
    n_ch = features_fix.shape[1]

    if disp_init is None:
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)


        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(
            0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)


        else:
            disp_hr = disp_soft

    else:
        disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr':lr}, {'params': weight_params, 'lr':1E-2}])#1E-2

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
                padding=1).permute(0, 2, 3, 4, 1)
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')



            if loss_func == 'SSD':

                weight_params_apply = torch.cat( ( (1 - torch.sum(weight_params)).unsqueeze(0), weight_params ) )
                loss_l1 = lambda_l1 * torch.sum(torch.abs(weight_params_apply))#L1 regularization optmizes for sparsity

                sampled_cost_diff = (patch_mov_sampled - patch_features_fix).pow(2) * torch.abs(weight_params_apply).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 12 # Now shape is [C, 1, 1, 1]
                # print('sampled_cost_diff', sampled_cost_diff.shape, 'weight_params_apply', weight_params_apply.shape)
                # sampled_cost =  sampled_cost_diff.mean(1) * 12 #why * 12? channel size?
                sampled_cost =  sampled_cost_diff.mean(1)
                loss = sampled_cost.mean()

                writer.add_histogram('weight_params', torch.abs(weight_params_apply), iter)
                for idx, param_value in enumerate(torch.abs(weight_params_apply)):
                    writer.add_scalar(f'Parameter/{idx + 1}', param_value.item(), iter)


            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'NCC_w0':
                # print('check 1',weight_params.is_leaf)  # Should return True for an optimizable tensor
                # print('weight_params', weight_params)

                weight_params_apply = torch.cat( ( (1 - torch.sum(weight_params)).unsqueeze(0), weight_params ) )
                loss_l1 = lambda_l1 * torch.sum(torch.abs(weight_params_apply))#L1 regularization optmizes for sparsity

                # loss_l1 = lambda_l1 * torch.pow(torch.sum(torch.abs(weight_params)) -1,2) #sum-to-one constraint of absolute value

                loss = losses.GPTNCC_w0(patch_features_fix, patch_mov_sampled, weight_params_apply)

                writer.add_histogram('weight_params', torch.abs(weight_params_apply), iter)
                for idx, param_value in enumerate(torch.abs(weight_params_apply)):
                    writer.add_scalar(f'Parameter/{idx + 1}', param_value.item(), iter)

            else:
                raise NotImplementedError

            # (loss + reg_loss + loss_l1).backward()
            (loss + reg_loss).backward()
            # print('weight_params.grad', weight_params.grad)
            #get gradient of weight_params
            optimizer.step()

            print("\roptimization iteration:{} {}: {:.05} regLoss {:.04} L1 {:.04} ".format(iter, loss_func, loss.item(), reg_loss.item(), loss_l1.item()), end="")
            #print to 4 decimal places


        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)
    writer.close()

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)


    return displacements

def convex_adam_3d_MIND(feature_fixed,
                feature_moving,
                mind_r=1,
                mind_d=2,
                lambda_weight=1.25,
                grid_sp=6,
                disp_hw=4,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                use_mask=False,
                path_fixed_mask=None,
                path_moving_mask=None,
                loss_func='NCC',
                lr=1,
                disp_init=None):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    mask_fixed = None
    mask_moving = None

    H, W, D = features_fix.shape[2:]
    print('ConvexAdam optimization H, W, D', H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    print('features_fix shape', features_fix.shape)
    # compute features and downsample (using average pooling)
    with torch.no_grad():

        features_fix, features_mov = extract_features(img_fixed=features_fix[0,0,:,:,:],
                                                      img_moving=features_mov[0,0,:,:,:],
                                                      mind_r=mind_r,
                                                      mind_d=mind_d,
                                                      use_mask=use_mask,
                                                      mask_fixed=mask_fixed,
                                                      mask_moving=mask_moving)
        features_fix = features_fix.to(torch.float32)
        features_mov = features_mov.to(torch.float32)
    #     # print(features_mov.shape)
    #
    #     features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
    #     features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)
    n_ch = features_fix.shape[1]

    if disp_init is None:
        features_fix_smooth = F.avg_pool3d(features_fix, grid_sp, stride=grid_sp)
        features_mov_smooth = F.avg_pool3d(features_mov, grid_sp, stride=grid_sp)


        # compute correlation volume with SSD
        ssd, ssd_argmin = correlate(features_fix_smooth, features_mov_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

        # provide auxiliary mesh grid
        disp_mesh_t = F.affine_grid(disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                                    (1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1), align_corners=True).permute(
            0, 4, 1, 2, 3).reshape(3, -1, 1)

        # perform coupled convex optimisation
        disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        # if "ic" flag is set: make inverse consistent
        if ic:
            scale = torch.tensor([H // grid_sp - 1, W // grid_sp - 1, D // grid_sp - 1]).view(1, 3, 1, 1,
                                                                                              1).cuda().half() / 2

            ssd_, ssd_argmin_ = correlate(features_mov_smooth, features_fix_smooth, disp_hw, grid_sp, (H, W, D), n_ch)

            disp_soft_ = coupled_convex(ssd_, ssd_argmin_, disp_mesh_t, grid_sp, (H, W, D))
            disp_ice, _ = inverse_consistency((disp_soft / scale).flip(1), (disp_soft_ / scale).flip(1), iter=15)

            disp_hr = F.interpolate(disp_ice.flip(1) * scale * grid_sp, size=(H, W, D), mode='trilinear',
                                    align_corners=False)


        else:
            disp_hr = disp_soft

    else:
        disp_hr = torch.from_numpy(disp_init).float().cuda()


    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
                padding=1).permute(0, 2, 3, 4, 1)
            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1) * 12
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")


        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam, size=(H, W, D), mode='trilinear', align_corners=False)

        if selected_smooth == 5:
            kernel_smooth = 5
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

        if selected_smooth == 3:
            kernel_smooth = 3
            padding_smooth = kernel_smooth // 2
            disp_hr = F.avg_pool3d(
                F.avg_pool3d(F.avg_pool3d(disp_hr, kernel_smooth, padding=padding_smooth, stride=1), kernel_smooth,
                             padding=padding_smooth, stride=1), kernel_smooth, padding=padding_smooth, stride=1)

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy()
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy()
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy()
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements

def translate_adam(feature_fixed,
                feature_moving,
                lambda_weight=1.25,
                selected_niter=80,
                selected_smooth=0,
                grid_sp_adam=2,
                ic=True,
                loss_func='MSE',
                lr=1,
                iter_smooth_kernel=3,
                iter_smooth_num = 3,
                end_smooth_kernel = 3,
                final_upsample = 2,
                disp_init=None):

    feature_fixed = torch.from_numpy(feature_fixed).float()
    feature_moving = torch.from_numpy(feature_moving).float()

    features_fix = einops.repeat(feature_fixed, 'i j k c-> b c i j k', b=1)
    features_mov = einops.repeat(feature_moving, 'i j k c-> b c i j k', b=1)


    features_fix = features_fix.cuda()
    features_mov = features_mov.cuda()

    C, H, W, D = features_fix.shape[1:]
    print('ConvexAdam optimization C, H, W, D', C, H, W, D)

    torch.cuda.synchronize()
    t0 = time.time()

    coarse_factor = 4
    coarse_features_fix = F.avg_pool3d(features_fix, 4, stride=4)
    coarse_features_mov = F.avg_pool3d(features_mov, 4, stride=4)    

    regspace_fix = torch.zeros(C, H//coarse_factor+20, W//coarse_factor+20, D//coarse_factor+20).cuda()
    regspace_mov = torch.zeros(C, H//coarse_factor+20, W//coarse_factor+20, D//coarse_factor+20).cuda()
    regspace_fix[:, 10:-10, 10:-10, 10:-10] = coarse_features_fix
    regspace_mov[:, 10:-10, 10:-10, 10:-10] = coarse_features_mov
    #create a foreground mask, threshold at 0.1. first get channel mean of absolute value
    channel_mean_fix = torch.mean(torch.abs(regspace_fix), dim=(0))
    channel_mean_mov = torch.mean(torch.abs(regspace_mov), dim=(0))
    regspace_mask_fix = (channel_mean_fix > 0.1).float()
    regspace_mask_mov = (channel_mean_mov > 0.1).float()
    regspace_mask_fix = regspace_mask_fix.unsqueeze(0)
    regspace_mask_mov = regspace_mask_mov.unsqueeze(0)



    #serach for the best translation

    def apply_shift_along_axis(tensor, shift_amount, axis):
        """
        Shifts a tensor along a given axis using NumPy indexing.
        The shifting will be padded with zeros.
        """
        tensor_np = tensor.cpu().numpy()  # Convert to numpy for easy manipulation
        shifted_tensor = np.roll(tensor_np, shift_amount, axis=axis)
        
        # Set the shifted-in elements to zero
        # if shift_amount > 0:
        #     slices = [slice(None)] * tensor_np.ndim
        #     slices[axis] = slice(0, shift_amount)
        #     shifted_tensor[tuple(slices)] = 0
        # elif shift_amount < 0:
        #     slices = [slice(None)] * tensor_np.ndim
        #     slices[axis] = slice(shift_amount, None)
        #     shifted_tensor[tuple(slices)] = 0

        return torch.from_numpy(shifted_tensor).cuda()

    # Initialize the search space
    best_shifts = [0, 0, 0]  # To store the best shifts for each axis
    best_mse = float('inf')

    # Define the translation search range (adjust as needed)
    shift_range = np.arange(-10, 11)

    for dim_id in range(3):
        # Start by searching for the best shift along dim 1 (H axis)
        for shift_amount in shift_range:
            shifted_regspace_mov = apply_shift_along_axis(regspace_mov, shift_amount, axis=dim_id+1)  # Shift along height (H)
            shifted_regspace_mask_mov = apply_shift_along_axis(regspace_mask_mov, shift_amount, axis=dim_id+1)  # Shift along height (H)

            # Calculate the MSE loss between the fixed and shifted moving images
            # shared_mask = regspace_mask_fix * shifted_regspace_mask_mov
            # mse_loss = F.mse_loss(regspace_fix*shared_mask, shifted_regspace_mov*shared_mask)
            mse_loss = F.mse_loss(regspace_fix, shifted_regspace_mov)

            print('dim_id', dim_id, 'shift_amount', shift_amount, 'mse_loss', mse_loss.item())

            if mse_loss.item() < best_mse:
                best_mse = mse_loss.item()
                best_shifts[dim_id] = shift_amount

    # Apply the best shifts to the moving image
    best_shifts = np.asarray(best_shifts) * coarse_factor * final_upsample * (-1)
    for dim_id, shift_amount in enumerate(best_shifts):
        # print('dim_id', dim_id, 'shift_amount', shift_amount)
        features_mov = apply_shift_along_axis(features_mov, shift_amount, axis=dim_id+2)
    print('best translation shifts', best_shifts)
    n_ch = C

    disp_hr = torch.from_numpy(disp_init).float().cuda()

    # run Adam instance optimisation
    if lambda_weight > 0:
        with torch.no_grad():

            patch_features_fix = F.avg_pool3d(features_fix, grid_sp_adam, stride=grid_sp_adam)
            patch_features_mov = F.avg_pool3d(features_mov, grid_sp_adam, stride=grid_sp_adam)

        # create optimisable displacement grid
        disp_lr = F.interpolate(disp_hr, size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                                mode='trilinear', align_corners=False)

        net = nn.Sequential(nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False))
        net[0].weight.data[:] = disp_lr.float().cpu().data / grid_sp_adam
        net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(),
                              (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), align_corners=False)

        # run Adam optimisation with diffusion regularisation and B-spline smoothing
        for iter in range(selected_niter):
            optimizer.zero_grad()

            # disp_sample = F.avg_pool3d(
            #     F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1), 3, stride=1,
            #     padding=1).permute(0, 2, 3, 4, 1)
            

            iter_smooth_padding = (iter_smooth_kernel -1) // 2

            disp_sample = net[0].weight
            for smooth_num in range (iter_smooth_num):
                disp_sample = F.avg_pool3d(disp_sample, iter_smooth_kernel, stride=1, padding=iter_smooth_padding)
            disp_sample = disp_sample.permute(0, 2, 3, 4, 1)

            reg_loss = lambda_weight * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()

            scale = torch.tensor([(H // grid_sp_adam - 1) / 2, (W // grid_sp_adam - 1) / 2,
                                  (D // grid_sp_adam - 1) / 2]).cuda().unsqueeze(0)
            grid_disp = grid0.view(-1, 3).cuda().float() + ((disp_sample.view(-1, 3)) / scale).flip(1).float()

            patch_mov_sampled = F.grid_sample(patch_features_mov.float(),
                                              grid_disp.view(1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam,
                                                             3).cuda(), align_corners=False, mode='bilinear')

            if loss_func == 'SSD':
                sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                # sampled_cost = (patch_mov_sampled - patch_features_fix).pow(2).mean(1)
                loss = sampled_cost.mean()

            elif loss_func == 'NCC':
                #repeat NCC calculation over all channels and mean
                loss = 0
                for i in range(n_ch):
                    # loss += losses.NCC(108).loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                    loss += losses.NCC().loss(patch_mov_sampled[:,i:i+1,:,:,:], patch_features_fix[:,i:i+1,:,:,:])
                loss /= n_ch
            elif loss_func == 'MI':
                loss = 0
                for i in range(n_ch):

                    loss += losses.MutualInformationLoss().loss(patch_mov_sampled[:,i,:,:,:], patch_features_fix[:,i,:,:,:])
                loss /= n_ch
            elif loss_func == 'nccTrue':
                # loss = losses.nccTrue().loss(patch_features_fix, patch_mov_sampled)
                loss = losses.NCC_neighbor().loss(patch_features_fix, patch_mov_sampled)
            elif loss_func == 'GPTNCC':
                loss = losses.GPTNCC(patch_features_fix, patch_mov_sampled)
            else:
                raise NotImplementedError

            (loss + reg_loss).backward()
            optimizer.step()

            print("\roptimization iteration:{} {}: {} regLoss {}".format(iter, loss_func, loss.item(), reg_loss.item()), end="")



        fitted_grid = disp_sample.detach().permute(0, 4, 1, 2, 3)
        disp_hr = F.interpolate(fitted_grid * grid_sp_adam * final_upsample, size=(H*final_upsample, W*final_upsample, D*final_upsample), 
                                mode='trilinear', align_corners=False)

        #keep same smoothing when outputing result
        if end_smooth_kernel > 1:
            disp_hr = F.avg_pool3d(disp_hr, end_smooth_kernel, stride=1, padding=end_smooth_padding)
        
        

    torch.cuda.synchronize()
    t1 = time.time()
    case_time = t1 - t0
    print('case time: ', case_time)

    x = disp_hr[0, 0, :, :, :].cpu().half().data.numpy() + best_shifts[0]
    y = disp_hr[0, 1, :, :, :].cpu().half().data.numpy() + best_shifts[1]
    z = disp_hr[0, 2, :, :, :].cpu().half().data.numpy() + best_shifts[2]
    displacements = np.stack((x, y, z), 3).astype(float)

    return displacements

# extract MIND and/or semantic nnUNet features
def extract_features(img_fixed,
                     img_moving,
                     mind_r,
                     mind_d,
                     use_mask,
                     mask_fixed,
                     mask_moving):
    # MIND features
    if use_mask:
        H, W, D = img_fixed.shape[-3:]

        # replicate masking
        avg3 = nn.Sequential(nn.ReplicationPad3d(1), nn.AvgPool3d(3, stride=1))
        avg3.cuda()

        mask = (avg3(mask_fixed.view(1, 1, H, W, D).cuda()) > 0.9).float()
        _, idx = edt((mask[0, 0, ::2, ::2, ::2] == 0).squeeze().cpu().numpy(), return_indices=True)
        fixed_r = F.interpolate((img_fixed[::2, ::2, ::2].cuda().reshape(-1)[
            idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]]).unsqueeze(0).unsqueeze(0), scale_factor=2,
                                mode='trilinear')
        fixed_r.view(-1)[mask.view(-1) != 0] = img_fixed.cuda().reshape(-1)[mask.view(-1) != 0]

        mask = (avg3(mask_moving.view(1, 1, H, W, D).cuda()) > 0.9).float()
        _, idx = edt((mask[0, 0, ::2, ::2, ::2] == 0).squeeze().cpu().numpy(), return_indices=True)
        moving_r = F.interpolate((img_moving[::2, ::2, ::2].cuda().reshape(-1)[
            idx[0] * D // 2 * W // 2 + idx[1] * D // 2 + idx[2]]).unsqueeze(0).unsqueeze(0), scale_factor=2,
                                 mode='trilinear')
        moving_r.view(-1)[mask.view(-1) != 0] = img_moving.cuda().reshape(-1)[mask.view(-1) != 0]

        features_fix = MINDSSC(fixed_r.cuda(), mind_r, mind_d).half()
        features_mov = MINDSSC(moving_r.cuda(), mind_r, mind_d).half()
    else:
        img_fixed = img_fixed.unsqueeze(0).unsqueeze(0)
        img_moving = img_moving.unsqueeze(0).unsqueeze(0)
        features_fix = MINDSSC(img_fixed.cuda(), mind_r, mind_d).half()
        features_mov = MINDSSC(img_moving.cuda(), mind_r, mind_d).half()

    return features_fix, features_mov


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-f","--path_img_fixed", type=str, required=True)
    # parser.add_argument("-f","--path_img_fixed", type=str, default='/fast/songx/datasets/AbdomenMRCT/imagesTs/AbdomenMRCT_0009_0000.nii.gz')
    parser.add_argument("-f", "--path_feature_fixed", type=str, default='/fast/songx/tempFiles/DINO_2D/lung/0000_features_imgShape_24dim.npy')

    # parser.add_argument("-m",'--path_img_moving', type=str, required=True)
    # parser.add_argument("-m",'--path_img_moving', type=str, default='/fast/songx/datasets/AbdomenMRCT/imagesTs/AbdomenMRCT_0009_0001.nii.gz')
    parser.add_argument("-m", '--path_feature_moving', type=str, default='/fast/songx/tempFiles/DINO_2D/lung/0001_features_imgShape_24dim.npy')
    parser.add_argument('--mind_r', type=int, default=1)
    parser.add_argument('--mind_d', type=int, default=2)
    parser.add_argument('--lambda_weight', type=float, default=1.25)
    parser.add_argument('--grid_sp', type=int, default=6)
    parser.add_argument('--disp_hw', type=int, default=4)
    # parser.add_argument('--selected_niter', type=int, default=80)
    parser.add_argument('--selected_niter', type=int, default=80)
    parser.add_argument('--selected_smooth', type=int, default=0)
    parser.add_argument('--grid_sp_adam', type=int, default=2)
    parser.add_argument('--ic', choices=('True', 'False'), default='True')
    parser.add_argument('--use_mask', choices=('True', 'False'), default='False')
    parser.add_argument('--path_mask_fixed', type=str, default=None)
    parser.add_argument('--path_mask_moving', type=str, default=None)
    parser.add_argument('--result_path', type=str, default='/fast/songx/tempFiles/DINO_2D/lung')
    parser.add_argument('--loss_func', type=str, default='SSD',choices=('SSD', 'NCC', 'MI'))
    parser.add_argument('--lr', type=float, default=1)


    args = parser.parse_args()

    if args.ic == 'True':
        ic = True
    else:
        ic = False

    if args.use_mask == 'True':
        use_mask = True
    else:
        use_mask = False

    convex_adam_3d(args.path_feature_fixed,
                args.path_feature_moving,
                args.mind_r,
                args.mind_d,
                args.lambda_weight,
                args.grid_sp,
                args.disp_hw,
                args.selected_niter,
                args.selected_smooth,
                args.grid_sp_adam,
                ic,
                use_mask,
                args.path_mask_fixed,
                args.path_mask_moving,
                args.result_path,
                args.loss_func,
                   args.lr)
