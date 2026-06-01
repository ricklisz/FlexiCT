# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import torch
import numpy as np
from torch import nn
from torchvision import transforms
import monai
import torch.nn.functional as F
from monai.transforms import Transform
from monai.utils import convert_to_tensor
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = transforms.Resize(
                gram_teacher_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = transforms.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = transforms.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = transforms.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = transforms.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output

class SimpleDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        use_intensity_transforms = False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        self.use_intensity_transforms = use_intensity_transforms
        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info(f"Use intensity transforms: {use_intensity_transforms}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = transforms.Resize(
                gram_teacher_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        # global_transfo1_extra = GaussianBlur(p=1.0)

        # global_transfo2_extra = transforms.Compose(
        #     [
        #         GaussianBlur(p=0.1)
        #     ]
        # )
        if self.use_intensity_transforms:
            self.color_jitter = transforms.Compose(
                [
                    monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
                    monai.transforms.RandGaussianSmooth(
                        sigma_x=(0.5, 1.0),sigma_y=(0.5, 1.0),sigma_z=(0.5, 1.0),prob=0.2,),
                    monai.transforms.RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
                    monai.transforms.RandSimulateLowResolution(prob=0.25,zoom_range=(0.5, 1.0)),
                    monai.transforms.RandAdjustContrast(prob=0.1, gamma=(0.7, 1.5)),
                    
                ]
            )
        global_transfo1_extra = nn.Identity()
        global_transfo2_extra = nn.Identity()
        local_transfo_extra = nn.Identity()

        self.global_transfo1 = transforms.Compose([resize_global, global_transfo1_extra])
        self.global_transfo2 = transforms.Compose([resize_global, global_transfo2_extra])
        self.local_transfo = transforms.Compose([local_transfo_extra])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs
        if self.use_intensity_transforms:
            image = self.color_jitter(image).as_tensor()
        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                im1_base,
                im2_base,
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.resize_gram_teacher(im1_base)
                gram_crop_2 = self.resize_gram_teacher(im2_base)
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output

class NoDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = transforms.Resize(
                gram_teacher_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        local_transfo_extra = nn.Identity()

        if self.share_color_jitter:
            self.global_transfo1 = transforms.Compose([resize_global])
            self.global_transfo2 = transforms.Compose([resize_global])
            self.local_transfo = transforms.Compose([local_transfo_extra])
        else:
            self.global_transfo1 = transforms.Compose(
                [resize_global]
            )
            self.global_transfo2 = transforms.Compose(
                [resize_global]
            )
            self.local_transfo = transforms.Compose([local_transfo_extra])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                im1_base,
                im2_base,
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.resize_gram_teacher(im1_base)
                gram_crop_2 = self.resize_gram_teacher(im2_base)
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output

class TwoViewRandomCrop3D(object):
    """
    Randomly crops two 3D views from an image ensuring an overlap of no less
    than a specified fraction.

    The first view is cropped at a random location.
    The second view is derived by applying a random offset (within a
    computed limit) to the first crop, ensuring the overlap is at least
    `min_overlap` in volume.

    Args:
        roi_size (tuple): The size (D, H, W) of the crop.
        min_overlap (float): The minimum overall fraction of overlap between the
                             two crops. Default is 0.65.
                             This is enforced by ensuring that each dimension
                             overlaps by at least (min_overlap)^(1/3).
    """
    def __init__(self, roi_size, min_overlap=0.65):
        self.roi_size = tuple(roi_size)
        self.min_overlap = float(min_overlap)
        # For 3D, if each dimension overlaps by at least factor,
        # then overall overlap is factor^3.
        self.factor = self.min_overlap ** (1.0 / 3.0)

    def __call__(self, img: torch.Tensor):
        # Assume img is a torch tensor of shape (C, D, H, W) or (D, H, W)
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        roi_d, roi_h, roi_w = self.roi_size

        if D < roi_d or H < roi_h or W < roi_w:
            raise ValueError("Image size is smaller than the ROI size.")

        device = img.device

        # Randomly select the starting coordinates for the first crop
        d1 = torch.randint(0, D - roi_d + 1, (1,), device=device).item()
        h1 = torch.randint(0, H - roi_h + 1, (1,), device=device).item()
        w1 = torch.randint(0, W - roi_w + 1, (1,), device=device).item()

        crop1 = img[..., d1:d1 + roi_d, h1:h1 + roi_h, w1:w1 + roi_w]

        # Max allowed offset per dimension to keep overlap >= factor
        max_offset_d = int(roi_d * (1.0 - self.factor))
        max_offset_h = int(roi_h * (1.0 - self.factor))
        max_offset_w = int(roi_w * (1.0 - self.factor))

        def rand_offset(m):
            if m <= 0:
                return 0
            # randint is [low, high), we want integers in [-m, m]
            return torch.randint(-m, m + 1, (1,), device=device).item()

        offset_d = rand_offset(max_offset_d)
        offset_h = rand_offset(max_offset_h)
        offset_w = rand_offset(max_offset_w)

        # Second crop start (clamped to image bounds)
        d2 = max(0, min(d1 + offset_d, D - roi_d))
        h2 = max(0, min(h1 + offset_h, H - roi_h))
        w2 = max(0, min(w1 + offset_w, W - roi_w))

        crop2 = img[..., d2:d2 + roi_d, h2:h2 + roi_h, w2:w2 + roi_w]

        return {"view1": crop1, "view2": crop2}


class SimpleDataAugmentationDINO3D(object):
    def __init__(
        self,
        global_crop_overlap,
        local_crops_number,
        global_crops_size=(128, 128, 128),
        local_crops_size=(64, 64, 64),
        gram_teacher_crops_size=None,
        use_intensity_transforms = False,
    ):
        self.global_crop_overlap = global_crop_overlap
        self.local_crops_number = local_crops_number
        self.global_crops_size = tuple(int(s) for s in global_crops_size)
        self.local_crops_size = tuple(int(s) for s in local_crops_size)
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.use_intensity_transforms = use_intensity_transforms

        if gram_teacher_crops_size is not None:
            self.gram_teacher_crops_size = tuple(int(s) for s in gram_teacher_crops_size)
            self.global_crop_max_size = tuple(
                max(gs, gts)
                for gs, gts in zip(self.global_crops_size, self.gram_teacher_crops_size)
            )
        else:
            self.gram_teacher_crops_size = None
            self.global_crop_max_size = self.global_crops_size

        # Global augmentation transforms using MONAI
        if self.use_intensity_transforms:
            self.color_jitter = monai.transforms.Compose([
                monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
                # monai.transforms.RandGaussianSmooth(sigma_x=(0.5, 1.0),sigma_y=(0.5, 1.0),sigma_z=(0.5, 1.0),prob=0.2,),
                monai.transforms.RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
                # monai.transforms.RandSimulateLowResolution(prob=0.25,zoom_range=(0.5, 1.0)),
                monai.transforms.RandAdjustContrast(prob=0.1, gamma=(0.7, 1.5)),
            ])

        self.geometric_augmentation_global = monai.transforms.Compose([
            TwoViewRandomCrop3D(roi_size=self.global_crop_max_size, min_overlap=global_crop_overlap),
        ])
        self.global_flip =   monai.transforms.Compose([
            monai.transforms.RandAxisFlip(prob=0.5)
        ])
        # Local augmentation transforms using MONAI (wrapped with torchvision Compose for consistency)
        self.geometric_augmentation_local = monai.transforms.Compose([
            monai.transforms.RandSpatialCrop(roi_size=self.local_crops_size),
            monai.transforms.RandAxisFlip(prob=0.5),
        ])

    def resize_3d_linear(self, x, size):
        """
        x: (C, D, H, W)
        out_size: (D, H, W)
        """
        if x.ndim == 4:
            x_in = x.unsqueeze(0)   # (1, C, D, H, W)
            x = F.interpolate(
                x_in,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
            return x.squeeze(0)
        else:
            return  F.interpolate(
                x,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
        
    def pad_min(self, img: torch.Tensor, target_size):
        """
        Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
        value in the tensor. Padding is added only on the *right side*
        (no centering, no symmetry).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        td, th, tw = target_size

        pad_d = max(0, td - D)
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)

        # split padding on both sides (left/right) to center the content
        pad_d_left  = pad_d // 2
        pad_d_right = pad_d - pad_d_left

        pad_h_left  = pad_h // 2
        pad_h_right = pad_h - pad_h_left

        pad_w_left  = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # F.pad expects: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)

        pad_value = img.amin().item()  # item() gives python scalar; safe for F.pad
        return F.pad(img, pad, mode="constant", value=pad_value)
    
    def __call__(self, sample):
        # Expecting sample to a tensor
        output = {}
        # Sample shape, 1,C,D,H,W
        # Global crops: generate two augmented versions
        sample = self.pad_min(sample, self.global_crop_max_size)
        if self.use_intensity_transforms:
            # sample = self.color_jitter(sample).as_tensor()
            sample = convert_to_tensor(self.color_jitter(sample), track_meta=False)
        global_crops = self.geometric_augmentation_global(sample)
        # g1_transf = self.global_flip(global_crops['view1']).as_tensor()
        g1_transf = convert_to_tensor(self.global_flip(global_crops['view1']), track_meta=False)
        g2_transf = global_crops['view2']
        global_crop_1 = self.resize_3d_linear(g1_transf, self.global_crops_size)
        global_crop_2 = self.resize_3d_linear(g2_transf, self.global_crops_size)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [g1_transf, g2_transf]

        if self.gram_teacher_crops_size is not None:
            gram_crop_1 = self.resize_3d_linear(g1_transf, self.gram_teacher_crops_size)
            gram_crop_2 = self.resize_3d_linear(g2_transf, self.gram_teacher_crops_size)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # Local crops: generate a list of local augmentations
        local_crops = []
        for _ in range(self.local_crops_number):
            # aug_local = self.geometric_augmentation_local(global_crop_2).as_tensor()
            aug_local = convert_to_tensor(self.geometric_augmentation_local(global_crop_2), track_meta=False)
            local_crops.append(aug_local)
        output["local_crops"] = local_crops
        # Optionally add other augmentation information (e.g., offsets)
        output["offsets"] = ()  
        return output
    
class RandScaleDataAugmentationDINO3D(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=(128, 128, 128),
        local_crops_size=(64, 64, 64),
        gram_teacher_crops_size=None,
        use_intensity_transforms = False,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = tuple(int(s) for s in global_crops_size)
        self.local_crops_size = tuple(int(s) for s in local_crops_size)
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.use_intensity_transforms = use_intensity_transforms
        if gram_teacher_crops_size is not None:
            self.gram_teacher_crops_size = tuple(int(s) for s in gram_teacher_crops_size)
            self.global_crop_max_size = tuple(
                max(gs, gts)
                for gs, gts in zip(self.global_crops_size, self.gram_teacher_crops_size)
            )
        else:
            self.gram_teacher_crops_size = None
            self.global_crop_max_size = self.global_crops_size
            
        if self.use_intensity_transforms:
            self.color_jitter = monai.transforms.Compose([
                monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
                monai.transforms.RandGaussianSmooth(sigma_x=(0.5, 1.0),sigma_y=(0.5, 1.0),sigma_z=(0.5, 1.0),prob=0.2,),
                monai.transforms.RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
                # monai.transforms.RandSimulateLowResolution(prob=0.25,zoom_range=(0.5, 1.0)),
                monai.transforms.RandAdjustContrast(prob=0.1, gamma=(0.7, 1.5)),
            ])
            
        # Global augmentation transforms using MONAI
        self.geometric_augmentation_global = monai.transforms.Compose([
            monai.transforms.RandScaleCrop(roi_scale= self.global_crops_scale[0], 
                                                    max_roi_scale=self.global_crops_scale[1],
                                                    random_center=True,
                                                    random_size=True,
        )])
                                                                  
        self.global_flip =   monai.transforms.Compose([
            monai.transforms.RandAxisFlip(prob=0.5)
            
        ])
        # Local augmentation transforms using MONAI (wrapped with torchvision Compose for consistency)
        self.geometric_augmentation_local = monai.transforms.Compose([
            monai.transforms.RandScaleCrop(roi_scale=local_crops_scale[0], 
                                                    max_roi_scale=local_crops_scale[1],
                                                    random_center=True,
                                                    random_size=True,),
            monai.transforms.RandAxisFlip(prob=0.5),
        ])

    def resize_3d_linear(self, x, size):
        """
        x: (C, D, H, W)
        out_size: (D, H, W)
        """
        if x.ndim == 4:
            x_in = x.unsqueeze(0)   # (1, C, D, H, W)
            x = F.interpolate(
                x_in,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
            return x.squeeze(0)
        else:
            return  F.interpolate(
                x,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
        
    # def pad_min(self, img: torch.Tensor, target_size):
    #     """
    #     Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
    #     value in the tensor. Padding is added only on the *right side*
    #     (no centering, no symmetry).
    #     """
    #     if not isinstance(img, torch.Tensor):
    #         img = torch.as_tensor(img)
    #     D, H, W = img.shape[-3:]
    #     td, th, tw = target_size
    #     pad_d = max(0, td - D)
    #     pad_h = max(0, th - H)
    #     pad_w = max(0, tw - W)
    #     # F.pad order: (w_left, w_right, h_left, h_right, d_left, d_right)
    #     pad = (0, pad_w, 0, pad_h, 0, pad_d)
    #     pad_value = img.min()
    #     img = F.pad(img, pad, mode="constant", value=pad_value)
    #     return img

    def pad_min(self, img: torch.Tensor, target_size):
        """
        Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
        value in the tensor. Padding is added only on the *right side*
        (no centering, no symmetry).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        td, th, tw = target_size

        pad_d = max(0, td - D)
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)

        # split padding on both sides (left/right) to center the content
        pad_d_left  = pad_d // 2
        pad_d_right = pad_d - pad_d_left

        pad_h_left  = pad_h // 2
        pad_h_right = pad_h - pad_h_left

        pad_w_left  = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # F.pad expects: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)

        pad_value = img.amin().item()  # item() gives python scalar; safe for F.pad
        return F.pad(img, pad, mode="constant", value=pad_value)

    def __call__(self, sample):
        # Expecting sample to a tensor
        output = {}
        # Sample shape, 1,C,D,H,W
        # Global crops: generate two augmented versions
        sample = self.pad_min(sample, self.global_crop_max_size)
        if self.use_intensity_transforms:
            sample = convert_to_tensor(self.color_jitter(sample), track_meta=False)
        global_crops = self.geometric_augmentation_global(sample)
        g1_transf = convert_to_tensor(self.global_flip(global_crops), track_meta=False)
        g2_transf = convert_to_tensor(self.geometric_augmentation_global(sample), track_meta=False)
        global_crop_1 = self.resize_3d_linear(g1_transf, self.global_crops_size)
        global_crop_2 = self.resize_3d_linear(g2_transf, self.global_crops_size)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [g1_transf, g2_transf]

        if self.gram_teacher_crops_size is not None:
            gram_crop_1 = self.resize_3d_linear(g1_transf, self.gram_teacher_crops_size)
            gram_crop_2 = self.resize_3d_linear(g2_transf, self.gram_teacher_crops_size)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # Local crops: generate a list of local augmentations
        local_crops = []
        for _ in range(self.local_crops_number):
            aug_local = convert_to_tensor(self.geometric_augmentation_local(global_crop_2), track_meta=False)
            local_crops.append(aug_local)
        output["local_crops"] = local_crops
        # Optionally add other augmentation information (e.g., offsets)
        output["offsets"] = ()  
        return output
    
class RandomCropDINO3D(object):
    def __init__(
        self,
        local_views_scale = (0.1875, 0.5),
        local_crops_number = 8,
        global_crops_size=(128, 128, 128),
        local_crops_size=(64, 64, 64),
        gram_teacher_crops_size=None,
        use_intensity_transforms = False,
    ):
        self.local_views_scale = local_views_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = tuple(int(s) for s in global_crops_size)
        self.local_crops_size = tuple(int(s) for s in local_crops_size)
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.use_intensity_transforms = use_intensity_transforms

        if gram_teacher_crops_size is not None:
            self.gram_teacher_crops_size = tuple(int(s) for s in gram_teacher_crops_size)
            self.global_crop_max_size = tuple(
                max(gs, gts)
                for gs, gts in zip(self.global_crops_size, self.gram_teacher_crops_size)
            )
        else:
            self.gram_teacher_crops_size = None
            self.global_crop_max_size = self.global_crops_size

        if self.use_intensity_transforms:
            self.color_jitter = monai.transforms.Compose([
                monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
                monai.transforms.RandGaussianSmooth(sigma_x=(0.5, 1.0),sigma_y=(0.5, 1.0),sigma_z=(0.5, 1.0),prob=0.2,),
                monai.transforms.RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
                monai.transforms.RandSimulateLowResolution(prob=0.25,zoom_range=(0.5, 1.0)),
                monai.transforms.RandAdjustContrast(prob=0.1, gamma=(0.7, 1.5)),
            ])
            
        self.geometric_augmentation_global = monai.transforms.RandSpatialCropSamples(
                            roi_size=tuple(int(local_views_scale[1] * sz) for sz in self.global_crop_max_size),
                            num_samples=1,
                            max_roi_size=self.global_crop_max_size,
                            random_center=True,
                            random_size=True,
                        )

        self.global_flip =   monai.transforms.Compose([
            monai.transforms.RandAxisFlip(prob=0.5)
            
        ])
        # Local augmentation transforms using MONAI (wrapped with torchvision Compose for consistency)
        self.geometric_augmentation_local = monai.transforms.Compose([
            monai.transforms.RandSpatialCropSamples(
                roi_size=tuple(int(self.local_views_scale[0] * sz) for sz in local_crops_size),
                num_samples=1,
                max_roi_size=tuple(int(self.local_views_scale[1] * sz) for sz in local_crops_size),
                random_center=True,
                random_size=True,
            ),
            monai.transforms.RandAxisFlip(prob=0.5)
        ])

    def resize_3d_linear(self, x, size):
        """
        x: (C, D, H, W)
        out_size: (D, H, W)
        """
        if x.ndim == 4:
            x_in = x.unsqueeze(0)   # (1, C, D, H, W)
            x = F.interpolate(
                x_in,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
            return x.squeeze(0)
        else:
            return  F.interpolate(
                x,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
        
    def pad_min(self, img: torch.Tensor, target_size):
        """
        Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
        value in the tensor. Padding is added only on the *right side*
        (no centering, no symmetry).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        td, th, tw = target_size

        pad_d = max(0, td - D)
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)

        # split padding on both sides (left/right) to center the content
        pad_d_left  = pad_d // 2
        pad_d_right = pad_d - pad_d_left

        pad_h_left  = pad_h // 2
        pad_h_right = pad_h - pad_h_left

        pad_w_left  = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # F.pad expects: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)

        pad_value = img.amin().item()  # item() gives python scalar; safe for F.pad
        return F.pad(img, pad, mode="constant", value=pad_value)

    def __call__(self, sample):
        # Expecting sample to a tensor
        output = {}
        # Sample shape, 1,C,D,H,W
        # Global crops: generate two augmented versions
        sample = self.pad_min(sample, self.global_crop_max_size)
        if self.use_intensity_transforms:
            sample = convert_to_tensor(self.color_jitter(sample), track_meta=False)
        global_crops1 = self.geometric_augmentation_global(sample)[0]
        global_crops2 = self.geometric_augmentation_global(sample)[0]
        g1_transf = convert_to_tensor(self.global_flip(global_crops1), track_meta=False)
        g2_transf = convert_to_tensor(global_crops2, track_meta=False)
        global_crop_1 = self.resize_3d_linear(g1_transf, self.global_crops_size)
        global_crop_2 = self.resize_3d_linear(g2_transf, self.global_crops_size)

        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [g1_transf, g2_transf]

        if self.gram_teacher_crops_size is not None:
            gram_crop_1 = self.resize_3d_linear(g1_transf, self.gram_teacher_crops_size)
            gram_crop_2 = self.resize_3d_linear(g2_transf, self.gram_teacher_crops_size)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # Local crops: generate a list of local augmentations
        local_crops = []
        for _ in range(self.local_crops_number):
            aug_local = convert_to_tensor(self.geometric_augmentation_local(sample)[0], track_meta=False)
            aug_local = self.resize_3d_linear(aug_local, self.local_crops_size)
            local_crops.append(aug_local)
        output["local_crops"] = local_crops
        # Optionally add other augmentation information (e.g., offsets)
        output["offsets"] = ()  
        return output

class RandomCropTIPS3D(object):
    def __init__(
        self,
        local_views_scale = (0.1875, 0.5),
        local_crops_number = 8,
        global_crops_size=(128, 128, 128),
        local_crops_size=(64, 64, 64),
        gram_teacher_crops_size=None,
        use_intensity_transforms = False,
    ):
        self.local_views_scale = local_views_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = tuple(int(s) for s in global_crops_size)
        self.local_crops_size = tuple(int(s) for s in local_crops_size)
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.use_intensity_transforms = use_intensity_transforms

        if gram_teacher_crops_size is not None:
            self.gram_teacher_crops_size = tuple(int(s) for s in gram_teacher_crops_size)
            self.global_crop_max_size = tuple(
                max(gs, gts)
                for gs, gts in zip(self.global_crops_size, self.gram_teacher_crops_size)
            )
        else:
            self.gram_teacher_crops_size = None
            self.global_crop_max_size = self.global_crops_size

        if self.use_intensity_transforms:
            self.color_jitter = monai.transforms.Compose([
                monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
                # monai.transforms.RandGaussianSmooth(sigma_x=(0.5, 1.0),sigma_y=(0.5, 1.0),sigma_z=(0.5, 1.0),prob=0.2,),
                monai.transforms.RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
                # monai.transforms.RandSimulateLowResolution(prob=0.25,zoom_range=(0.5, 1.0)),
                monai.transforms.RandAdjustContrast(prob=0.1, gamma=(0.7, 1.5)),
            ])
            
        self.geometric_augmentation_global = monai.transforms.CenterSpatialCrop(
                            roi_size=tuple(sz for sz in self.global_crop_max_size),
                        )
        # Local augmentation transforms using MONAI (wrapped with torchvision Compose for consistency)
        self.geometric_augmentation_local = monai.transforms.Compose([
            monai.transforms.RandSpatialCropSamples(
                roi_size=tuple(int(self.local_views_scale[0] * sz) for sz in local_crops_size),
                num_samples=1,
                max_roi_size=tuple(int(self.local_views_scale[1] * sz) for sz in local_crops_size),
                random_center=True,
                random_size=True,
            )
        ])

    def resize_3d_linear(self, x, size):
        """
        x: (C, D, H, W)
        out_size: (D, H, W)
        """
        if x.ndim == 4:
            x_in = x.unsqueeze(0)   # (1, C, D, H, W)
            x = F.interpolate(
                x_in,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
            return x.squeeze(0)
        else:
            return  F.interpolate(
                x,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
        
    def pad_min(self, img: torch.Tensor, target_size):
        """
        Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
        value in the tensor. Padding is added only on the *right side*
        (no centering, no symmetry).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        td, th, tw = target_size

        pad_d = max(0, td - D)
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)

        # split padding on both sides (left/right) to center the content
        pad_d_left  = pad_d // 2
        pad_d_right = pad_d - pad_d_left

        pad_h_left  = pad_h // 2
        pad_h_right = pad_h - pad_h_left

        pad_w_left  = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # F.pad expects: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)

        pad_value = img.amin().item()  # item() gives python scalar; safe for F.pad
        return F.pad(img, pad, mode="constant", value=pad_value)

    def __call__(self, sample):
        # Expecting sample to a dict
        image = sample["image"]
        caption = sample["caption"]
        output = {}
        # Sample shape, 1,C,D,H,W
        # Global crops: generate two augmented versions
        image = self.pad_min(image, self.global_crop_max_size)
        if self.use_intensity_transforms:
            image = convert_to_tensor(self.color_jitter(image), track_meta=False)
        global_crops1 = self.geometric_augmentation_global(image)
        g1_transf = convert_to_tensor(global_crops1, track_meta=False)
        global_crop_1 = self.resize_3d_linear(g1_transf, self.global_crops_size)
        
        output["global_crops"] = [global_crop_1]
        output["global_crops_teacher"] = [g1_transf]

        if self.gram_teacher_crops_size is not None:
            gram_crop_1 = self.resize_3d_linear(g1_transf, self.gram_teacher_crops_size)
            output["gram_teacher_crops"] = [gram_crop_1]

        # Local crops: generate a list of local augmentations
        local_crops = []
        for _ in range(self.local_crops_number):
            aug_local = convert_to_tensor(self.geometric_augmentation_local(image)[0], track_meta=False)
            aug_local = self.resize_3d_linear(aug_local, self.local_crops_size)
            local_crops.append(aug_local)
        output["local_crops"] = local_crops
        # Optionally add other augmentation information (e.g., offsets)
        output["offsets"] = ()  
        output["caption"] = caption
        if "file_path" in sample:
            output["file_path"] = sample["file_path"]
        
        # Preserve OSL (Opposite Sentence Loss) data if present
        if "osl_pairs" in sample:
            output["osl_pairs"] = sample["osl_pairs"]
        if "osl_labels" in sample:
            output["osl_labels"] = sample["osl_labels"]
        
        return output
    

class RandomCropCLIP3D(object):
    def __init__(
        self,
        local_views_scale = (0.1875, 0.5),
        local_crops_number = 8,
        global_crops_size=(128, 128, 128),
        local_crops_size=(64, 64, 64),
        gram_teacher_crops_size=None,
        use_intensity_transforms = False,
    ):
        self.local_views_scale = local_views_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = tuple(int(s) for s in global_crops_size)
        self.local_crops_size = tuple(int(s) for s in local_crops_size)
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.use_intensity_transforms = use_intensity_transforms

        if gram_teacher_crops_size is not None:
            self.gram_teacher_crops_size = tuple(int(s) for s in gram_teacher_crops_size)
            self.global_crop_max_size = tuple(
                max(gs, gts)
                for gs, gts in zip(self.global_crops_size, self.gram_teacher_crops_size)
            )
        else:
            self.gram_teacher_crops_size = None
            self.global_crop_max_size = self.global_crops_size

        if self.use_intensity_transforms:
            self.color_jitter = monai.transforms.Compose([
                monai.transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
                # monai.transforms.RandGaussianSmooth(sigma_x=(0.5, 1.0),sigma_y=(0.5, 1.0),sigma_z=(0.5, 1.0),prob=0.2,),
                monai.transforms.RandScaleIntensity(factors=(-0.25, 0.25), prob=0.15),
                # monai.transforms.RandSimulateLowResolution(prob=0.25,zoom_range=(0.5, 1.0)),
                monai.transforms.RandAdjustContrast(prob=0.1, gamma=(0.7, 1.5)),
            ])
            
        self.geometric_augmentation_global = monai.transforms.CenterSpatialCrop(
                            roi_size=tuple(sz for sz in self.global_crop_max_size),
                        )

    def resize_3d_linear(self, x, size):
        """
        x: (C, D, H, W)
        out_size: (D, H, W)
        """
        if x.ndim == 4:
            x_in = x.unsqueeze(0)   # (1, C, D, H, W)
            x = F.interpolate(
                x_in,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
            return x.squeeze(0)
        else:
            return  F.interpolate(
                x,
                size=size,
                mode="trilinear",     
                align_corners=False,
            )
        
    def pad_min(self, img: torch.Tensor, target_size):
        """
        Pads a 3D or 4D tensor to target_size (D, H, W) using the minimum
        value in the tensor. Padding is added only on the *right side*
        (no centering, no symmetry).
        """
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(img)

        D, H, W = img.shape[-3:]
        td, th, tw = target_size

        pad_d = max(0, td - D)
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)

        # split padding on both sides (left/right) to center the content
        pad_d_left  = pad_d // 2
        pad_d_right = pad_d - pad_d_left

        pad_h_left  = pad_h // 2
        pad_h_right = pad_h - pad_h_left

        pad_w_left  = pad_w // 2
        pad_w_right = pad_w - pad_w_left

        # F.pad expects: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)

        pad_value = img.amin().item()  # item() gives python scalar; safe for F.pad
        return F.pad(img, pad, mode="constant", value=pad_value)

    def __call__(self, sample):
        # Expecting sample to a dict
        image = sample["image"]
        caption = sample["caption"]
        output = {}
        # Sample shape, 1,C,D,H,W
        # Global crops: generate two augmented versions
        image = self.pad_min(image, self.global_crop_max_size)
        if self.use_intensity_transforms:
            image = convert_to_tensor(self.color_jitter(image), track_meta=False)
        global_crops1 = self.geometric_augmentation_global(image)
        g1_transf = convert_to_tensor(global_crops1, track_meta=False)
        global_crop_1 = self.resize_3d_linear(g1_transf, self.global_crops_size)
        
        output["global_crops"] = [global_crop_1]
        output["global_crops_teacher"] = [g1_transf]

        if self.gram_teacher_crops_size is not None:
            gram_crop_1 = self.resize_3d_linear(g1_transf, self.gram_teacher_crops_size)
            output["gram_teacher_crops"] = [gram_crop_1]

        # Optionally add other augmentation information (e.g., offsets)
        output["offsets"] = ()  
        output["caption"] = caption
        if "file_path" in sample:
            output["file_path"] = sample["file_path"]
        
        # Preserve OSL (Opposite Sentence Loss) data if present
        if "osl_pairs" in sample:
            output["osl_pairs"] = sample["osl_pairs"]
        if "osl_labels" in sample:
            output["osl_labels"] = sample["osl_labels"]
        
        return output