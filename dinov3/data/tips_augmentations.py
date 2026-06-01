# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
#
# TIPS: Text-Image Pretraining with Spatial awareness
# Data augmentation and collate functions for TIPS training.
#
# Key difference from DINOv2: Uses SINGLE global crop instead of 2.

import logging
import random

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from dinov3.data.transforms import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov3")


class DataAugmentationTIPS(object):
    """
    Data augmentation for TIPS training.
    
    Key difference from DataAugmentationDINO: produces only 1 global crop
    instead of 2, as specified in the TIPS paper Section 3.2.
    This increases throughput by ~25%.
    """
    
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
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using TIPS data augmentation (1 global crop)")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"patch_size: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # Random resized crop and flip for global crop
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

        # Resize transforms
        resize_global = nn.Identity()
        self.resize_global_post_transf = nn.Identity()
        self.resize_gram_teacher = None
        
        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                self.resize_gram_teacher = transforms.Resize(
                    gram_teacher_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            else:
                self.resize_gram_teacher = transforms.Compose(
                    [
                        transforms.Resize(
                            gram_teacher_crops_size,
                            interpolation=transforms.InterpolationMode.BICUBIC,
                        ),
                    ]
                )
            if gram_teacher_crops_size > global_crops_size:
                resize_global = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            elif gram_teacher_crops_size < global_crops_size:
                self.resize_global_post_transf = transforms.Resize(
                    global_crops_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )

        # Local crops
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

        # Color jittering
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        # Normalization
        self.normalize = make_normalize_transform(mean=mean, std=std)

        # Global transform (with solarization and blur)
        global_transfo_extra = transforms.Compose(
            [
                GaussianBlur(p=1.0),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        # Local transform (with blur only)
        local_transfo_extra = transforms.Compose(
            [
                GaussianBlur(p=0.5),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo = transforms.Compose([resize_global, global_transfo_extra, self.normalize])
            self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo = transforms.Compose(
                [resize_global, color_jittering, global_transfo_extra, self.normalize]
            )
            self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # TIPS: Single global crop (key difference from DINOv2)
        im_base = self.geometric_augmentation_global(image)
        global_crop_transf = self.global_transfo(im_base)
        global_crop = self.resize_global_post_transf(global_crop_transf)

        # Output single global crop as a list (for compatibility with collate)
        output["global_crops"] = [global_crop]

        # Global crop for teacher
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [self.normalize(im_base)]
        else:
            output["global_crops_teacher"] = [global_crop]

        # Gram teacher crops (if needed)
        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                gram_crop = self.normalize(self.resize_gram_teacher(im_base))
            else:
                gram_crop = self.resize_gram_teacher(global_crop_transf)
            output["gram_teacher_crops"] = [gram_crop]

        # Local crops (all derived from the single global crop base)
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image))
            for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output


def collate_data_and_cast_tips(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    """
    Collate function for TIPS training.
    
    Key difference from standard collate: expects 1 global crop per sample.
    """
    # TIPS: hardcoded 1 global crop
    n_global_crops = 1
    n_local_crops = len(samples_list[0][0]["local_crops"])

    # Verify we have exactly 1 global crop
    assert len(samples_list[0][0]["global_crops"]) == 1, (
        f"TIPS expects 1 global crop, got {len(samples_list[0][0]['global_crops'])}"
    )

    # Collate global crops [1 * B, ...]
    collated_global_crops = torch.stack(
        [s[0]["global_crops"][0] for s in samples_list]
    )
    
    # Collate local crops [n_local_crops * B, ...]
    collated_local_crops = torch.stack(
        [s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )
    
    # Collate gram teacher crops if present
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][0] for s in samples_list]
        )
    else:
        collated_gram_teacher_crops = None

    # Determine batch size for masking
    if local_batch_size is not None:
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    
    return out


class TIPSCollateFunction:
    """
    Collate function wrapper for TIPS training with image-caption pairs.
    
    Handles both image collation (1 global crop) and text tokenization.
    """
    
    def __init__(
        self,
        mask_ratio_tuple,
        mask_probability,
        dtype,
        n_tokens,
        mask_generator,
        tokenizer=None,
        context_length: int = 77,
        random_circular_shift=False,
    ):
        """
        Args:
            mask_ratio_tuple: Min/max mask ratio for iBOT
            mask_probability: Probability of applying mask
            dtype: Data type for tensors
            n_tokens: Number of tokens (patches)
            mask_generator: Mask generator for iBOT
            tokenizer: Text tokenizer for captions (optional)
            context_length: Maximum context length for tokenization
            random_circular_shift: Whether to apply random circular shift to masks
        """
        self.mask_ratio_tuple = mask_ratio_tuple
        self.mask_probability = mask_probability
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.mask_generator = mask_generator
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.random_circular_shift = random_circular_shift
    
    def __call__(self, batch):
        """
        Collate a batch of samples.
        
        Args:
            batch: List of tuples where each tuple is ((image_data_dict, label), caption)
                   or just (image_data_dict, label) if no caption
                   
        Returns:
            Dictionary with collated image data and optionally tokenized text
        """
        # Check if batch contains captions
        has_captions = False
        captions = None
        
        if len(batch) > 0:
            first_item = batch[0]
            # Expected format: ((image_data_dict, label), caption) or (image_data_dict, label)
            if isinstance(first_item, tuple) and len(first_item) == 2:
                # Check if second element is a string (caption)
                if isinstance(first_item[1], str):
                    has_captions = True
        
        if has_captions:
            # Separate image data and captions
            # batch is [((img_dict, label), caption), ...]
            samples_list = [item[0] for item in batch]  # [(img_dict, label), ...]
            captions = [item[1] for item in batch]
        else:
            # No captions, batch is [(img_dict, label), ...]
            samples_list = batch
        
        # Collate images using TIPS collate (1 global crop)
        collated_data = collate_data_and_cast_tips(
            samples_list=[(s,) if not isinstance(s, tuple) else (s[0],) if isinstance(s[0], dict) else s for s in samples_list],
            mask_ratio_tuple=self.mask_ratio_tuple,
            mask_probability=self.mask_probability,
            dtype=self.dtype,
            n_tokens=self.n_tokens,
            mask_generator=self.mask_generator,
            random_circular_shift=self.random_circular_shift,
        )
        
        # Tokenize captions if present
        if has_captions and self.tokenizer is not None:
            text_tokens = self.tokenizer.tokenize(captions, context_length=self.context_length)
            collated_data["text_tokens"] = text_tokens
        
        return collated_data
