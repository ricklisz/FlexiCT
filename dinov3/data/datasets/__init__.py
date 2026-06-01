# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .ade20k import ADE20K
from .coco_captions import CocoCaptions
from .image_net import ImageNet, JSONImageNet
from .image_net_22k import ImageNet22k
from .ct_dataset import CT3D, LMDBSliceDataset, LMDBSliceDatasetv2, LMDBSliceMinMaxDataset, LMDBSliceMinMaxDatasetPerModality
from .xray_dataset import XrayLMDBSliceDataset