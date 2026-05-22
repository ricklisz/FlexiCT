"""Flexi-CT segmentation heads used by the downstream nnU-Net trainer."""
from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Literal

import torch
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch import nn

from .flexi_ct_2d import Flexi_CT_2D, _BACKBONE_KWARGS, _load_teacher_into_backbone
from .flexi_ct_3d import Flexi_CT_3D
from .models import flexi_ct_backbone_base


_DEFAULT_INTERACTION_INDICES = (3, 7, 11, 15)
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.patch_encode_decode import LayerNormNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
import numpy as np

import math

class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size: int, 
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
        dim=2
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()
        assert patch_size > 0
        n = int(math.log2(patch_size))

        assert 2 ** n == patch_size and n >= 1

        ch = [embed_dim]
        for _ in range(n):
            ch.append(ch[-1]//2)
        ch.append(out_channels)

        stages = []

        if dim == 2:
            for i in range(n):
                stages.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                        norm(ch[i + 1]),
                        activation(),
                    )
                )
            stages.append(nn.Conv2d(ch[-2], ch[-1], kernel_size=1))
        elif dim == 3:
            for i in range(n):
                stages.append(
                    nn.Sequential(
                        nn.ConvTranspose3d(ch[i], ch[i + 1], kernel_size=2, stride=2),
                        norm(ch[i + 1]),
                        activation(),
                    )
                )
            stages.append(nn.Conv3d(ch[-2], ch[-1], kernel_size=1))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = False

class Primus(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        dim = 2
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__()

        self.in_chans = in_chans
        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act, dim=dim
        )
        self.dim = dim
        self.decoder.apply(InitWeights_He(1e-2))

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1,self.in_chans,1,1)
            elif x.dim() == 5:
                x = x.repeat(1,self.in_chans,1,1,1)
        x = self.dino_encoder.get_intermediate_layers(x,  n=1, reshape = True)[0]
        dec_out = self.decoder(x)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
    
class Primus_v2(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        dim = 2,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.dim = dim
        self.decoder = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act, 
            dim = dim
        )
        proj_dim = (embed_dim * len(interaction_indices))
        
        if dim == 2:
            self.projectors =  nn.Sequential(
                    nn.Conv2d(
                        proj_dim,
                        embed_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNormNd(embed_dim),
                    )
        else:
            self.projectors =  nn.Sequential(
                    nn.Conv3d(
                        proj_dim,
                        embed_dim,
                        kernel_size=1,
                        bias=False,
                    ),
                    LayerNormNd(embed_dim),
                    )
        
        self.in_chans = in_chans
        self.dino_encoder = dino_encoder
        self.decoder.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1,self.in_chans,1,1)
            elif x.dim() == 5:
                x = x.repeat(1,self.in_chans,1,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        hier = self.projectors(hier)
        dec_out = self.decoder(hier)
        return dec_out
    
class Primus_Multiscale(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_embed_size: int,
        num_classes: int,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        dino_encoder = None,
        dim = 2,
        interaction_indices =[1,2,3,4]
    ):
        """
        We follow a similar design as ViT-adapter, using intermediate layers and concat along channel dimension.
        """
        super().__init__()

        self.dim = dim
        self.decoder = PatchDecode(
            patch_embed_size, embed_dim * len(interaction_indices), num_classes, norm=decoder_norm, activation=decoder_act, 
            dim = dim
        )
        self.in_chans = in_chans
        # we need to compute the ref_feat_shape for eva
        self.dino_encoder = dino_encoder
        self.decoder.apply(InitWeights_He(1e-2))
        self.interaction_indices=interaction_indices

    def forward(self, x):
        if x.shape[1] != self.in_chans:
            if x.dim() == 4:
                x = x.repeat(1,self.in_chans,1,1)
            elif x.dim() == 5:
                x = x.repeat(1,self.in_chans,1,1,1)
        hier = self.dino_encoder.get_intermediate_layers(x,  n=self.interaction_indices, reshape = True)
        hier = torch.cat(hier, dim=1)
        dec_out = self.decoder(hier)
        return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")
