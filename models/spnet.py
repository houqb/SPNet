###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .base import BaseNet
from .fcn import FCNHead
from .customize import PyramidPooling, StripPooling

class SPNet(BaseNet):
    def __init__(self, nclass, backbone, pretrained, criterion=None, aux=True, norm_layer=nn.BatchNorm2d, spm_on=False, **kwargs):
        super(SPNet, self).__init__(nclass, backbone, aux, pretrained, norm_layer=norm_layer, spm_on=spm_on, **kwargs)
        self.head = SPHead(2048, nclass, norm_layer, self._up_kwargs)
        self.criterion = criterion
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x, y=None):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = interpolate(x, (h,w), **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = interpolate(auxout, (h,w), **self._up_kwargs)
        
        if self.training:
            aux = self.auxlayer(c3)
            aux = interpolate(aux, (h, w), **self._up_kwargs)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x
