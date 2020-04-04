###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.scatter_gather import scatter

from . import resnet
#from ..utils import batch_pix_accuracy, batch_intersection_union

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

__all__ = ['BaseNet', 'MultiEvalModule']

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, pretrained, dilated=True, norm_layer=None, spm_on=False):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        # copying modules from pretrained models
        self.backbone = backbone
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=pretrained, dilated=dilated,
                                              norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=pretrained, dilated=dilated,
                                               norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=pretrained, dilated=dilated,
                                               norm_layer=norm_layer, multi_grid=True, spm_on=spm_on)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        if self.backbone.startswith('wideresnet'):
            x = self.pretrained.mod1(x)
            x = self.pretrained.pool2(x)
            x = self.pretrained.mod2(x)
            x = self.pretrained.pool3(x)
            x = self.pretrained.mod3(x)
            x = self.pretrained.mod4(x)
            x = self.pretrained.mod5(x)
            c3 = x.clone()
            x = self.pretrained.mod6(x)
            x = self.pretrained.mod7(x)
            x = self.pretrained.bn_out(x)
            return None, None, c3, x
        else:
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union
