# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


import nets.layers
from nets.DABNet import DABModule
from nets.PConv import Pconv
from nets.attention import Att
from nets.layers import *
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
from nets.attention import CoTAttention



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from nets.odconv import ODConv2d


# def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,reduction=reduction, kernel_num=kernel_num)
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Pconv(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Pconv(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock1(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int ,
            planes: int ,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        # self.nam = Att(planes ,shape=48)
        self.cot = CoTAttention(planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # out = self.nam(out)
        out = self.cot(out)
        out += identity
        out = self.relu(out)

        return out




class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,use_triplet_attention=True,
                 norm_layer=None,no_spatial=False):
        super(Bottleneck, self).__init__()
        self.conv1  = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1    = norm_layer(planes)

        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        # self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=[3,5,7,9], conv_groups=[1,4,8,16])
        # self.conv2 = CoTAttention(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
        #                        bias=False)
        self.bn2    = norm_layer(planes)

        self.conv3  = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3    = norm_layer(planes * 4)


        self.relu   = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.dilation   = dilation
        self.stride     = stride
        self.no_spatial = no_spatial
        # self.nam = Att(planes * 4, no_spatial=self.no_spatial, shape=48)
        # self.amm = AMM(planes * 4,16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # out = self.nam(out)
        # print('111')
        # out = self.amm(out)
        out += residual
        out = self.relu(out)
        return out

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)
        # self.conv.cuda()

    def forward(self, x):
        out = self.pad(x)
        # out = self.conv(out)
        out = self.conv(out)
        return out


class Transblock(nn.Module):
    def __init__(self, dim):
        super(Transblock, self).__init__()
        self.conv1 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim, eps=1e-6)
        self.G = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.LN(x)
        x = self.conv1(x)
        x = self.LN(x)
        x = self.G(x)
        x = x.permute(0, 3, 1, 2)
        return x


class PWSA(nn.Module):
    def __init__(self, inplanes, planes):
        super(PWSA, self).__init__()
        self.resclock = BasicBlock(inplanes, planes)
        self.resclock1 = BasicBlock1(inplanes, planes)
        self.conv1x1 = nets.layers.Conv1x1(inplanes, planes)
        self.conv1x1_Matt = nets.layers.Conv1x1(1024, 256)
        self.conv1x1_up = nets.layers.Conv1x1(256, 512)
        self.Relu = nn.ReLU()
        self.Softmax = nn.Softmax(dim=-1)
        self.dab = DABModule(inplanes,planes)

    def forward(self, x,y):
        # x, y = input
        # Add = torch.cat([x,y],1)
        Add = torch.add(x, y)
        # Add = self.conv1x1_up(Add)
        # x = self.resclock(x)
        x = self.dab(x)
        # x = self.conv1x1(x)
        #Att = self.Relu(x)
        Att = self.Softmax(x)
        z = torch.mul(Add, Att)
        # z = self.dab(z)
        # z = self.resclock(z)
        z = torch.add(z, Add)

        z = nets.layers.upsample(z)
        z = self.conv1x1(z)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1):
        super(ConvDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales
        self.device = torch.device("cuda")
        self.num_ch_enc = num_ch_enc

        self.Upsample = OrderedDict()
        self.num_ch_dec = [self.num_ch_enc[0], self.num_ch_enc[0], self.num_ch_enc[1], self.num_ch_enc[2]]
        self.Linear_ch = [self.num_ch_enc[0], self.num_ch_enc[0], self.num_ch_enc[0], self.num_ch_enc[1]]
        self.resblock = BasicBlock(self.num_ch_enc[-1], self.num_ch_enc[-1])

        self.PWSA = nn.ModuleList()
        for i in range(4):
            Patt = nn.Sequential(
                *[PWSA(self.num_ch_dec[i], self.num_ch_dec[i])])
            self.PWSA.append(Patt)

        self.Transblock = nn.ModuleList()

        for i in range(4):
            trans = nn.Sequential(
                *[Transblock(self.num_ch_enc[i])])
            self.Transblock.append(trans)

        self.conv1x1_3 = nets.layers.Conv1x1(self.num_ch_enc[3], self.num_ch_enc[2])
        self.conv1x1_2 = nets.layers.Conv1x1(self.num_ch_enc[2], self.num_ch_enc[1])
        self.conv1x1_1 = nets.layers.Conv1x1(self.num_ch_enc[1], self.num_ch_enc[0])
        self.conv1x1_0 = nets.layers.Conv1x1(self.num_ch_enc[0], self.num_ch_enc[0])

        self.convs = OrderedDict()
        for i in range(len(self.num_ch_enc)):
            # outputs
            self.convs[("outputs", i, 2)] = Conv3x3(self.Linear_ch[i], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        # decoder

    def forward(self, input_features):
        self.outputs = {}

        f0 = self.Transblock[0](input_features[0])
        f1 = self.Transblock[1](input_features[1])
        f2 = self.Transblock[2](input_features[2])
        f3 = self.Transblock[3](input_features[3])

        x = self.resblock(f3)
        x = nets.layers.upsample(x)
        x = self.conv1x1_3(x)

        x = self.PWSA[3]((f2, x))
        x = self.conv1x1_2(x)
        self.outputs[("disp", 3)] = self.sigmoid(self.convs[("outputs", 3, 2)](x))

        x = self.PWSA[2]((f1, x))
        x = self.conv1x1_1(x)
        self.outputs[("disp", 2)] = self.sigmoid(self.convs[("outputs", 2, 2)](x))

        x = self.PWSA[1]((f0, x))
        x = self.conv1x1_0(x)
        self.outputs[("disp", 1)] = self.sigmoid(self.convs[("outputs", 1, 2)](x))

        f0 = nets.layers.upsample(f0)
        x = self.PWSA[0]((f0, x))
        x = self.conv1x1_0(x)
        self.outputs[("disp", 0)] = self.sigmoid(self.convs[("outputs", 0, 2)](x))

        return self.outputs
