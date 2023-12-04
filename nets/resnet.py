import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
from torch.nn.modules.utils import _triple, _pair, _single
# import softpool_cuda
# from SoftPool import soft_pool2d, SoftPool2d
from torch.autograd import Function
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from nets.PConv import Pconv
from nets.attention import Att, AMM, ECAAttention
# from nets.odconv import ODConv2d
from nets.pooling import StripPooling


model_urls = {
    'resnet50': 'https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/resnet50s-a75c83cf.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
}



class SoftPool2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(SoftPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool
    #------------------------------------------------------------------#

    #-------------------------------------------------------------------#
# def odconv3x3(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,reduction=reduction, kernel_num=kernel_num
#                    )


# def odconv1x1(in_planes, out_planes, stride=1, reduction=0.0625, kernel_num=1):
#     return ODConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0,reduction=reduction, kernel_num=kernel_num
#                    )




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, previous_dilation=1,use_triplet_attention=True,reduction=0.0625, kernel_num=1,
                 norm_layer=None,no_spatial=False,kernel_size=1):
        super(Bottleneck, self).__init__()
        self.conv1  = Pconv(inplanes, planes, kernel_size=1,n_div=2,
                 forward=str('train'))

        self.bn1    = norm_layer(planes)

        self.conv2  = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

        # self.conv2 = PSAModule(planes, planes, stride=stride, conv_kernels=[3,5,7,9], conv_groups=[1,4,8,16])
        # self.conv2 = CoTAttention(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
        #                        bias=False)
        self.bn2    = norm_layer(planes)

        self.conv3  = Pconv(planes, planes * 4, kernel_size=1,n_div=2,
                 forward=str('train'))

        self.bn3    = norm_layer(planes * 4)


        self.relu   = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.dilation   = dilation
        self.stride     = stride
        self.no_spatial = no_spatial
        # self.nam = Att(planes * 4, no_spatial=self.no_spatial, shape=48)
        # self.amm = AMM(planes * 4,16)
        self.eca = ECAAttention(255)

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
       # out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # out = self.nam(out)
        # print('111')
        # out = self.amm(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, dilated=False, deep_base=True, norm_layer=nn.BatchNorm2d):
        self.inplanes = 128 if deep_base else 64
        super(ResNet, self).__init__()
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),

                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                # Pconv(64, 64, kernel_size=3),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
                # Pconv(64, 128, kernel_size=3),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1        = norm_layer(self.inplanes)
        self.relu       = nn.ReLU(inplace=True)
        
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = SoftPool2D(kernel_size=3, stride=2)
        # self.maxpool = StripPooling(128, (20, 12), norm_layer, 2)

        
        self.layer1     = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2     = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                            dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
            
        self.avgpool    = nn.AvgPool2d(7, stride=1)
        self.fc         = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        multi_dilations = [4, 8, 16]
        if multi_grid:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilations[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if multi_grid:
                layers.append(block(self.inplanes, planes, dilation=multi_dilations[i],
                                    previous_dilation=dilation, norm_layer=norm_layer))
            else:
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['resnet50'], "./model_data"), strict=False)
    return model
