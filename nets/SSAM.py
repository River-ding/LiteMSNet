import torch.nn as nn
import torch
import torch.nn.functional as F


class SSAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSAM, self).__init__()
        self.conv_shared = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn_shared_max = nn.BatchNorm2d(in_channels)
        self.bn_shared_avg = nn.BatchNorm2d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        max_pool = F.max_pool2d(x, [1, W])
        max_pool = self.conv_shared(max_pool)
        max_pool = self.bn_shared_max(max_pool)

        avg_pool = F.avg_pool2d(x, [1, W])
        avg_pool = self.conv_shared(avg_pool)
        avg_pool = self.bn_shared_avg(avg_pool)

        att = torch.softmax(torch.mul(max_pool, avg_pool), 1)

        f_scale = att * max_pool + att * avg_pool
        out = F.relu(self.gamma * f_scale + (1 - self.gamma) * x)
        return out