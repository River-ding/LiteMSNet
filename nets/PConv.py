import torch.nn as nn
import torch


class Pconv(nn.Module):
    def __init__(self, in_channel,
                 out_channel,
                 n_div=2,
                 forward=str('train'),
                 kernel_size=3):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.dim_conv = in_channel // n_div
        self.dim_untouched = in_channel - self.dim_conv
        self.conv1 = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            self.in_channel,
            self.out_channel,
            kernel_size=1
        )
        if forward == 'train':
            self.forward = self.forward_train
        elif forward == 'test':
            self.forward = self.forward_test

    def forward_train(self, x):
        x1, x2 = torch.split(x, (self.dim_conv, self.dim_untouched), dim=1)
        x1 = self.conv1(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv2(x)
        return x

    def forward_test(self, x):
        x[:, :self.dim_conv, :, :] = self.conv1(x[:, :self.dim_conv, :, :])
        x = self.conv2(x)
        return x