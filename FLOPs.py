import torch
from thop import profile
from nets.deeplabv3_plus_LiteMSNet import DeepLab
net = DeepLab(num_classes=11, backbone="mobilenetv3", pretrained=True, downsample_factor=16)
inputs = torch.randn(1, 3, 512, 512)
flops, params = profile(net, (inputs,))
print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops / 1e9, params / 1e6))