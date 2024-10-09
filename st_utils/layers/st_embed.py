import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from st_utils.node.st_LIFNode import *

class SPS(BaseModule):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256):
        super().__init__(step=step, encode_type=encode_type)
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size

        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = st_LIFNode(step=step)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = st_LIFNode(step=step)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = st_LIFNode(step=step)


        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = st_LIFNode(step=step)


        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = st_LIFNode(step=step)

    #Conv - BN - LIF layer in SPS
    def ConvBnSn(self,x,conv,bn,lif):
        T, B, C, H, W = x.shape
        x = conv(x)
        x = bn(x).reshape(T, B, -1, H, W).contiguous()
        x = lif(x.flatten(0, 1)).contiguous()
        return x

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method")

# Spikformer SPS
# Default: For dvs data with step equals to 10
class SPSv1(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256):
        super().__init__(step=step, encode_type=encode_type)

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.rpe_lif = st_LIFNode(step=step)

    def forward(self, x):
        T, B, C, H, W = x.shape

        assert self.embed_dims % 8 == 0, 'embed_dims must be divisible by 8 in Spikformer'

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x.flatten(0, 1)).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x.flatten(0, 1)).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x.flatten(0, 1)).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj_lif3(x.flatten(0, 1)).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.rpe_lif(x.flatten(0, 1)).reshape(T, B, -1, H // 4, W // 4)

        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x

#Spikformer v2 SPS
