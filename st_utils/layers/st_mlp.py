import torch.nn as nn
from braincog.model_zoo.base_module import BaseModule
from st_utils.node.st_LIFNode import *

class SCS_block(nn.Module):
    def __init__(self, embed_dims, scs_ratio=2, *args, **kwargs):
        super(SCS_block,self).__init__()
        self.scs_dim = int(embed_dims * scs_ratio)
        self.scs_block = nn.Sequential(
            nn.Conv2d(embed_dims, self.scs_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.scs_dim),
            nn.Conv2d(self.scs_dim, embed_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims)
        )

    def forward(self, x):
        x = self.scs_block(x)
        return x