import torch.nn as nn
from braincog.model_zoo.base_module import BaseModule
from networkx.classes.filters import hide_nodes

from st_utils.node.st_LIFNode import *

class MLP(BaseModule):
    def __init__(self, in_features, step=10, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.):
        super().__init__(step=step, encode_type=encode_type)
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = st_LIFNode(step=step)

        self.MLP_drop = nn.Dropout(self.mlp_drop)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = st_LIFNode(step=step)

    def forward(self, x):
        self.reset()

        T, B, C, N = x.shape

        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, -1, N).contiguous()  # T B C N
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, -1, N).contiguous()

        if self.mlp_drop > 0 :
            x = self.MLP_drop(x)

        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, N).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, C, N).contiguous()
        return x


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