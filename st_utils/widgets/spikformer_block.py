from braincog.model_zoo.base_module import BaseModule
from st_utils.node.st_LIFNode import *
from st_utils.layers.st_mlp import *
from st_utils.layers.st_embed import *
from st_utils.layers.st_attn import *
from st_utils.widgets import *
from st_utils.node import *

class Block(nn.Module):
    def __init__(self, embed_dim=256, num_heads=16, step=10, mlp_ratio=4., scale=0., attn_drop=0.,mlp_drop=0.,):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SSA(embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop, scale=scale,)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x
