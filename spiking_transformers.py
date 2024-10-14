from timm.models.layers import trunc_normal_
from braincog.base.strategy.surrogate import *
from models.spikformer import spikformer
from st_utils.layers.st_mlp import *
from st_utils.layers.st_embed import *
from st_utils.layers.st_attn import *
from st_utils.layers.st_heads import *
from st_utils.widgets import spikformer_block,TIM
from st_utils.node import *


class Spikformer(nn.Module):
    def __init__(self, step=10,
                 img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratios=4, scale=0.25,
                 mlp_drop=0., attn_drop=0.,depths=2,):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        patch_embed = SPS(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim)

        block = nn.ModuleList([spikformer_block.Block(step=step,embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratios,
                                     scale=scale, mlp_drop=mlp_drop, attn_drop=attn_drop)

                               for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)
        # classification head
        self.head = cls_head(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x