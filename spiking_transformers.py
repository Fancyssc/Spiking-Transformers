from timm.models import register_model
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _cfg
from braincog.base.strategy.surrogate import *
from st_utils.node import *
from st_utils.layer import *

#original Spikformer(ICLR 2023)
class Spikformer(nn.Module):
    def __init__(self, step=10,
                 img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25,mlp_drop=0.,
                 attn_drop=0.,depths=2,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        patch_embed = SPSv1(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim,
                          node=node,tau=tau,
                          act_func=act_func,threshold=threshold)

        block = nn.ModuleList([Spikf_Block(step=step,embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio,
                                           scale=scale, mlp_drop=mlp_drop, attn_drop=attn_drop,
                                           node=node,tau=2.0,act_func=act_func,threshold=threshold)

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

# Spikformer with TIM(IJCAI 2024)
class Spikformer_TIM(nn.Module):
    def __init__(self, step=10,
                 img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25,mlp_drop=0.,
                 attn_drop=0.,depths=2,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        patch_embed = SPSv1(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim,
                          node=node,tau=tau,
                          act_func=act_func,threshold=threshold)
        #if_TIM = True
        block = nn.ModuleList([Spikf_Block(step=step,embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio,
                                           scale=scale, mlp_drop=mlp_drop, attn_drop=attn_drop,
                                           node=node,tau=2.0,act_func=act_func,threshold=threshold,if_TIM=True)

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


# Spike-driven Transformer(NeurIPS 2023)
class SpikeDrivnTransf(nn.Module):
    def __init__(self, step=10,
                 img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25,mlp_drop=0.,
                 attn_drop=0.,depths=2,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        patch_embed = SPS_sdt(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim,
                          node=node,tau=tau,
                          act_func=act_func,threshold=threshold)
        #if_TIM = True
        block = nn.ModuleList([Sdt_Block(step=step,embed_dim=embed_dim,
                                           num_heads=num_heads, mlp_ratio=mlp_ratio,
                                           scale=scale, mlp_drop=mlp_drop, attn_drop=attn_drop,
                                           node=node,tau=2.0,act_func=act_func,threshold=threshold,if_TIM=True)

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
        return x.flatten(3).mean(3) #sdt needs flatten

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

#QKFormer (Nips 2024)
class QKFormer(nn.Module):
    def __init__(self, step=10,img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25,mlp_drop=0.,
                 attn_drop=0.,depths=2,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__()
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        patch_embed1 = PEDS_init(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim // 2,
                          node=node,tau=tau,
                          act_func=act_func,threshold=threshold)

        stage1 = nn.ModuleList([SSA(embed_dim=embed_dim//2, num_heads=num_heads,node=node,act_func=act_func,threshold=threshold,tau=tau)
            for j in range(1)])

        patch_embed2 = PEDS_stage(step=step,
                          img_h=img_h,
                          img_w=img_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dim,
                          node=node,tau=tau,
                          act_func=act_func,threshold=threshold)
        stage2 = nn.ModuleList([SSA(
            embed_dim=embed_dim, num_heads=num_heads,scale=0.25,attn_drop=0.,node=node,
            tau=tau,act_func=act_func,threshold=threshold)
            for j in range(depths-1)])

        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"stage1", stage1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage2", stage2)

        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pose_embed'}

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        return None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
             x = blk(x)

        return x.flatten(3).mean(3)


    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

# Registered Models
@register_model
def spikformer(pretrained=False,**kwargs):
    model = Spikformer(step=10,
                 img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25,mlp_drop=0.,
                 attn_drop=0.,depths=2,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5
    )
    model.default_cfg = _cfg()
    return model

@register_model
def spikformer_TIM(pretrained=False,**kwargs):
    model = Spikformer_TIM(step=10,
                 img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                 embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25,mlp_drop=0.,
                 attn_drop=0.,depths=2,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5)
    model.default_cfg = _cfg()
    return model

@register_model
def sdt(pretrained=False,**kwargs):
    model = SpikeDrivnTransf(step=10,
                           img_h=128, img_w=128, patch_size=16, in_channels=2, num_classes=10,
                           embed_dim=256, num_heads=16, mlp_ratio=4, scale=0.25, mlp_drop=0.,
                           attn_drop=0., depths=2, node=st_LIFNode, tau=2.0, act_func=Sigmoid_Grad, threshold=0.5)
    model.default_cfg = _cfg()
    return model

@register_model
def qkformer(pretrained=False, **kwargs):
    model = QKFormer(
        patch_size=16, embed_dim=256, num_heads=16, mlp_ratio=4,
        in_channels=2, num_classes=10, scale=0.25, mlp_drop=0.,
                           attn_drop=0., depths=2, node=st_LIFNode, tau=2.0, act_func=Sigmoid_Grad, threshold=0.5
    )
    model.default_cfg = _cfg()
    return model

