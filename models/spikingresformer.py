import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *
from LIFNode import MyNode  # LIFNode setting for Spiking Tranformers
from functools import partial

__all__ = ['spikformer']

# This is the original spikingresformer desgined only for ImageNet Classfication
class MLP(BaseModule):
    # Linear -> BN -> LIF -> Linear -> BN -> LIF
    def __init__(self, in_features, step=4, encode_type='rate', hidden_features=None, out_features=None, drop=0.):
        super().__init__(step=4, encode_type='rate')
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MyNode(step=step,tau=2.0)

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MyNode(step=step,tau=2.0)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        self.reset()

        T, B, N, C = x.shape

        x_ = x.flatten(0, 1)  # TB N C

        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()  # T B N C
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, N, self.c_hidden)

        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, N, self.c_output)
        return x


class SSA(BaseModule):
    def __init__(self, dim, step=4, encode_type='rate', num_heads=12, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_ratio=1):
        super().__init__(step=4, encode_type='rate')
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        # 多头注意力 # of heads
        self.num_heads = num_heads
        # scale参数，用于防止KQ乘积结果过大
        self.scale = 0.125

        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MyNode(step=step,tau=2.0)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MyNode(step=step,tau=2.0)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MyNode(step=step,tau=2.0)

        self.attn_lif = MyNode(step=step, tau=2.0, v_threshold=0.5, )

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MyNode(step=step, tau=2.0, )

    def forward(self, x):
        self.reset()

        T, B, N, C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C

        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N,
                                                                                           C).contiguous()  # T B N C
        q_linear_out = self.q_lif(q_linear_out.flatten(0, 1)).reshape(T, B, N, C)  # TB N C
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out.flatten(0, 1)).reshape(T, B, N, C)  # TB N C
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out.flatten(0, 1)).reshape(T, B, N, C)  # TB N C
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # @表示矩阵乘法,与matmul等价
        # K,QV -> attention -> scale -> LIF -> Linear -> BN
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x.flatten(0, 1)).reshape(T, B, N, C)  # T B N C
        x = x.flatten(0, 1)  # TB N C
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2)).reshape(T, B, N, C)
        return x


class DSSA(BaseModule):
    def __init__(self, dim, patch_size, step=4, encode_type='direcrt', num_heads=12)
        super().__init__(step=step, encode_type=encode_type)
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.input_lif = MyNode(step=step,tau=2.0)
        

    def forward(self, x):
        self.reset()

        T, B, C, H, W = x.shape

        x_feat = x.clone()
        x = self.input_lif(x)



        x_for_qkv = x.flatten(0, 1)  # TB, N, C

        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N,
                                                                                           C).contiguous()  # T B N C
        q_linear_out = self.q_lif(q_linear_out.flatten(0, 1)).reshape(T, B, N, C)  # TB N C
        q = q_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out.flatten(0, 1)).reshape(T, B, N, C)  # TB N C
        k = k_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out.flatten(0, 1)).reshape(T, B, N, C)  # TB N C
        v = v_linear_out.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # @表示矩阵乘法,与matmul等价
        # K,QV -> attention -> scale -> LIF -> Linear -> BN
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x.flatten(0, 1)).reshape(T, B, N, C)  # T B N C
        x = x.flatten(0, 1)  # TB N C
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2)).reshape(T, B, N, C)
        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, step =4,  mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.step = 4
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, step=self.step, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(step=self.step, in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # residual
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



# SPS is abandoned in SpikingResFormer
class STEM(nn.Module):
    def __init__(self, img_size_h=224, img_size_w=224, in_channels=3,
                 out_channels=64  ):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=7,padding=3,stride=2,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mp = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.out_channels = out_channels

    def forward(self, x):
        
        # x: T B C W H 
        T, B, C, W, H = x.shape
        x = self.conv(x.flatten(0,1))
        x = self.bn(x)
        x = self.mp(x).reshape(T, B, C, W//4, H//4).contiguous()
        
        #op: imgsize = imgsize // 4 (imnet 224 -> 56) 
        return x, self.out_channels # out_channels should be written as the in_channels for downsample layer 

class DownsampleLayer(BaseModule):
    def __init__(self, in_channels, out_channels=384, step = 4, encode_type='direct',):
        super().__init__(step = 4, encode_type='direct')
        
        self.lif = MyNode(step=step, tau=2.0)
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x: torch.Tensor):
        self.reset()

        T, B, C, W, H = x.shape

        x = self.lif(x.flatten(0,1)).reshape(T, B, C, W, H).contiguous()
        x = self.conv(x.flatten(0,1))
        x = self.bn(x).reshape(T, B, -1, W//2, H//2).contiguous()

        return x
    

class Spikformer(BaseModule):
    def __init__(self, step=4, encode_type='direct',
                 img_size_h=224, img_size_w=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dims=384, num_heads=12, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=4, sr_ratios=4,
                 ):
        super().__init__(step=4, encode_type='direct')
        self.step = step  # time step
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(step = self.step,
                          img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)

        block = nn.ModuleList([Block(step=self.step,
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)

            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

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
        return x.mean(2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def spikingresformer_imnet(pretrained=False, **kwargs):
    model = Spikformer(
        step=4,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def spikingresformer_dvsc10(pretrained=False, **kwargs):
    model = Spikformer(
        step=4,
        img_size_h=224, img_size_w=224,
        patch_size=16, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=1000, qkv_bias=False,
        depths=12, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model