from functools import partial
from braincog.model_zoo.base_module import BaseModule
from pandas.compat.numpy.function import validate_take_with_convert

from st_utils.node import *
import importlib
import torch.nn as nn

'''
    Spiking Transformer Patch Embeddings
'''
class SPS(BaseModule):
    """
    :param: node: The neuron model used in the Spiking Transformer. The structure of node should obey BaseNode in Braincog
    :param: step: The number of time steps that the neuron will be simulated for.
    :param: encode_type: The encoding type of the input data. 'direct' for direct encoding
    :param: img_h: The height of the input image.
    :param: img_w: The width of the input image.
    :param: patch_size: The size of the patch.
    :param: in_channels: The number of input channels.
    :param: embed_dims: The dimension of the embedding.
    """
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)


    #Conv - BN - LIF layer in SPS
    def ConvBnSn(self,x,conv,bn,lif=None):
        T, B, C, H, W = x.shape
        x = conv(x.flatten(0,1)) # TB C H W
        x = bn(x).reshape(T, B, -1, H, W).contiguous()
        if lif is not None:
            x = lif(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        return x

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method")

# Spikformer SPS
# Default: For dvs data with step equals to 10
class SPSv1(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape
        assert self.embed_dims % 8 == 0, 'embed_dims must be divisible by 8 in Spikformer'

        x = self.ConvBnSn(x,self.proj_conv,self.proj_bn,self.proj_lif)
        x = self.maxpool(x.flatten(0,1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)
        x = self.maxpool1(x.flatten(0,1)).reshape(T, B, -1, H // 4, W // 4).contiguous()

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.maxpool2(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8).contiguous()

        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,self.proj_lif3)
        x = self.maxpool3(x.flatten(0,1)).reshape(T, B, -1, H // 16, W // 16).contiguous()

        x_feat = x.reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.ConvBnSn(x,self.rpe_conv,self.rpe_bn,self.rpe_lif)

        x = x + x_feat
        x = x.flatten(-2)  # T,B,C,N
        return x

#Spikformer v2 SPS
## AKA SCS in Spikformer v2
## Since Spikformer v2 is not open-sourced yet, some params are evaluated by reproducers according to the paper
class SPSv2(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,scs_ratio=2.0,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=2, stride=2, bias=False)
        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=2, stride=2, bias=False)
        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=2, stride=2, bias=False)
        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=2, stride=2, bias=False)

        self.scs_block = SCS_block(embed_dims//8)
        self.scs_block1 = SCS_block(embed_dims//4)
        self.scs_block2 = SCS_block(embed_dims//2)
        self.scs_block3 = SCS_block(embed_dims)
        #conv in SPS v2 down samples image
    def ConvBnSn(self, x, conv, bn, lif = None):
        T, B, C, H, W = x.shape
        x = conv(x.flatten(0, 1))  # TB C H W
        x = bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        if lif is not None:
            x = lif(x.flatten(0, 1)).reshape(T, B, -1, H//2, W//2).contiguous()
        return x

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape

        x = self.ConvBnSn(x,self.proj_conv,self.proj_bn,self.proj_lif)
        x = self.scs_block(x.flatten(0,1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)
        x = self.scs_block1(x.flatten(0,1)).reshape(T, B, -1, H // 4, W // 4).contiguous()

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.scs_block2(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8).contiguous()

        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,self.proj_lif3)
        x = self.scs_block3(x.flatten(0,1))

        x_feat = x.reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.ConvBnSn(x, self.rpe_conv, self.rpe_bn, self.rpe_lif)

        x = x + x_feat
        x = x.flatten(-2) # T B C N
        return x

#Spike-Driven Transformer SPS
class SPS_sdt(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,node=st_LIFNode,tau=2.0, act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape
        assert self.embed_dims % 8 == 0, 'embed_dims must be divisible by 8 in Spikformer'

        x = self.ConvBnSn(x,self.proj_conv,self.proj_bn,self.proj_lif)
        x = self.maxpool(x.flatten(0,1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)
        x = self.maxpool1(x.flatten(0,1)).reshape(T, B, -1, H // 4, W // 4).contiguous()

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.maxpool2(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8).contiguous()

        # maxpool -> lif
        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,None)
        x = self.maxpool3(x.flatten(0,1))
        x = self.proj_lif3(x).reshape(T, B, -1, H // 16, W // 16).contiguous()

        x_feat = x
        x = self.ConvBnSn(x,self.rpe_conv,self.rpe_bn,None)

        return x+x_feat # T B -1 H/16 W/16

# QKFormer (Nips 2024)
class PEDS_init(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type,embed_dims=embed_dims)
        del self.rpe_conv, self.rpe_bn, self.rpe_lif # less param
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)
    # Conv - BN - MP - LIF  in PEDS
    def ConvBnMpSN(self, x, conv, bn, mp, lif):
        T, B, C, H, W = x.shape
        x = conv(x.flatten(0, 1))  # TB C H W
        x = bn(x).reshape(T, B, -1, H, W).contiguous()
        if mp is not None:
            x = mp(x.flatten(0,1)).reshape(T, B, -1, H//2, W//2).contiguous()
        if lif is not None:
            _,_,_,new_h, new_w = x.shape
            x = lif(x.flatten(0, 1)).reshape(T, B, -1, new_h, new_w).contiguous()
        return x
    def forward(self, x):
        self.reset()
        assert self.embed_dims % 8 == 0, 'embed_dims must be divisible by 8 in Spikformer'

        x = self.ConvBnMpSN(x,self.proj_conv,self.proj_bn,None,self.proj_lif)
        x = self.ConvBnMpSN(x,self.proj_conv1,self.proj_bn1,self.maxpool1,self.proj_lif1)

        x_feat = x
        x = self.ConvBnMpSN(x,self.proj_conv2,self.proj_bn2,self.maxpool2,self.proj_lif2)
        x = self.ConvBnMpSN(x,self.proj_conv3,self.proj_bn3,self.maxpool3,self.proj_lif3)

        T, B, C, H, W = x.shape
        x_feat = self.proj_res_conv(x_feat.flatten(0,1))
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, C, H, W).contiguous()
        x_feat = self.proj_res_lif(x_feat.flatten(0,1)).reshape(T, B, C, H, W).contiguous()

        return x+x_feat  # T B Dim H//8 W//8

# QKFormer (Nips 2024)
class PEDS_stage(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=2,
                 embed_dims=256,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        del self.rpe_conv, self.rpe_bn, self.rpe_lif # less param
        del self.proj_conv, self.proj_bn, self.proj_lif
        del self.proj_conv1, self.proj_bn1, self.proj_lif1
        del self.proj_conv2, self.proj_bn2, self.proj_lif2
        del self.proj_conv3, self.proj_bn3, self.proj_lif3

        self.proj_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_conv4 = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn4 = nn.BatchNorm2d(embed_dims)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj_lif4 = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_res_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)
    # Conv - BN - MP - LIF  in PEDS
    def ConvBnMpSN(self, x, conv, bn, mp, lif):
        T, B, C, H, W = x.shape
        x = conv(x.flatten(0, 1))  # TB C H W
        x = bn(x).reshape(T, B, -1, H, W).contiguous()
        if mp is not None:
            x = mp(x.flatten(0,1)).reshape(T, B, -1, H//2, W//2).contiguous()
        if lif is not None:
            _,_,_,new_h, new_w = x.shape
            x = lif(x.flatten(0, 1)).reshape(T, B, -1, new_h, new_w).contiguous()
        return x
    def forward(self, x):
        self.reset()
        T, B, _, _, _ = x.shape
        assert self.embed_dims % 8 == 0, 'embed_dims must be divisible by 8 in Spikformer'
        x_feat = x

        x = self.ConvBnMpSN(x,self.proj_conv,self.proj_bn,None,self.proj_lif)
        x = self.ConvBnMpSN(x,self.proj_conv4,self.proj_bn4,self.maxpool4,self.proj_lif4)


        x_feat = self.proj_res_conv(x_feat.flatten(0, 1))
        _, c_f, h_f, w_f = x_feat.shape
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, c_f, h_f, w_f).contiguous()
        x_feat = self.proj_res_lif(x_feat.flatten(0, 1)).reshape(T, B, c_f, h_f, w_f).contiguous()


        return x+x_feat  # T B Dim H//2 W//2




'''
    Spiking Transformer Attentions
'''
# Spikformer（ICLR 2023）
class SSA(BaseModule):
    #num_heads: 16 for dvsc10, 8 for dvsg
    def __init__(self,embed_dim, step=10,encode_type='direct',num_heads=16,scale=0.25,attn_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super(SSA, self).__init__(step=step,encode_type=encode_type)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = scale
        self.attn_drop_rate = attn_drop

        self.q_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.k_conv = nn.Conv1d(embed_dim,embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.v_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.res_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)
        self.attn_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
    def qkv(self,x,conv,bn,lif,num_heads=16):
        if len(x.shape) == 5:
            x = x.flatten(-2,-1)
        T, B, C, N = x.shape
        r = conv(x.flatten(0,1))  # TB C N
        r = bn(r).reshape(T, B, C, N).contiguous()  # T B C N
        r = lif(r.flatten(0, 1)).reshape(T, B, C, N)  # T B C N
        return r.transpose(-2,-1).reshape(T, B, N, num_heads, C // num_heads).permute(0, 1, 3, 2, 4).contiguous() #T B H N C/H

    def attn_cal(self,q,k,v):
        T, B, H, N, CoH = q.shape  # CoH： C/H
        C = CoH * H

        attn = (q @ k.transpose(-2, -1)) * self.scale
        r = (attn @ v) * self.scale
        if(self.attn_drop_rate>0):
            r = self.attn_drop(attn)
        r = r.transpose(3, 4).reshape(T, B, C, N).contiguous()  # T B C N
        r = self.attn_lif(r.flatten(0, 1))  # TB C N
        return self.proj_lif(self.proj_bn(self.proj_conv(r))).reshape(T, B, C, N)  # T B C N


    def forward(self, x):
        self.reset()

        q = self.qkv(x,self.q_conv,self.q_bn,self.q_lif,self.num_heads)
        k = self.qkv(x,self.k_conv,self.k_bn,self.k_lif,self.num_heads)
        v = self.qkv(x,self.v_conv,self.v_bn,self.v_lif,self.num_heads)

        x = self.attn_cal(q,k,v)

        return x

# TIM（IJCAI 2024）
class TIM(BaseModule):
    def __init__(self,embed_dim,num_heads, encode_type='direct',TIM_alpha=0.5,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=1, encode_type=encode_type)
        self.in_channels = embed_dim // num_heads
        #  channels may depends on the shape of input
        self.TIM_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=5, stride=1,
                                    padding=2, bias=True)
        self.in_lif = node(step=1,tau=tau,act_func=act_func,threshold=threshold)  # spike-driven
        self.out_lif = node(step=1,tau=tau,act_func=act_func,threshold=threshold)  # spike-driven
        self.tim_alpha = TIM_alpha

    # input [T, B, H, N, C/H]
    def forward(self, x):
        self.reset()
        T, B, H, N, CoH = x.shape
        output = []
        x_tim = torch.empty_like(x[0])
        # temporal interaction
        for i in range(T):
            # 1st step
            if i == 0:
                x_tim = x[i]
                output.append(x_tim)
            # other steps
            else:
                x_tim = self.TIM_conv(x_tim.flatten(0, 1).transpose(-2,-1)).transpose(-2,-1).reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                output.append(x_tim)

        return torch.stack(output)  # T B H, N, C/H
# TIM (IJCAI 2024)
class SSA_TIM(SSA):
    def __init__(self,embed_dim, num_heads, TIM_alpha=0.5, step=10, encode_type='direct', scale=0.25):
        super(SSA_TIM, self).__init__(embed_dim, num_heads=num_heads, step=step, encode_type=encode_type, scale=scale)
        self.tim_alpha = TIM_alpha
        self.tim = TIM(embed_dim, num_heads, encode_type=encode_type, TIM_alpha=TIM_alpha)

    def forward(self, x):
        self.reset()

        q = self.qkv(x,self.q_conv,self.q_bn,self.q_lif,self.num_heads)
        k = self.qkv(x,self.k_conv,self.k_bn,self.k_lif,self.num_heads)
        v = self.qkv(x,self.v_conv,self.v_bn,self.v_lif,self.num_heads)

        q = self.tim(q)

        x = self.attn_cal(q,k,v)

        return x

    def attn_cal(self,q,k,v):
        kv = k.mul(v)  # piont-wise multiplication

# Spike-driven Transformer (Nips 2024)
class SDSA(SSA):
    def __init__(self,embed_dim, step=10,encode_type='direct',num_heads=16,scale=0.25,attn_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super(SDSA, self).__init__(embed_dim, num_heads=num_heads, step=step, encode_type=encode_type, scale=scale)
        self.shortcut_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.q_conv = nn.Conv2d(embed_dim,embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(embed_dim)

        self.k_conv = nn.Conv2d(embed_dim,embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(embed_dim)

        self.v_conv = nn.Conv2d(embed_dim,embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(embed_dim)

        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

        self.kv_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)
    def qkv(self,x,conv,bn,lif,num_heads=None):
        T, B, C, H, W = x.shape
        r = conv(x.flatten(0,1))  # TB C N
        r = bn(r).reshape(T, B, C, H, W ).contiguous()  # T B C N
        r = lif(r.flatten(0, 1)).reshape(T, B, C, H, W ).contiguous()  # T B C H W
        return r

    def sdsa_cal(self,q,k,v,lif):
        T, B, H, N, CoH = q.shape  # CoH： C/H
        C = CoH * H

        kv = k.mul(v) # point-wise multiplication
        kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = lif(kv.flatten(0,1)).reshape(T, B, H ,-1, CoH).contiguous()
        return q.mul(kv)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape
        N = H * W
        x_id = x
        x = self.shortcut_lif(x.flatten(0,1)).reshape(T, B, C, H, W).contiguous()

        q = self.qkv(x,self.q_conv,self.q_bn,self.q_lif)
        q = q.flatten(-2,-1).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k = self.qkv(x,self.k_conv,self.k_bn,self.k_lif)
        k = self.pool(k).flatten(-2,-1).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v = self.qkv(x,self.v_conv,self.v_bn,self.v_lif)
        v = self.pool(v).flatten(-2,-1).transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        x = self.sdsa_cal(q,k,v,self.kv_lif)
        x = self.pool(x).transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        return x + x_id


# QKFormer (Nips 2024)
class QKTA(SSA):
    # No Scale here!
    # num_heads: 8 for dvsc10 in original code
    def __init__(self,embed_dim, step=10,encode_type='direct',num_heads=16,attn_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super(QKTA, self).__init__(embed_dim=embed_dim,step=step,encode_type=encode_type)
        del self.v_lif, self.v_conv, self.v_bn,self.res_lif   # less param
        del self.pool

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn_drop_rate = attn_drop

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape

        q = self.qkv(x,self.q_conv,self.q_bn,self.q_lif,self.num_heads) #T B H N C/H
        k = self.qkv(x,self.k_conv,self.k_bn,self.k_lif,self.num_heads)

        q = torch.sum(q, dim=3, keepdim=True)
        _, _, q_h, q_n, q_c = q.shape
        attn = self.attn_lif(q.flatten(0,1)).reshape(T, B, q_h, q_n, q_c).contiguous()
        x = torch.mul(attn, k) #torch.Size([10, 16, 16, 256, 8])

        x = x.transpose(-2,-1).flatten(2,3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x.flatten(0,1)).reshape(T, B, C, H, W).contiguous()

        return x




'''
    Spiking Transformer MLPs
'''
class MLP(BaseModule):
    def __init__(self, in_features, step=10, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.MLP_drop = nn.Dropout(self.mlp_drop)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

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

# Spike-driven Transformer MLP
# Shortcut
class Sdt_MLP(BaseModule):

    def __init__(self, in_features, step=10, encode_type='direct', mlp_ratio=2.0, out_features=None, mlp_drop=0.,
                 node=st_LIFNode, tau=2.0, act_func=Sigmoid_Grad, threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop

        self.fc1_lif = node(step=step, tau=tau, act_func=act_func, threshold=threshold)
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)

        self.MLP_drop = nn.Dropout(self.mlp_drop)

        self.fc2_lif = node(step=step, tau=tau, act_func=act_func, threshold=threshold)
        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        self.reset()
        x_id = x
        T, B, C, H, W = x.shape
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, -1, H, W).contiguous()

        if self.mlp_drop > 0:
            x = self.MLP_drop(x)
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        return x + x_id
'''
    Spiking Transformer OTHER Useful Blocks
'''
# Spikformer block
class Spikf_Block(nn.Module):
    """
    :param: if_TIM: if use Temporal Interaction Module(IJCAI2024)
    """
    def __init__(self, embed_dim=256, num_heads=16, step=10, mlp_ratio=4., scale=0., attn_drop=0.,mlp_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5,if_TIM=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        if if_TIM:
            self.attn = SSA_TIM(embed_dim, num_heads=num_heads, step=step, scale=scale)
        else:
            self.attn = SSA(embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop, scale=scale,node=node,tau=tau,act_func=act_func,threshold=threshold)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,node=node,tau=tau,act_func=act_func,threshold=threshold)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# Spike-driven Transformer block
class Sdt_Block(nn.Module):
    def __init__(self, embed_dim=256, num_heads=16, step=10, mlp_ratio=4., scale=0., attn_drop=0., mlp_drop=0.,
                 node=st_LIFNode, tau=2.0, act_func=Sigmoid_Grad, threshold=0.5, if_TIM=False):
        super().__init__()
        self.attn = SDSA(embed_dim, num_heads=num_heads, step=step, scale=scale)
        self.mlp = Sdt_MLP(in_features=embed_dim, mlp_ratio=mlp_ratio, mlp_drop=mlp_drop, node=node, tau=tau,
                           act_func=act_func, threshold=threshold)

    def forward(self, x):
        # residual completed in shortcut calculation
        x_attn = self.attn(x)
        return self.mlp(x_attn)


class cls_head(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(cls_head, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.head(x)
        return x
