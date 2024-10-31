from functools import partial
from braincog.model_zoo.base_module import BaseModule
from pandas.compat.numpy.function import validate_take_with_convert

from static_utils.node import *
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
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=3,
                 embed_dims=384,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
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
class SPSv1_s(SPS):
    def __init__(self, step=10, encode_type='direct', img_h=128, img_w=128, patch_size=16, in_channels=3,
                 embed_dims=384,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type,embed_dims=embed_dims,tau=tau,act_func=act_func,
                         threshold=threshold,img_h=img_h, img_w=img_w, patch_size=patch_size, in_channels=in_channels,node=node)
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        # self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape
        assert self.embed_dims % 8 == 0, 'embed_dims must be divisible by 8 in Spikformer'

        x = self.ConvBnSn(x,self.proj_conv,self.proj_bn,self.proj_lif)

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.maxpool2(x.flatten(0,1)).reshape(T, B, -1, H // 2, W // 2).contiguous()

        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,self.proj_lif3)
        x = self.maxpool3(x.flatten(0,1)).reshape(T, B, -1, H // 4, W // 4).contiguous()

        x_feat = x.reshape(T, B, -1, H // 4, W // 4).contiguous()
        x = self.ConvBnSn(x,self.rpe_conv,self.rpe_bn,self.rpe_lif)

        x = x + x_feat
        x = x.flatten(-2).transpose(-1,-2)  # T B N C
        return x


'''
    Spiking Transformer Attentions
'''
# done
class SSA_s(BaseModule):
    #num_heads: 16 for dvsc10, 8 for dvsg
    def __init__(self,embed_dim, step=4,encode_type='direct',num_heads=16,scale=0.125,attn_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super(SSA_s, self).__init__(step=step,encode_type=encode_type)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = scale
        self.attn_drop_rate = attn_drop

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.attn_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.proj_linear = nn.Linear(embed_dim, embed_dim)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

    def qkv(self,x,linear,bn,lif,num_heads=16):
        if len(x.shape) == 5:
            x = x.flatten(-2,-1)
        T, B, N, C = x.shape
        r = linear(x.flatten(0,1))  # TB N C
        r = bn(r.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()  # T B N C
        r = lif(r.flatten(0, 1)).reshape(T, B, N, C)  # T B N C
        return r.reshape(T, B, N, num_heads, C // num_heads).permute(0, 1, 3, 2, 4).contiguous() #T B H N C/H

    def attn_cal(self,q,k,v):
        T, B, H, N, CoH = q.shape  # CoH： C/H
        C = CoH * H

        attn = (q @ k.transpose(-2, -1)) * self.scale
        r = (attn @ v) * self.scale
        if(self.attn_drop_rate>0):
            r = self.attn_drop(attn)
        r = r.transpose(2, 3).reshape(T, B, N, C).contiguous()  # T B N C
        r = self.attn_lif(r.flatten(0, 1))  # TB N C
        r = self.proj_bn(self.proj_linear(r).transpose(-1,-2)).transpose(-1,-2)
        return self.proj_lif(r).reshape(T, B, N, C)  # T B N C


    def forward(self, x):
        self.reset()

        q = self.qkv(x,self.q_linear,self.q_bn,self.q_lif,self.num_heads)
        k = self.qkv(x,self.k_linear,self.k_bn,self.k_lif,self.num_heads)
        v = self.qkv(x,self.v_linear,self.v_bn,self.v_lif,self.num_heads)

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
class SSA_TIM(SSA_s):
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




'''
    Spiking Transformer MLPs
'''
# done
class MLP_s(BaseModule):
    def __init__(self, in_features, step=4, encode_type='direct', mlp_ratio = 4.0, out_features=None,mlp_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=step, encode_type=encode_type)
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.mlp_drop = mlp_drop
        self.fc1_linear = nn.Linear(in_features,hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

        self.MLP_drop = nn.Dropout(self.mlp_drop)

        self.fc2_linear = nn.Linear(hidden_features,out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = node(step=step,tau=tau,act_func=act_func,threshold=threshold)

    def forward(self, x):
        self.reset()

        T, B, N, C = x.shape

        x = self.fc1_linear(x.flatten(0,1))
        x = self.fc1_bn(x.transpose(-1,-2)).transpose(-1,-2).reshape(T, B, N,-1).contiguous()  # T B N C
        x = self.fc1_lif(x.flatten(0, 1)).reshape(T, B, N, -1).contiguous()

        if self.mlp_drop > 0 :
            x = self.MLP_drop(x)

        x = self.fc2_linear(x.flatten(0,1))
        x = self.fc2_bn(x.transpose(-1,-2)).transpose(-1,-2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x.flatten(0, 1)).reshape(T, B, N, C).contiguous()
        return x

'''
    Spiking Transformer OTHER Useful Blocks
'''
# Spikformer block
class Spikf_Block_s(nn.Module):
    """
    :param: if_TIM: if use Temporal Interaction Module(IJCAI2024)
    """
    def __init__(self, embed_dim=384, num_heads=12, step=4, mlp_ratio=4., scale=0., attn_drop=0.,mlp_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5,if_TIM=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        if if_TIM:
            self.attn = SSA_TIM(embed_dim, num_heads=num_heads, step=step, scale=scale)
        else:
            self.attn = SSA_s(embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop, scale=scale,node=node,tau=tau,act_func=act_func,threshold=threshold)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP_s(step=step,in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,node=node,tau=tau,act_func=act_func,threshold=threshold)
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class cls_head(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(cls_head, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.head(x)
        return x
