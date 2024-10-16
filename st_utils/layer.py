from functools import partial
from braincog.model_zoo.base_module import BaseModule
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
    def ConvBnSn(self,x,conv,bn,lif):
        T, B, C, H, W = x.shape
        x = conv(x.flatten(0,1)) # TB C H W
        x = bn(x).reshape(T, B, -1, H, W).contiguous()
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
        x = self.maxpool(x.flatten(0,1)).reshape(T, B, -1, H // 2, W // 2)

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)
        x = self.maxpool1(x.flatten(0,1)).reshape(T, B, -1, H // 4, W // 4)

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.maxpool2(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8)

        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,self.proj_lif3)
        x = self.maxpool3(x.flatten(0,1)).reshape(T, B, -1, H // 16, W // 16)

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
    def ConvBnSn(self, x, conv, bn, lif):
        T, B, C, H, W = x.shape
        x = conv(x.flatten(0, 1))  # TB C H W
        x = bn(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = lif(x.flatten(0, 1)).reshape(T, B, -1, H//2, W//2).contiguous()
        return x

    def forward(self, x):
        self.reset()
        T, B, C, H, W = x.shape

        x = self.ConvBnSn(x,self.proj_conv,self.proj_bn,self.proj_lif)
        x = self.scs_block(x.flatten(0,1)).reshape(T, B, -1, H // 2, W // 2)

        x = self.ConvBnSn(x,self.proj_conv1,self.proj_bn1,self.proj_lif1)
        x = self.scs_block1(x.flatten(0,1)).reshape(T, B, -1, H // 4, W // 4)

        x = self.ConvBnSn(x,self.proj_conv2,self.proj_bn2,self.proj_lif2)
        x = self.scs_block2(x.flatten(0,1)).reshape(T, B, -1, H // 8, W // 8)

        x = self.ConvBnSn(x,self.proj_conv3,self.proj_bn3,self.proj_lif3)
        x = self.scs_block3(x.flatten(0,1))

        x_feat = x.reshape(T, B, -1, H // 16, W // 16).contiguous()
        x = self.ConvBnSn(x, self.rpe_conv, self.rpe_bn, self.rpe_lif)

        x = x + x_feat
        x = x.flatten(-2) # T B C N
        return x



'''
    Spiking Transformer Attentions
'''
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

    def qkv(self,x,conv,bn,lif):
        T, B, C, N = x.shape
        r = conv(x.flatten(0,1))  # TB C N
        r = bn(r).reshape(T, B, C, N).contiguous()  # T B C N
        r = lif(r.flatten(0, 1)).reshape(T, B, C, N)  # T B C N
        return r.transpose(-2,-1).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous() #T B H N C/H

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

        q = self.qkv(x,self.q_conv,self.q_bn,self.q_lif)
        k = self.qkv(x,self.k_conv,self.k_bn,self.k_lif)
        v = self.qkv(x,self.v_conv,self.v_bn,self.v_lif)

        x = self.attn_cal(q,k,v)

        return x


class TIM(BaseModule):
    def __init__(self,embed_dim,num_heads, encode_type='direct',TIM_alpha=0.5,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__(step=1, encode_type=encode_type)
        self.in_channels = embed_dim // num_heads
        #  channels may depends on the shape of input
        self.TIM_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=5, stride=1,
                                    padding=2, bias=True)
        self.in_lif = node(tau=tau,act_func=act_func,threshold=threshold)  # spike-driven
        self.out_lif = node(tau=tau,act_func=act_func,threshold=threshold)  # spike-driven
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
                x_tim = self.TIM_conv(x_tim.flatten(0, 1)).reshape(B, H, N, CoH).contiguous()
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1 - self.tim_alpha)
                x_tim = self.out_lif(x_tim)
                output.append(x_tim)

        return torch.stack(output)  # T B H, N, C/H
class SSA_TIM(SSA):
    def __init__(self,embed_dim, num_heads, TIM_alpha=0.5, step=10, encode_type='direct', scale=0.25):
        super(SSA_TIM, self).__init__(embed_dim, num_heads=num_heads, step=step, encode_type=encode_type, scale=scale)
        self.tim_alpha = TIM_alpha
        self.tim = TIM(embed_dim, num_heads, encode_type=encode_type, TIM_alpha=TIM_alpha)
    def forward(self, x):
        self.reset()
        x_for_qkv = x.flatten(0, 1)  # TB, C N

        q = self.qkv(x_for_qkv,self.q_conv,self.q_bn,self.q_lif)
        k = self.qkv(x_for_qkv,self.k_conv,self.k_bn,self.k_lif)
        v = self.qkv(x_for_qkv,self.v_conv,self.v_bn,self.v_lif)

        q = self.TIM(q)

        x = self.attn_cal(q,k,v)

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

'''
    Spiking Transformer OTHER Useful Blocks
'''
class Spikf_Block(nn.Module):
    def __init__(self, embed_dim=256, num_heads=16, step=10, mlp_ratio=4., scale=0., attn_drop=0.,mlp_drop=0.,node=st_LIFNode,tau=2.0,act_func=Sigmoid_Grad,threshold=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SSA(embed_dim, step=step, num_heads=num_heads,attn_drop=attn_drop, scale=scale,node=node,tau=tau,act_func=act_func,threshold=threshold)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim,mlp_ratio=mlp_ratio,out_features=embed_dim,mlp_drop=mlp_drop,node=node,tau=tau,act_func=act_func,threshold=threshold)
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

