from braincog.model_zoo.base_module import BaseModule
from st_utils.widgets.TIM import *
from st_utils.node.st_LIFNode import *

class SSA(BaseModule):
    #num_heads: 16 for dvsc10, 8 for dvsg
    def __init__(self,embed_dim, step=10,encode_type='direct',num_heads=16,scale=0.25,attn_drop=0,**kwargs):
        super(SSA, self).__init__(step=step,encode_type=encode_type)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = scale
        self.attn_drop_rate = attn_drop

        self.q_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = st_LIFNode(step=step)

        self.k_conv = nn.Conv1d(embed_dim,embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = st_LIFNode(step=step)

        self.v_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = st_LIFNode(step=step)

        self.attn_drop = nn.Dropout(self.attn_drop)
        self.res_lif = st_LIFNode(step=step)
        self.attn_lif = st_LIFNode(step=step)

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = st_LIFNode(step=step)

    def qkv(self,x,conv,bn,lif):
        T, B, C, N = x.shape
        r = conv(x)  # TB C N
        r = bn(r).reshape(T, B, C, N).contiguous()  # T B C N
        r = lif(r.flatten(0, 1)).reshape(T, B, C, N)  # TB C N
        return r.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

    def attn_cal(self,q,k,v):
        T, B, N, C = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.scale
        r = (attn @ v) * self.scale
        if(self.attn_drop_rate>0):
            r = self.attn_drop(attn)
        r = r.transpose(3, 4).reshape(T, B, C, N).contiguous()  # T B C N
        r = self.attn_lif(r.flatten(0, 1))  # TB C N
        return self.proj_lif(self.proj_bn(self.proj_conv(r))).reshape(T, B, C, N)  # T B C N


    def forward(self, x):
        self.reset()
        x_for_qkv = x.flatten(0, 1)  # TB, C N

        q = self.qkv(x_for_qkv,self.q_conv,self.q_bn,self.q_lif)
        k = self.qkv(x_for_qkv,self.k_conv,self.k_bn,self.k_lif)
        v = self.qkv(x_for_qkv,self.v_conv,self.v_bn,self.v_lif)

        x = self.attn_cal(q,k,v)

        return x

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



