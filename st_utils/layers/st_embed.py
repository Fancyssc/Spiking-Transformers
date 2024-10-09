import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule

from st_utils.grad import *
from st_utils.node import *

# Spikformer SPS
class SPS(BaseModule):
    def __init__(self, step=10, encode_type='direct', img_size_h=32, img_size_w=32, patch_size=4, in_channels=3,
                 embed_dims=384):
        super().__init__(step=step, encode_type=encode_type)