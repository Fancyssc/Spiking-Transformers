from braincog.model_zoo.base_module import BaseModule
from st_utils.widgets.TIM import *
from st_utils.node.st_LIFNode import *

class cls_head(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(cls_head, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.head(x)
        return x