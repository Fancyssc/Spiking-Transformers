from braincog.base.node.node import *
from braincog.base.connection.layer import *
from st_utils.grad.st_grad import *


# Spiking Transformers prefer layer-by-layer processing

class lbl_BaseNode(BaseNode):
    def __init__(self, threshold=0.5, step=10, layer_by_layer=True, mem_detach=True):
        super().__init__(threshold=threshold, step=step, layer_by_layer=layer_by_layer, mem_detach=mem_detach)

    def rearrange2node(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, 'b (c t) w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, 'b (c t) -> t b c', t=self.step)
            else:
                raise NotImplementedError

        elif self.layer_by_layer:
            if len(inputs.shape) == 4:
                outputs = rearrange(inputs, '(t b) c w h -> t b c w h', t=self.step)
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, '(t b) n c -> t b n c', t=self.step)
            elif len(inputs.shape) == 2:
                outputs = rearrange(inputs, '(t b) c -> t b c', t=self.step)
            else:
                raise NotImplementedError


        else:
            outputs = inputs

        return outputs

    def rearrange2op(self, inputs):
        if self.groups != 1:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> b (c t) w h')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> b (c t)')
            else:
                raise NotImplementedError
        elif self.layer_by_layer:
            if len(inputs.shape) == 5:
                outputs = rearrange(inputs, 't b c w h -> (t b) c w h')

            # 加入适配Transformer T B N C的rearange2op分支
            elif len(inputs.shape) == 4:
                outputs = rearrange(inputs, ' t b n c -> (t b) n c')
            elif len(inputs.shape) == 3:
                outputs = rearrange(inputs, ' t b c -> (t b) c')
            else:
                raise NotImplementedError

        else:
            outputs = inputs

        return outputs

# Default： Layer by layer LIFNode with Sigmoid Grad
class st_LIFNode(lbl_BaseNode):
    def __init__(self, threshold=1., step=4, layer_by_layer=True, tau=2., act_fun=Sigmoid_Grad,mem_detach=True, *args,
                 **kwargs):
        super().__init__(threshold=threshold, step=step, layer_by_layer=layer_by_layer, mem_detach=mem_detach)
        self.tau = tau
        if isinstance(act_fun, str):
            act_fun = eval(act_fun)
        self.act_fun = act_fun()

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold)
        self.mem = self.mem * (1 - self.spike.detach())