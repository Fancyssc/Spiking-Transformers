from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *



class Sigmoid_Grad(SurrogateFunctionBase):
    """
    Sigmoid activation function with gradient
    Overwrite sigmoid function in BrainCog
    """
    def __init__(self, alpha=4., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid.apply(x, alpha)


class QGate_Grad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return quadratic_gate.apply(x, alpha)

class lbl_BaseNode(BaseNode):
    """
    Base Node for Layer by Layer forward propagation
    New OP added to adapt specific dim needed by Spiking Transformers
    :param threshold: The threshold that a neuron needs to reach in order to fire an action potential.
    :param step: The number of time steps that the neuron will be simulated for.
    """
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

class st_LIFNode(lbl_BaseNode):
    """
    Leaky Integrate-and-Fire (LIF) neuron model
    :param threshold: The threshold that a neuron needs to reach in order to fire an action potential.
    :param step: The number of time steps that the neuron will be simulated for.
    :param tau: The time constant of the neuron.
    :param act_fun: The activation function of the neuron.
    """
    def __init__(self, threshold=1., step=10, layer_by_layer=True, tau=2., act_fun=Sigmoid_Grad,mem_detach=True, *args,
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


# Create your self defined node below by following code structure in Braicog