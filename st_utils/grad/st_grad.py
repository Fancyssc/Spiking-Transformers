from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *


#Alpha of Sigmoid Grad is set to 1 in Braincog
#Here to keep the same settings as Spikingjelly
class Sigmoid_Grad(SurrogateFunctionBase):
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