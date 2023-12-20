# write sigmoid mlop using ops

import numpy as np
from teenygrad.lazy import LazyBuffer
from teenygrad.tensor import Function
from teenygrad.ops import UnaryOps, BinaryOps
import math

'''

class Function:
  def __init__(self, device:str, *tensors:Tensor):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if self.requires_grad: self.parents = tensors

  def forward(self, *args, **kwargs): raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, *args, **kwargs): raise RuntimeError(f"backward not implemented for {type(self)}")

  @classmethod
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx    # used by autograd engine
    return ret

class UnaryOps(Enum): NOOP = auto(); EXP2 = auto(); LOG2 = auto(); CAST = auto(); SIN = auto(); SQRT = auto(); RECIP = auto(); NEG = auto() # noqa: E702
class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto(); MAX = auto(); MOD = auto(); CMPLT = auto() # noqa: E702
class ReduceOps(Enum): SUM = auto(); MAX = auto() # noqa: E702
class TernaryOps(Enum): MULACC = auto(); WHERE = auto() # noqa: E702
class MovementOps(Enum): RESHAPE = auto(); PERMUTE = auto(); EXPAND = auto(); PAD = auto(); SHRINK = auto(); STRIDE = auto() # noqa: E702
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto() # noqa: E702

'''


x = LazyBuffer(np.array([-2., -1., 0., 1., 2., 3., 4.]))

class Sigmoid(Function): # 1/(1+e^-x)
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        # negate
        # should use self.ret
        # x = x.e(UnaryOps.NEG)
        # exponentiate
        self.res = x.e(BinaryOps.DIV, x.const(-math.log(2)))
        self.res = self.res.e(UnaryOps.EXP2)
        # add 1
        self.res = self.res.e(BinaryOps.ADD, x.const(1))
        # reciprocal
        self.res = self.res.e(UnaryOps.RECIP)
        return self.res
    
    # sigmoid * (1 - sigmoid)
    # self.res = sigmoid
    def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
        # (1 - sigmoid)
        ret = self.res.const(1).e(BinaryOps.SUB, self.res)
        # sigmoid *
        ret = ret.e(BinaryOps.MUL, self.res)
        # multiply grad_output by derivative
        ret = ret.e(BinaryOps.MUL, grad_output)
        return ret

print(x._np)
print(Sigmoid(device=None).forward(x)._np)


# SELF.RET TO STORE RESULT FOR BACKWARD()
# more optimized because you can get rid of negation op
class Sigmoid2(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.const(1).e(BinaryOps.DIV, x.const(1).e(BinaryOps.ADD, x.e(BinaryOps.MUL, x.const(-1/math.log(2))).e(UnaryOps.EXP2)))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.e(BinaryOps.MUL, self.ret.const(1).e(BinaryOps.SUB, self.ret)).e(BinaryOps.MUL, grad_output)
  
# print(Sigmoid2(device=None).forward(x)._np)ß