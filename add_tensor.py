from teenygrad import Tensor

x = Tensor([1, 2, 3, 4])
y = Tensor([3, 4, 5, 6])
z = x + y
print(z.numpy())

'''

Tensor (data is represented by LazyBuffer internally):

 def add(self, x:Union[Tensor, float], reverse=False) -> Tensor:
    x = self._to_float(x)
    return mlops.Add.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else self

class Function:
  def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
    ctx = fxn(x[0].device, *x)
    ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
    if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx    # used by autograd engine
    return ret

mlops:    

class Add(Function):
  def forward(self, x:LazyBuffer, y:LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.ADD, y)

ops:

class BinaryOps(Enum): ADD = auto();

lazy:

def e(self, op, *srcs:LazyBuffer):
    if op == BinaryOps.ADD: ret = self._np + srcs[0]._np # adding two LazyBuffers together
    # no custom backend in teenygrad so just use numpy representations

'''