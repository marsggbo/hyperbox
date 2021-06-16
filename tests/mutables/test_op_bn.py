import torch
from hyperbox.mutables import ops
from hyperbox.mutables.spaces import ValueSpace
from hyperbox.mutator import RandomMutator

if __name__ == '__main__':
    x = torch.rand(4,3)
    print('Normal case')
    linear = ops.Linear(3,10)
    bn = ops.BatchNorm1d(10)
    y = linear(x)
    print(y.shape)
    y = bn(y)
    print(y.shape)

    print('Search case')
    vs1 = ValueSpace([1,2,3])
    linear = ops.Linear(3, vs1, bias=False)
    bn = ops.BatchNorm1d(vs1)
    m = RandomMutator(linear)
    m.reset()
    print(m._cache)
    print(linear.weight.shape)
    y = linear(x)
    print(y.shape)
    y = bn(y)
    print(y.shape)
    