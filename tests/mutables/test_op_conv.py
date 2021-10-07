import torch
from hyperbox.mutables import ops
from hyperbox.mutables.spaces import ValueSpace
from hyperbox.mutator import RandomMutator

from hyperbox.utils.calc_model_size import flops_size_counter

if __name__ == '__main__':
    x = torch.rand(1,3,64,64)
    vs1 = ValueSpace([10,2])
    vs2 = ValueSpace([3,5,7])
    conv = ops.Conv2d(3, vs1, vs2,bias=False)
    bn = ops.BatchNorm2d(vs1)
    op = torch.nn.Sequential(
        # torch.nn.Conv2d(3,3,3,1),
        conv, bn
    )
    m = RandomMutator(op)
    m.reset()
    print(op)
    print(m._cache)
    print(conv.weight.shape)
    print(conv(x).shape)
    r = flops_size_counter(op, (1,3,8,8), True, True)
    print(r)
    op = torch.nn.Sequential(
        ops.Conv2d(3,8,3,1),
        ops.BatchNorm2d(8)
    )
    r = flops_size_counter(op, (1,3,8,8), False, True)
    print(conv(x).shape)