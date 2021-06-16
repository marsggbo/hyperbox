import torch
from hyperbox.mutables import ops
from hyperbox.mutables.spaces import ValueSpace
from hyperbox.mutator import RandomMutator

from hyperbox.utils.calc_model_size import flops_size_counter

if __name__ == '__main__':
    x = torch.rand(1,3)
    vs1 = ValueSpace([1,2,3])
    linear = ops.Linear(3, vs1, bias=False)
    m = RandomMutator(linear)
    m.reset()
    print(m._cache)
    print(linear.weight.shape)
    print(linear(x).shape)
    r = flops_size_counter(linear, (2,3), False, True)
    
    linear = ops.Linear(3,10)
    print(linear(x).shape)
    r = flops_size_counter(linear, (2,3), False, True)