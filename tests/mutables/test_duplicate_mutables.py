import torch
import torch.nn as nn

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutator import RandomMutator


class ToyDuplicateNet(BaseNASNetwork):
    def __init__(self):
        super().__init__()
        self.op1 = OperationSpace(candidates=[nn.ReLU(), nn.Sigmoid()], key='op')
        self.op2 = OperationSpace(candidates=[nn.ReLU(), nn.Sigmoid()], key='op')
    
    def forward(self, x):
        out1 = self.op1(x)
        out2 = self.op2(x)
        out = out1 - out2
        return out

    @property
    def arch(self):
        return f"op1:{self.op1.mask}-op2:{self.op2.mask}"


if __name__ == '__main__':
    model = ToyDuplicateNet()
    mutator = RandomMutator(model)
    x = torch.rand(10)
    for i in range(3):
        mutator.reset()
        print(model.arch)
        y = model(x)
        print(y)
        print('='*20)