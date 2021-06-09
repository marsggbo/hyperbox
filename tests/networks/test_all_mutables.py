import torch
import torch.nn as nn

import mutables, mutator

class Net(nn.Module):
    def __init__(self,):
        super().__init__()
        ops = [
            nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(3,4,kernel_size=5,stride=1,padding=2),
            nn.Conv2d(3,4,kernel_size=7,stride=1,padding=3),
            nn.Identity()
        ]
        self.candidate_op1 = mutables.LayerChoice(ops, key='candidate1')
        self.candidate_op2 = mutables.LayerChoice(ops, key='candidate2')
        
        v1 = mutables.ValueChoice([4,8,16])
        self.fop1 = mutables.FinegrainedConv2d(4, v1, kernel_size=3,stride=1,padding=1)
        self.fop2 = mutables.FinegrainedConv2d(v1, 8, kernel_size=3,stride=1,padding=1)
        
        self.input_op = mutables.InputChoice(n_candidates=2, n_chosen=1, key='input1')

    def forward(self, x):
        out1 = self.candidate_op1(x)
        out2 = self.candidate_op2(x)
        
        out = self.input_op([out1, out2])
        out = self.fop1(out)
        out = self.fop2(out)
        return out

if __name__ == '__main__':
    net = Net()
    random = mutator.RandomMutator(net)
    random.reset()
    x = torch.rand(2,3,16,16)
    y = net(x)
    print(y.shape)
        