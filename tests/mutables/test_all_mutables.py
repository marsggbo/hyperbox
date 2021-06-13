import torch
import torch.nn as nn

from hyperbox.mutator import RandomMutator
from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.mutables.ops import Conv2d


class Net(nn.Module):
    def __init__(self,):
        super().__init__()
        ops = [
            nn.Conv2d(3,4,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(3,4,kernel_size=5,stride=1,padding=2),
            nn.Conv2d(3,4,kernel_size=7,stride=1,padding=3),
            nn.Identity()
        ]
        self.candidate_op1 = OperationSpace(ops, key='candidate1')
        self.candidate_op2 = OperationSpace(ops, key='candidate2')
        
        v1 = ValueSpace([4,8,16])
        self.fop1 = Conv2d(4, v1, kernel_size=3,stride=1,padding=1)
        self.fop2 = Conv2d(v1, 8, kernel_size=3,stride=1,padding=1)
        
        self.input_op = InputSpace(n_candidates=2, n_chosen=1, key='input1')

    def forward(self, x):
        out1 = self.candidate_op1(x)
        out2 = self.candidate_op2(x)
        
        out = self.input_op([out1, out2])
        out = self.fop1(out)
        out = self.fop2(out)
        return out

if __name__ == '__main__':
    net = Net()
    random = RandomMutator(net)
    random.reset()
    x = torch.rand(2,3,16,16)
    y = net(x)
    print(y.shape)


    # test OperationSpace
    x = torch.rand(2,10)
    ops = [
        nn.Linear(10,10,bias=False),
        nn.Linear(10,100,bias=False),
        nn.Linear(10,100,bias=False),
        nn.Identity()
    ]
    mixop = OperationSpace(
        ops,
        mask=[0,1,0,0]
    )
    y = mixop(x)
    print(y.shape)

    mixop = OperationSpace(
        ops, return_mask=True
    )
    m = RandomMutator(mixop)
    m.reset()
    y, mask = mixop(x)
    print(y.shape, mask)
    
    # test InputSpace
    input1 = torch.rand(1,3)
    input2 = torch.rand(1,2)
    input3 = torch.rand(2,1)
    inputs = [input1, input2, input3]
    ic1 = InputSpace(n_candidates=3, n_chosen=1, return_mask=True)
    m = RandomMutator(ic1)
    m.reset()
    out, mask = ic1(inputs)
    print(out.shape, mask)

    inputs = {'key1':input1, 'key2':input2, 'key3':input3}
    ic2 = InputSpace(choose_from=['key1', 'key2', 'key3'], n_chosen=1, return_mask=True)
    m = RandomMutator(ic2)
    m.reset()
    out, mask = ic2(inputs)    
    print(out.shape, mask)
    
    vc1 = ValueSpace([8,16,24], index=1)
    vc2 = ValueSpace([1,2,3,4,5,6,7,8,9])
    vc = nn.ModuleList([vc1,vc2])
    m = RandomMutator(vc)
    m.reset()
    print(vc1, vc2)
