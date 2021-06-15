
import torch

from hyperbox.mutator import RandomMutator
from hyperbox.mutables import ops, spaces
from hyperbox.networks.mobilenet.mobile_net import MobileNet
from hyperbox.utils.calc_model_size import flops_size_counter


class HybridNet(torch.nn.Module):
    def __init__(self, mask=None):
        super().__init__()
        self.op_space1 = spaces.OperationSpace(
            candidates=[
                torch.nn.Conv2d(3,20,3,padding=1),
                torch.nn.Conv2d(3,20,5,padding=2)
            ],
            mask=mask
        )
        self.op_space2 = spaces.OperationSpace(
            candidates=[
                torch.nn.Conv2d(3,20,7,padding=3),
                torch.nn.Conv2d(3,20,3,padding=1)
            ],
            mask=mask
        )
        self.input_space = spaces.InputSpace(n_candidates=2, n_chosen=1, mask=mask)
        
        vs_channel = spaces.ValueSpace([24,16,18], mask=mask)
        vs_kernel = spaces.ValueSpace([3,5,7], mask=mask)
        vs_stride = spaces.ValueSpace([1,2], mask=mask)
        self.finegrained_conv = ops.Conv2d(20, vs_channel, vs_kernel, stride=vs_stride, auto_padding=True)
        self.finegrained_bn = ops.BatchNorm2d(vs_channel)
        self.finegrained_linear = ops.Linear(vs_channel, 1)

    def forward(self, x):
        bs = x.size(0)
        out1 = self.op_space1(x)
        out2 = self.op_space2(x)
        out = self.input_space([out1, out2])
        out = self.finegrained_conv(out)
        out = self.finegrained_bn(out)
        out = torch.nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(bs, -1)
        out = self.finegrained_linear(out)
        return out


if __name__ == '__main__':

    net = MobileNet()
    mutator = RandomMutator(net)
    mutator.reset()

    x = torch.rand(1,3,128,128)
    result = flops_size_counter(net, (x,))
    flops, params = [result[k] for k in result]
    print(f"{flops} MFLOPS, {params:.4f} MB")

    x = torch.rand(1,3,64,64)
    result = flops_size_counter(net, (x,))
    flops, params = [result[k] for k in result]
    print(f"{flops} MFLOPS, {params:.4f} MB")

    mask = {
        "OperationSpace1": [True, False],
        "OperationSpace2": [False, True],
        "InputSpace3": [True, False],
        "ValueSpace6": [True, False],
        "ValueSpace4": [False, True, False],
        "ValueSpace5": [False, False, True]
    }
    net = HybridNet(mask)
    x = torch.rand(1,3,64,64)
    result = flops_size_counter(net, (x,))
    flops, params = [result[k] for k in result]
    print(f"{flops} MFLOPS, {params:.4f} MB")

    net = HybridNet()
    mutator = RandomMutator(net)
    mutator.reset()
    x = torch.rand(1,3,32,32)
    result = flops_size_counter(net, (x,))
    flops, params = [result[k] for k in result]
    print(f"{flops} MFLOPS, {params:.4f} MB")
