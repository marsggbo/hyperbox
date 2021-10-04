import torch
import torch.nn as nn

__all__ = [
    'blocks_dict',
    'InvertedResidual',
    'conv_1x1_bn',
    'conv_bn'
]


blocks_dict = {
    'k3r3':lambda inp, oup, stride : InvertedResidual(inp, oup, 3, 1, stride, 3),
    'k3r6':lambda inp, oup, stride : InvertedResidual(inp, oup, 3, 1, stride, 6),
    'k5r3':lambda inp, oup, stride : InvertedResidual(inp, oup, 5, 2, stride, 3),
    'k5r6':lambda inp, oup, stride : InvertedResidual(inp, oup, 5, 2, stride, 6),
    'k7r3':lambda inp, oup, stride : InvertedResidual(inp, oup, 7, 3, stride, 3),
    'k7r6':lambda inp, oup, stride : InvertedResidual(inp, oup, 7, 3, stride, 6),
}


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ksize, padding, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, ksize, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, ksize, stride, padding, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

