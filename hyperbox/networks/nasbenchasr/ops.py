import functools

import torch
import torch.nn as nn


class PadConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, strides, groups=1, dropout_rate=0, context=4, name='PadConvRelu'):
        super().__init__()
        self.name = name

        if int(context / strides) >= (kernel_size*dilation-strides):
            rpad = kernel_size*dilation-strides
            lpad = 0
        else:
            rpad = int(context / strides)
            lpad = int((kernel_size - 1)*dilation - rpad)

        self.pad = nn.ZeroPad2d((lpad, rpad, 0, 0))
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=strides, dilation=dilation, groups=groups)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout(x)
        return x


class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Linear'):
        super().__init__()
        self.name = name

        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        shape = x.shape
        x = x.permute(0,2,1)
        x = self.linear(x)
        x = self.relu(x)
        x = torch.clamp_max_(x, 20)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        return x


class Identity(nn.Module):
    def __init__(self, name='Identity'):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, name='Zero'):
        super(Zero, self).__init__()
        self.name = name

    def forward(self, x):
        return torch.zeros_like(x)


_ops = {
    'linear': Linear,
    'conv5': functools.partial(PadConvRelu, kernel_size=5, dilation=1, strides=1, groups=100, name='conv5'),
    'conv5d2': functools.partial(PadConvRelu, kernel_size=5, dilation=2, strides=1, groups=100, name='conv52d'),
    'conv7': functools.partial(PadConvRelu, kernel_size=7, dilation=1, strides=1, groups=100, name='conv7'),
    'conv7d2': functools.partial(PadConvRelu, kernel_size=7, dilation=2, strides=1, groups=100, name='conv52d'),
    'zero': lambda *args, **kwargs: Zero(name='zero')
}

_branch_ops = {
    0: Zero, # branch not present
    1: Identity # branch present
}

