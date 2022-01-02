# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'SharpSepConv',
    'MixSeparableConv',
    'InvertedResidual',
    'InvertedResidualSE',
    'StdConv',
    'FactorizedDownsample',
    'ReductionLayer',
    'FactorizedUpsample',
    'DilConv',
    'SeparableConv',
    'Calibration',
    'ZeroLayer',
    'Hswish',
    'Hsigmoid',
    'SEModule',
    'ShuffleLayer',
    'ResidualBlock',
    'topk_gumbel_softmax',
    'TopKGumbelSoftmax'
]


class ShuffleLayer(nn.Module):

    def __init__(self, groups):
        super(ShuffleLayer, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, height, width)
        return x

    def __repr__(self):
        return 'ShuffleLayer(groups=%d)' % self.groups


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hsigmoid()'


class SEModule(nn.Module):
    REDUCTION = 4

    def __init__(self, channel, reduction=None):
        super(SEModule, self).__init__()

        self.channel = channel
        self.reduction = SEModule.REDUCTION if reduction is None else reduction

        num_mid = make_divisible(self.channel // self.reduction, divisor=MyNetwork.CHANNEL_DIVISIBLE)

        self.fc = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(self.channel, num_mid, 1, 1, 0, bias=True)),
            ('relu', nn.ReLU(inplace=True)),
            ('expand', nn.Conv2d(num_mid, self.channel, 1, 1, 0, bias=True)),
            ('h_sigmoid', Hsigmoid(inplace=True)),
        ]))

    def forward(self, x):
        y = x.mean(3, keepdim=True).mean(2, keepdim=True)
        y = self.fc(y)
        return x * y

    def __repr__(self):
        return 'SE(channel=%d, reduction=%d)' % (self.channel, self.reduction)


class ZeroLayer(nn.Module):
    
    def __init__(self, stride=None):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        '''n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding'''
        if self.stride == 1:
            return x.mul(0.)
        if len(x.shape)==4:
            return x[:,:,::self.stride,::self.stride].mul(0.)
        elif len(x.shape)==5:
            return x[:,:,::self.stride,::self.stride,::self.stride].mul(0.)

    @staticmethod
    def is_zero_layer():
        return True



# sharpSepConv
class SharpSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, 
            padding=1, dilation=1, affine=True, C_mid_mult=1, C_mid=None):
        super(SharpSepConv, self).__init__()
        cmid = int(C_out * C_mid_mult)
        cmid = C_mid if C_mid else cmid
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation, groups=C_in, bias=0),
            nn.Conv2d(C_in, cmid, kernel_size=1, padding=0, bias=0),
            nn.BatchNorm2d(cmid, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(cmid, cmid, kernel_size, stride=1, padding=(kernel_size-1)//2, dilation=1, groups=cmid, bias=0),
            nn.Conv2d(cmid, C_out, kernel_size=1, padding=0, bias=0),
            nn.BatchNorm2d(C_out, affine=affine))
    def forward(self, x):
        return self.op(x)

# Mix Separable Conv
def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class MDConv(nn.Module):
    def __init__(self, C_in, n_chunks, stride=1, bias=False):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(C_in, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            padding = (kernel_size - 1) // 2
            c_in_group = self.split_out_channels[idx] # the input channels of each group
            depthwise = nn.Conv2d(c_in_group, c_in_group, kernel_size=kernel_size, padding=padding,
                                  stride=stride, groups=c_in_group, bias=bias)
            self.layers.append(depthwise)

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return out


class MixSeparableConv(nn.Module):
    def __init__(self, C_in, C_out, n_chunks, stride=1, bias=False):
        super(MixSeparableConv, self).__init__()
        self.depthwise = MDConv(C_in, n_chunks, stride=1, bias=bias)
        self.pointwise = nn.Conv2d(C_in, C_out, 1, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = nn.ReLU()(out)
        return out


# InvertedResidual
class InvertedResidual(nn.Module):
    def __init__(self, C_in, C_out, expand_ratio=3, stride=1, kernel=3):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and C_in == C_out

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(C_in, C_in * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_in * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(C_in * expand_ratio, C_in * expand_ratio, kernel, stride, kernel // 2, groups=C_in * expand_ratio, bias=False),
            nn.BatchNorm2d(C_in * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(C_in * expand_ratio, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# InvertedResidualSE
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidualSE(nn.Module):
    '''The InvertedResidual with SELayer
    '''
    def __init__(self, C_in, C_out, se_ratio=4, expand_ratio=3, stride=1, kernel=3):
        super(InvertedResidualSE, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and C_in == C_out

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(C_in, C_in * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_in * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(C_in * expand_ratio, C_in * expand_ratio, kernel, stride, kernel // 2, groups=C_in * expand_ratio, bias=False),
            nn.BatchNorm2d(C_in * expand_ratio),
            nn.ReLU6(inplace=True),
            # se-layer
            SELayer(C_in * expand_ratio, reduction=se_ratio),
            # pw-linear
            nn.Conv2d(C_in * expand_ratio, C_out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# stdconv
class StdConv(nn.Module):
    def __init__(self, C_in, C_out):
        super(StdConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class FactorizedDownsample(nn.Module):
    '''used for downsample layer
    '''
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = nn.ReLU(inplace=True)(out)
        return out


class ReductionLayer(nn.Module):
    def __init__(self, in_channels_pp, in_channels_p, out_channels):
        super().__init__()
        self.reduce0 = FactorizedDownsample(in_channels_pp, out_channels, affine=False)
        self.reduce1 = FactorizedDownsample(in_channels_p, out_channels, affine=False)

    def forward(self, pprev, prev):
        return self.reduce0(pprev), self.reduce1(prev)


class FactorizedUpsample(nn.Module):
    '''used for Upsample layer
    h_out = (h_in - 1)*stride - 2*padding + dilation*(k-1) + out_padding + 1
    '''
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(C_in, C_out//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.conv2 = nn.ConvTranspose2d(C_in, C_out//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, dilation=2)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        out = nn.LeakyReLU(inplace=True)(out)
        return out


class DilConv(nn.Module):
    """
    (Dilated) depthwise separable conv.
    ReLU - (Dilated) depthwise separable - Pointwise - BN.
    If dilation == 2, 3x3 conv => 5x5 receptive field, 5x5 conv => 7x7 receptive field.
    o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    To keep the same size as the input:
        for 3x3 kernel, padding=dilation=2
        for 5x5 kernel, padding=2*dilation
    """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=2, dilation=2, affine=True):
        super().__init__()
        assert kernel_size in [3, 5], "kernel_size must be 3 or 5"
        if kernel_size == 3:
            padding = dilation
        elif kernel_size == 5:
            padding = 2*dilation
        self.dil_conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = self.dil_conv(x)
        out = self.bn(out)
        out = nn.ReLU()(out)
        return out


class SeparableConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding, stride=1):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, stride=stride,
                                   groups=C_in, bias=False)
        self.pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = nn.ReLU()(out)
        return out


class Calibration(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.process = None
        if in_channels != out_channels:
            self.process = StdConv(in_channels, out_channels)

    def forward(self, x):
        if self.process is None:
            return x
        return self.process(x)


class ResidualBlock(nn.Module):

    def __init__(self, conv, shortcut):
        super(ResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.conv is None or isinstance(self.conv, ZeroLayer):
            res = x
        elif self.shortcut is None or isinstance(self.shortcut, ZeroLayer):
            res = self.conv(x)
        else:
            res = self.conv(x) + self.shortcut(x)
        return res


class TopKGumbelSoftmax(nn.Module):
    def __init__(
        self,
        tau: float=1,
        hard: bool=False,
        topk: int=3,
        eps: float=1e-10,
        dim: int=-1
    ):
        super(TopKGumbelSoftmax, self).__init__()
        self.tau = tau
        self.hard = hard
        self.topk = topk
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        return topk_gumbel_softmax(x, self.tau, self.hard, self.topk, self.eps, self.dim)

def topk_gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False,
        topk: int = None, eps: float = 1e-10, dim: int = -1):
    # gumbels = (
    #     -torch.empty_like(
    #         logits, memory_format=torch.legacy_contiguous_format
    #     ).exponential_().log()
    # )
    gumbels = torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).uniform_()
    gumbels = -torch.log(eps - torch.log(gumbels+eps))
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    if topk is None:
        topk = (gumbels>gumbels.mean(dim, keepdim=True)).int().sum(dim)
    if hard:
        # Straight through.
        if len(torch.tensor(topk).view(-1)) == 1:
            indices = y_soft.topk(topk)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, indices, 1.0)
        else:
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
            for i, k in enumerate(topk):
                indices = y_soft[i].topk(k)[1]
                y_hard[i].scatter_(dim, indices, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret