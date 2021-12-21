import argparse
import json
import logging
import os
import pprint
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from hyperbox.mutables import spaces
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.networks.nasbench201.db_gen import PRIMITIVES, query, query_nb201_trial_stats


class Pooling(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    bn_affine: bool
        If set to ``True``, ``torch.nn.BatchNorm2d`` will have learnable affine parameters. Default: True
    bn_momentun: float
        the value used for the running_mean and running_var computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, ``torch.nn.BatchNorm2d`` tracks the running mean and variance. Default: True
    """

    def __init__(
        self,
        C_in,
        C_out,
        stride,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(Pooling, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 0, bn_affine, bn_momentum, bn_track_running_stats
            )
        self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.preprocess:
            x = self.preprocess(x)
        return self.op(x)


class Zero(nn.Module):
    """
    Parameters
    ---
    C_in: int
        the number of input channels
    C_out: int
        the number of output channels
    stride: int
        stride of the convolution
    """

    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        """
        Parameters
        ---
        x: torch.Tensor
            input tensor
        """
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros


class FactorizedReduce(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(FactorizedReduce, self).__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(
                    nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False)
                )
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError("Invalid stride : {:}".format(stride))
        self.bn = nn.BatchNorm2d(
            C_out,
            affine=bn_affine,
            momentum=bn_momentum,
            track_running_stats=bn_track_running_stats,
        )

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class NASBench201Cell(nn.Module):
    """
    Builtin cell structure of NAS Bench 201. One cell contains four nodes. The First node serves as an input node
    accepting the output of the previous cell. And other nodes connect to all previous nodes with an edge that
    represents an operation chosen from a set to transform the tensor from the source node to the target node.
    Every node accepts all its inputs and adds them as its output.
    Parameters
    ---
    cell_id: str
        the name of this cell
    C_in: int
        the number of input channels of the cell
    C_out: int
        the number of output channels of the cell
    stride: int
        stride of all convolution operations in the cell
    bn_affine: bool
        If set to ``True``, all ``torch.nn.BatchNorm2d`` in this cell will have learnable affine parameters. Default: True
    bn_momentum: float
        the value used for the running_mean and running_var computation. Default: 0.1
    bn_track_running_stats: bool
        When set to ``True``, all ``torch.nn.BatchNorm2d`` in this cell tracks the running mean and variance. Default: True
    """

    def __init__(
        self,
        cell_id,
        C_in,
        C_out,
        stride,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
        mask=None
    ):
        super(NASBench201Cell, self).__init__()

        self.NUM_NODES = 4
        self.layers = nn.ModuleList()

        OPS = lambda layer_idx: [
            Zero(C_in, C_out, stride),
            Pooling(C_in, C_out, stride if layer_idx == 0 else 1,
                bn_affine,bn_momentum, bn_track_running_stats),
            ReLUConvBN( C_in, C_out, 3, stride if layer_idx == 0 else 1,
                1, 1, bn_affine, bn_momentum, bn_track_running_stats),
            ReLUConvBN(C_in, C_out, 1, stride if layer_idx == 0 else 1,
                0, 1, bn_affine, bn_momentum, bn_track_running_stats),
            nn.Identity() if stride == 1 and C_in == C_out
            else FactorizedReduce(C_in, C_out, stride if layer_idx == 0 else 1,
                bn_affine, bn_momentum, bn_track_running_stats),
        ]

        for i in range(self.NUM_NODES):
            node_ops = nn.ModuleList()
            for j in range(0, i):
                node_ops.append(spaces.OperationSpace(OPS(j), key="%d_%d" % (j, i), mask=mask))

            self.layers.append(node_ops)

        self.in_dim = C_in
        self.out_dim = C_out
        self.cell_id = cell_id

    def forward(self, input):  # pylint: disable=W0622
        """
        Parameters
        ---
        input: torch.tensor
            the output of the previous layer
        """
        nodes = [input]
        for i in range(1, self.NUM_NODES):
            node_feature = sum(self.layers[i][k](nodes[k]) for k in range(i))
            nodes.append(node_feature)
        return nodes[-1]


class ReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(
                C_out,
                affine=bn_affine,
                momentum=bn_momentum,
                track_running_stats=bn_track_running_stats,
            ),
        )

    def forward(self, x):
        return self.op(x)


class ResNetBasicBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        stride,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
    ):
        super(ResNetBasicBlock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = ReLUConvBN(
            inplanes,
            planes,
            3,
            stride,
            1,
            1,
            bn_affine,
            bn_momentum,
            bn_track_running_stats,
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, bn_affine, bn_momentum, bn_track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes,
                planes,
                1,
                1,
                0,
                1,
                bn_affine,
                bn_momentum,
                bn_track_running_stats,
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)
        return inputs + basicblock


class NASBench201Network(BaseNASNetwork):
    def __init__(
        self,
        stem_out_channels: int=16,
        num_modules_per_stack: int=5,
        bn_affine=True,
        bn_momentum=0.1,
        bn_track_running_stats=True,
        num_classes=10,
        mask=None
    ):
        super(NASBench201Network, self).__init__(mask=mask)
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_classes = num_classes

        self.bn_momentum = bn_momentum
        self.bn_affine = bn_affine
        self.bn_track_running_stats = bn_track_running_stats

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C, momentum=self.bn_momentum),
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()

        for i, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicBlock(
                    C_prev,
                    C_curr,
                    2,
                    self.bn_affine,
                    self.bn_momentum,
                    self.bn_track_running_stats,
                )
            else:
                cell = NASBench201Cell(
                    i,
                    C_prev,
                    C_curr,
                    1,
                    self.bn_affine,
                    self.bn_momentum,
                    self.bn_track_running_stats,
                    mask=self.mask
                )

            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev, momentum=self.bn_momentum), nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_classes)
        self.init_weights()

    def forward(self, inputs):
        out = self.stem(inputs)
        bs = inputs.shape[0]
        for idx, cell in enumerate(self.cells):
            out = cell(out)

        out = self.lastact(out)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    @property
    def arch(self):
        arch_json = {}
        for name, module in self.named_modules():
            if isinstance(module, spaces.Mutable):
                key = module.key
                if self.mask is not None:
                    mask = self.mask[key]
                else:
                    mask = module.mask
                idx = torch.argmax(mask.float())
                op_name = PRIMITIVES[idx]
                arch_json[key] = op_name
        return arch_json

    def sync_mask_for_all_cells(self, mask):
        '''
        Should be called after mutator.reset().
        For example:
        net = NASBench201Network()
        rm = RandomMutator(net)
        mutator.reset()
        net.sync_mask_for_all_cells(rm._cache)
        '''
        for cell in self.cells:
            if not hasattr(cell, 'layers'):
                continue
            for op_list in cell.layers:
                if len(op_list) > 0:
                    for op in op_list:
                        op.mask = mask[op.key]

    def print_cell_mask_by_idx(self, idx):
        for op_list in self.cells.__getitem__(idx).layers:
            if len(op_list) > 0:
                for op in op_list:
                    print(op.key, op.mask)

    def query_by_key(self, key='test_acc', arch=None, num_epochs=200, dataset='cifar10', reduction='mean'):
        if arch is None:
            arch = self.arch
        self.arch_info = next(query_nb201_trial_stats(arch, num_epochs, dataset, reduction))
        return self.arch_info.get(key, '-1')


if __name__ == "__main__":
    from hyperbox.mutator import RandomMutator
    from hyperbox.networks.nasbench201.db_gen import query_nb201_trial_stats
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = NASBench201Network(16, 5).to(device)
    net.verbose = 0
    rm = RandomMutator(net)
    num = 3
    for i in range(5):
        rm.reset()
        x = torch.randn(num,3,64,64).to(device)
        preds = net(x)
        print(preds.argmax(-1))
    net = net.eval()
    for i in range(5):
        # rm.reset()
        x = torch.randn(num,3,64,64).to(device)
        preds = net(x)
        print(preds.argmax(-1))
    # for i in range(3):
    #     rm.reset()
    #     net.sync_mask_for_all_cells(rm._cache)
    #     net.print_cell_mask_by_idx(2)
    #     a = torch.rand(2, 3, 64, 64).to(device)
    #     b = net(a)
    #     arch_json = net.arch 
    #     acc = net.query_by_key()
    #     for t in query_nb201_trial_stats(arch_json, 200, 'cifar10'):
    #         pprint.pprint(t)
