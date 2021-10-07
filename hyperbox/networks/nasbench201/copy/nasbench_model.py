# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict

import torch
import torch.nn as nn

import hyperbox.mutables.spaces as spaces

from ..base_nas_network import BaseNASNetwork
from .nasbench_ops import (
    POOLING,
    DualSepConv,
    FactorizedReduce,
    Identity,
    PartAwareOp,
    ReLUConvBN,
    ResNetBasicblock,
    SepConv,
    Zero,
    drop_path,
)

__all__ = [
    'Node',
    'DartsCell',
    'DartsNetwork'
]

'''
none: Zero(C_in, C_out, stride)
skip_connect: Identity()
nor_conv_1x1: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride),
                         (0, 0), (1, 1), affine, track_running_stats)
nor_conv_3x3: ReLUConvBN(C_in,C_out,(3, 3),(stride, stride),(1, 1),(1, 1),affine,track_running_stats)
avg_pool_3x3: POOLING(C_in, C_out, stride, "avg", affine, track_running_stats)
'''


class NasBenchNode(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect, mask=None):
        '''
        node_id: 用于标记当前是第几个节点
        num_prev_nodes: 当前节点之前有多少个节点
        channels: 输出通道个数设置
        num_downsample_connect: int 降采样输入节点
        '''
        super().__init__()
        self.ops = nn.ModuleList()  # 所有的候选op
        choice_keys = []  # 保存所有key，用于后续input

        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            affine = False
            track_running_stats = False
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                spaces.OperationSpace(
                    [
                        Zero(channels, channels, stride),
                        Identity(),
                        ReLUConvBN(channels, channels, (1, 1), (stride, stride),
                                   (0, 0), (1, 1), affine, track_running_stats),
                        ReLUConvBN(channels, channels, (3, 3), (stride, stride),
                                   (1, 1), (1, 1), affine, track_running_stats),
                        POOLING(
                            channels, channels, stride, "avg", affine, track_running_stats)
                    ], key=choice_keys[-1], mask=mask
                )
            )
        self.drop_path = drop_path()
        self.input_switch = spaces.InputSpace(
            choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id), mask=mask)
    
    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = []
        for op, node in zip(self.ops, prev_nodes):
            _out = op(node) 
            out.append(_out)
        
        out = [self.drop_path(o) if o is not None else None for o in out]
        
        return self.input_switch(out)


class InferCell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(InferCell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev is None:
            self.preprocess0 = Identity()
        elif reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 2)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            # 降采样
            step_ops, concat = genotype.reduce, genotype.reduce_concat
        else:
            step_ops, concat = genotype.normal, genotype.normal_concat

        self._steps = len(step_ops)  # [2,3,4,5]
        self._concat = concat
        self._multiplier = len(concat)  # [2,3,4,5,6]
        self._ops = nn.ModuleList()
        self._indices = []
        '''
        PNASNet = Genotype(
            normal = [
                (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
                (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
                (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
                (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
                (('sep_conv_3x3', 0), ('skip_connect', 1)),
            ],
            normal_concat = [2, 3, 4, 5, 6],
            reduce = [
                (('sep_conv_5x5', 0), ('max_pool_3x3', 0)),
                (('sep_conv_7x7', 1), ('max_pool_3x3', 1)),
                (('sep_conv_5x5', 1), ('sep_conv_3x3', 1)),
                (('sep_conv_3x3', 4), ('max_pool_3x3', 1)),
                (('sep_conv_3x3', 0), ('skip_connect', 1)),
            ],
            reduce_concat = [2, 3, 4, 5, 6],
            connectN=None, connects=None,
            )
        '''

        for operations in step_ops:  # 一个cell内部有几个op
            # operations = (('sep_conv_5x5', 0), ('max_pool_3x3', 0))
            for name, index in operations:
                # name sep_conv_5x5, index = 0
                stride = 2 if reduction and index < 2 else 1

                if reduction_prev is None and index == 0:
                    op = OPS[name](C_prev_prev, C, stride, True)
                else:
                    op = OPS[name](C, C, stride, True)

                self._ops.append(op)
                self._indices.append(index)

    def extra_repr(self):
        return ('{name}(steps={_steps}, concat={_concat})'.format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, S0, S1, drop_prob):
        s0 = self.preprocess0(S0)
        s1 = self.preprocess1(S1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]

            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)

            state = h1 + h2
            states += [state]
        output = torch.cat([states[i] for i in self._concat], dim=1)
        return output


# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):
    def __init__(self, C, N, genotype, num_classes):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(genotype, C_prev, C_curr, 1)

            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += "\n {:02d}/{:02d} :: {:}".format(
                i, len(self.cells), cell.extra_repr()
            )
        return string

    def extra_repr(self):
        return "{name}(C={_C}, N={_layerN}, L={_Layer})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits


class Node(nn.Module):
    def __init__(self, node_id, num_prev_nodes, channels, num_downsample_connect, mask=None):
        """
        builtin Darts Node structure

        Parameters
        ---
        node_id: str
        num_prev_nodes: int
            the number of previous nodes in this cell
        channels: int
            output channels
        num_downsample_connect: int
            downsample the input node if this cell is reduction cell
        """
        super().__init__()
        self.ops = nn.ModuleList()
        choice_keys = []
        for i in range(num_prev_nodes):
            stride = 2 if i < num_downsample_connect else 1
            choice_keys.append("{}_p{}".format(node_id, i))
            self.ops.append(
                spaces.OperationSpace(
                    [
                        PoolBN('max', channels, 3, stride, 1, affine=False),
                        PoolBN('avg', channels, 3, stride, 1, affine=False),
                        nn.Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=False),
                        SepConv(channels, channels, 3, stride, 1, affine=False),
                        SepConv(channels, channels, 5, stride, 2, affine=False),
                        DilConv(channels, channels, 3, stride, 2, 2, affine=False),
                        DilConv(channels, channels, 5, stride, 4, 2, affine=False)
                    ], key=choice_keys[-1], mask=mask)
            )
        self.drop_path = DropPath()
        self.input_switch = spaces.InputSpace(
            choose_from=choice_keys, n_chosen=2, key="{}_switch".format(node_id), mask=mask)

    def forward(self, prev_nodes):
        assert len(self.ops) == len(prev_nodes)
        out = []
        for op, node in zip(self.ops, prev_nodes):
            _out = op(node)
            out.append(_out)
        # out = [op(node) for op, node in zip(self.ops, prev_nodes)]
        out = [self.drop_path(o) if o is not None else None for o in out]
        return self.input_switch(out)


class DartsCell(nn.Module):
    """
    Builtin Darts Cell structure. There are ``n_nodes`` nodes in one cell, in which the first two nodes' values are
    fixed to the results of previous previous cell and previous cell respectively. One node will connect all
    the nodes after with predefined operations in a mutable way. The last node accepts five inputs from nodes
    before and it concats all inputs in channels as the output of the current cell, and the number of output
    channels is ``n_nodes`` times ``channels``.

    Parameters
    ---
    n_nodes: int
        the number of nodes contained in this cell
    channels_pp: int
        the number of previous previous cell's output channels
    channels_p: int
        the number of previous cell's output channels
    channels: int
        the number of output channels for each node
    reduction_p: bool
        Is previous cell a reduction cell
    reduction: bool
        is current cell a reduction cell
    """

    def __init__(self, n_nodes, channels_pp, channels_p, channels, reduction_p, reduction, mask=None):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = StdConv(channels_pp, channels, 1, 1, 0, affine=False)
        self.preproc1 = StdConv(channels_p, channels, 1, 1, 0, affine=False)

        # generate dag
        self.mutable_ops = nn.ModuleList()
        for depth in range(2, self.n_nodes + 2):
            self.mutable_ops.append(Node("{}_n{}".format("reduce" if reduction else "normal", depth),
                                         depth, channels, 2 if reduction else 0, mask=mask))

    def forward(self, pprev, prev):
        """
        Parameters
        ---
        pprev: torch.Tensor
            the output of the previous previous layer
        prev: torch.Tensor
            the output of the previous layer
        """
        tensors = [self.preproc0(pprev), self.preproc1(prev)]
        for idx, node in enumerate(self.mutable_ops):
            cur_tensor = node(tensors)
            tensors.append(cur_tensor)

        output = torch.cat(tensors[2:], dim=1)
        return output


class DartsNetwork(BaseNASNetwork):
    """
    builtin Darts Search Mutable
    Compared to Darts example, DartsSearchSpace removes Auxiliary Head, which
    is considered as a trick rather than part of model.

    Attributes
    ---
        in_channels: int
            the number of input channels
        channels: int
            the number of initial channels expected
        n_classes: int
            classes for final classification
        n_layers: int
            the number of cells contained in this network
        factory_func: function
            return a callable instance for demand cell structure.
            user should pass in ``__init__`` of the cell class with required parameters (see nni.nas.DartsCell for detail)
        n_nodes: int
            the number of nodes contained in each cell
        stem_multiplier: int
            channels multiply coefficient when passing a cell
    """

    def __init__(self, in_channels, channels, n_classes, n_layers, factory_func=DartsCell, n_nodes=4,
                 stem_multiplier=3, mask=None):
        super(DartsNetwork, self).__init__(mask)
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] channels_pp and channels_p is output channel size, but c_cur is input channel size.
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels

        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                c_cur *= 2
                reduction = True

            cell = factory_func(n_nodes, channels_pp, channels_p, c_cur,
                                reduction_p, reduction, mask=self.mask)
            self.cells.append(cell)
            c_cur_out = c_cur * n_nodes
            channels_pp, channels_p = channels_p, c_cur_out

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(channels_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        for idx, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)

        return logits

    def drop_path_prob(self, p):
        for module in self.modules():
            if isinstance(module, DropPath):
                module.p = p

    @property
    def arch(self):
        arch = 'normal'
        # normal
        cell_ops = self.cells[0].mutable_ops
        for node_ops in cell_ops:
            sub_arch = ''
            for op in node_ops.ops:
                index = op.mask.cpu().detach().numpy().argmax()
                sub_arch += f'{index}'
            arch += f'-{sub_arch}'
        if len(self.cells._modules) < 3:
            return arch
        # reduce
        arch += '-reduce-'
        cell_ops = self.cells[2].mutable_ops
        for node_ops in cell_ops:
            sub_arch = ''
            for op in node_ops.ops:
                index = op.mask.cpu().detach().numpy().argmax()
                sub_arch += f'{index}'
            arch += f'-{sub_arch}'
        return arch
