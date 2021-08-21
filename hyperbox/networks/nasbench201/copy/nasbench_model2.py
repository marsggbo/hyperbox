# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict, namedtuple

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
            step_ops, concat = genotype.reduce, genotype.reduce_concat
        else:
            step_ops, concat = genotype.normal, genotype.normal_concat

        self._steps = len(step_ops)
        self._concat = concat
        self._multiplier = len(concat)
        self._ops = nn.ModuleList()
        self._indices = []

        for operations in step_ops:
            for name, index in operations:
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


def dict2config(xdict, logger):
    assert isinstance(xdict, dict), "invalid type : {:}".format(type(xdict))
    Arguments = namedtuple("Configure", " ".join(xdict.keys()))
    content = Arguments(**xdict)
    if hasattr(logger, "log"):
        logger.log("{:}".format(content))
    return content


# Cell-based NAS Models
def get_cell_based_tiny_net(config):
    if isinstance(config, dict):
        config = dict2config(config, None)  # to support the argument being a dict
    super_type = getattr(config, "super_type", "basic")

    group_names = ["DARTS-V1", "DARTS-V2", "GDAS", "SETN", "ENAS", "RANDOM", "generic"]
    
    if super_type == "basic" and config.name in group_names:
        from .cell_searchs import nas201_super_nets as nas_super_nets

        try:
            return nas_super_nets[config.name](
                config.C,
                config.N,
                config.max_nodes,
                config.num_classes,
                config.space,
                config.affine,
                config.track_running_stats,
            )
        except:
            return nas_super_nets[config.name](
                config.C, config.N, config.max_nodes, config.num_classes, config.space
            )
    elif super_type == "search-shape":
        from .shape_searchs import GenericNAS301Model

        genotype = CellStructure.str2structure(config.genotype)
        return GenericNAS301Model(
            config.candidate_Cs,
            config.max_num_Cs,
            genotype,
            config.num_classes,
            config.affine,
            config.track_running_stats,
        )
    elif super_type == "nasnet-super":
        from .cell_searchs import nasnet_super_nets as nas_super_nets

        return nas_super_nets[config.name](
            config.C,
            config.N,
            config.steps,
            config.multiplier,
            config.stem_multiplier,
            config.num_classes,
            config.space,
            config.affine,
            config.track_running_stats,
        )
    elif config.name == "infer.tiny":
        from .cell_infers import TinyNetwork

        if hasattr(config, "genotype"):
            genotype = config.genotype
        elif hasattr(config, "arch_str"):
            genotype = CellStructure.str2structure(config.arch_str)
        else:
            raise ValueError(
                "Can not find genotype from this config : {:}".format(config)
            )
        return TinyNetwork(config.C, config.N, genotype, config.num_classes)
    elif config.name == "infer.shape.tiny":
        from .shape_infers import DynamicShapeTinyNet

        if isinstance(config.channels, str):
            channels = tuple([int(x) for x in config.channels.split(":")])
        else:
            channels = config.channels
        genotype = CellStructure.str2structure(config.genotype)
        return DynamicShapeTinyNet(channels, genotype, config.num_classes)
    elif config.name == "infer.nasnet-cifar":
        from .cell_infers import NASNetonCIFAR

        raise NotImplementedError
    else:
        raise ValueError("invalid network name : {:}".format(config.name))
