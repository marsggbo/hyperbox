# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import OrderedDict, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import hyperbox.mutables.spaces as spaces
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.networks.darts.darts_ops import (
    DilConv,
    DropPath,
    FactorizedReduce,
    PoolBN,
    SepConv,
    StdConv,
)
from hyperbox.utils.visualize_darts_cell import PRIMITIVES, convert_genotypes, plot

# from hyperbox_app.nasbench301.utils import generate_results

__all__ = [
    'Node',
    'DartsCell',
    'NASBench301Network'
]

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


'''
git clone https://github.com/automl/nasbench301.git
cd nasbench301
cat requirements.txt | xargs -n 1 -L 1 pip install
pip install .
'''


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


class NASBench301Network(BaseNASNetwork):
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

    def __init__(self, in_channels=3, channels=16, n_classes=10, n_layers=8, factory_func=DartsCell, n_nodes=4,
                 stem_multiplier=3, mask=None):
        super(NASBench301Network, self).__init__(mask)
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
        # self.features_list = []
        s0 = s1 = self.stem(x)

        for idx, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            # self.features_list.append(s1)

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


if __name__ == "__main__":

    # generate geno from json files
    from hyperbox.mutator import DartsMutator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    net = NASBench301Network().to(device)
    dm = DartsMutator(net)
    for i in range(10):
        if i > 5:
            json_path = "/home/comp/18481086/code/hyperbox/hyperbox_app/nasbench301/mask_json/mask_epoch_110.json"
            net = NASBench301Network(mask=json_path).to(device)
        else:
            dm.reset()
        x = torch.rand(2,3,64,64).to(device)
        y = net(x)
        print(y.argmax(-1))
    # geno = convert_genotypes(json_path)
    # precision, runtime = generate_results(genotype_config=geno)
    # print(f"Genotype architecture performance: {precision}%, runtime {runtime}s")
    # plot(geno.normal, "normal")
    # plot(geno.reduce, "reduction")
