"""Builds the Pytorch computational graph.

Tensors flowing into a single vertex are added together for all vertices
except the output, which is concatenated instead. Tensors flowing out of input
are always added.

If interior edge channels don't match, drop the extra channels (channels are
guaranteed non-decreasing). Tensors flowing out of the input as always
projected instead.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# tf1.x https://github.com/google-research/nasbench.git
# tf2.x https://github.com/gabikadlecova/nasbench.git
from nasbench import api

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.networks.nasbench101.base_ops import *
from hyperbox.mutables.spaces import ValueSpace


class NASBench101Network(BaseNASNetwork):
    ROOT_PATH = os.path.expanduser('~/.hyperbox/nasbench101')
    TFRECORD_FULL_PATH = os.path.join(ROOT_PATH, 'nasbench_full.tfrecord')
    TFRECORD_108_PATH = os.path.join(ROOT_PATH, 'nasbench_only_108.tfrecord')
    DB_PATH = os.path.join(ROOT_PATH, 'nasbench101.db')

    def __init__(
        self,
        spec,
        stem_out_channels: int=128,
        num_stacks: int=3,
        num_modules_per_stack: int=3,
        num_classes: int=10,
        query_type: str='tfrecord',
        query_file: str=None,
        mask=None
    ):
        super(NASBench101Network, self).__init__(mask)
        self.query_api = None
        self.layers = nn.ModuleList([])

        in_channels = 3
        out_channels = stem_out_channels

        self.query_type = query_type
        if query_file is not None:
            self.query_file = query_file
        else:
            if query_type == 'tfrecord':
                self.query_file = self.TFRECORD_FULL_PATH
            else:
                self.query_file = self.DB_PATH

        # initial stem convolution
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1)
        self.layers.append(stem_conv)

        in_channels = out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for module_num in range(num_modules_per_stack):
                cell = Cell(spec, in_channels, out_channels)
                self.layers.append(cell)
                in_channels = out_channels

        self.classifier = nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        out = torch.mean(x, (2, 3))
        out = self.classifier(out)

        return out

    @property
    def arch_db(self):
        arch = {}
        for i, op_name in enumerate(self.spec.ops):
            if op_name not in ['input', 'output']:
                arch[f'op{i}'] = op_name
        matrix = np.array(self.spec.matrix)
        for i in range(len(matrix)):
            onehot_list = matrix[:, i]
            # different from original matrix, here we should collect the input node indices for each node
            # e.g., 'input3': [0, 2] indicates the input nodes for node 3 are from node 0 and node 2
            if i != 0:
                arch[f"input{i}"] = onehot2indices(onehot_list)
        return arch

    @property
    def arch_spec(self):
        return self.spec

    @property
    def arch(self):
        if self.query_type == 'tfrecord':
            return self.arch_spec
        elif self.query_type == 'db':
            return self.arch_db

    def query_by_key(self, key='test_acc', spec=None, num_epochs=108, isomorphism=True, reduction='mean'):
        if not os.path.exists(self.ROOT_PATH):
            os.makedirs(self.ROOT_PATH)
        if not os.path.exists(self.TFRECORD_FULL_PATH):
            os.system('wget https://storage.googleapis.com/nasbench/nasbench_full.tfrecord')
            os.system(f'cp nasbench_full.tfrecord {self.ROOT_PATH}')
        if not os.path.exists(self.TFRECORD_108_PATH):
            os.system('wget https://storage.googleapis.com/nasbench/nasbench_only_108.tfrecord')
            os.system(f'cp nasbench_only_108.tfrecord {self.ROOT_PATH}')
        if self.query_api is None:
            if self.query_type == 'tfrecord':
                if self.query_file is not None:
                    self.query_api = api.NASBench(self.query_file).query
                else:
                    raise ValueError('query_file must be specified for query_type=tfrecord')
            elif self.query_type == 'db':
                from hyperbox.networks.nasbench101.db_gen.query import query_nb101_trial_stats
                self.query_api = query_nb101_trial_stats

        if self.query_type == 'tfrecord':
            self.arch_info = self.query_api(self.arch)
            return self.arch_info[key]
        else:
            # num_epochs: 4, 12, 36, 108
            # reduction: 'none', 'mean
            '''return example
            {
                'id': None,
                'config': {
                    'id': 4,
                    'arch': {'op1': 'conv3x3-bn-relu', 'op2': 'maxpool3x3', 'op3': 'conv3x3-bn-relu',
                        'op4': 'conv3x3-bn-relu', 'op5': 'conv1x1-bn-relu', 'input1': [0],
                        'input2': [1], 'input3': [2], 'input4': [0], 'input5': [0, 3, 4], 'input6': [2, 5]},
                    'num_vertices': 7,
                    'hash': '00005c142e6f48ac74fdcf73e3439874',
                    'num_epochs': 108
                    },
                'train_acc': 100.0,
                'valid_acc': 92.64155824979146,
                'test_acc': 92.06063151359558, 
                'parameters': 8.55553,
                'training_time': 106127.09716796875
            }
            '''
            self.arch_info = next(self.query_api(self.arch, num_epochs, isomorphism, reduction))
            return self.arch_info.get(key)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """
    def __init__(self, spec, in_channels, out_channels):
        super(Cell, self).__init__()

        self.spec = spec
        self.num_vertices = np.shape(self.spec.matrix)[0]

        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = ComputeVertexChannels(in_channels, out_channels, self.spec.matrix)
        #self.vertex_channels = [in_channels] + [out_channels] * (self.num_vertices - 1)

        # operation for each node
        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices-1):
            op = OP_MAP[spec.ops[t]](self.vertex_channels[t], self.vertex_channels[t])
            self.vertex_op.append(op)

        # operation for input on each vertex
        self.input_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices):
            if self.spec.matrix[0, t]:
                self.input_op.append(Projection(in_channels, self.vertex_channels[t]))
            else:
                self.input_op.append(None)

    def forward(self, x):
        tensors = [x]

        out_concat = []
        for t in range(1, self.num_vertices-1):
            fan_in = [Truncate(tensors[src], self.vertex_channels[t]) for src in range(1, t) if self.spec.matrix[src, t]]

            if self.spec.matrix[0, t]:
                fan_in.append(self.input_op[t](x))

            # perform operation on node
            #vertex_input = torch.stack(fan_in, dim=0).sum(dim=0)
            vertex_input = sum(fan_in)
            #vertex_input = sum(fan_in) / len(fan_in)
            vertex_output = self.vertex_op[t](vertex_input)

            tensors.append(vertex_output)
            if self.spec.matrix[t, self.num_vertices-1]:
                out_concat.append(tensors[t])

        if not out_concat:
            assert self.spec.matrix[0, self.num_vertices-1]
            outputs = self.input_op[self.num_vertices-1](tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.spec.matrix[0, self.num_vertices-1]:
                outputs += self.input_op[self.num_vertices-1](tensors[0])

            #if self.spec.matrix[0, self.num_vertices-1]:
            #    out_concat.append(self.input_op[self.num_vertices-1](tensors[0]))
            #outputs = sum(out_concat) / len(out_concat)

        return outputs


def Projection(in_channels, out_channels):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1)


def Truncate(inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def ComputeVertexChannels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.

    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.

    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
          vertex_channels[v] = interior_channels
          if correction:
              vertex_channels[v] += 1
              correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


def onehot2indices(onehot_list):
    indices = []
    for i, value in enumerate(onehot_list):
        if value == 1:
            indices.append(i)
    return indices


if __name__ == '__main__':
    import torch
    from nasbench import api
    from hyperbox.networks.nasbench101.model_spec import ModelSpec

    PATH = os.path.expanduser('~/.hyperbox/nasbench101')
    matrix = [
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]
    ]
    operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
    spec = api.ModelSpec(matrix, operations)
    query_api = api.NASBench(f'{PATH}/nasbench_only108.tfrecord')
    data = query_api.query(spec)
    for k, v in data.items():
        print('%s: %s' % (k, str(v)))

    x = torch.rand(2,3,64,64)

    net1 = NASBench101Network(spec)
    print(net1(x).shape)
    # net1.query_by_key()

    matrix2 = [[0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0]]
    operations2 = ['input',
        'conv3x3-bn-relu',
        'maxpool3x3',
        'conv3x3-bn-relu',
        'conv3x3-bn-relu',
        'conv1x1-bn-relu',
        'output']
    spec = ModelSpec(matrix2, operations2)

    # net2 = NASBench101Network(spec, query_type='db')    
    # print(net2.query_by_key('test_acc'))
    # print(net2(x).shape)
    
    spec = ModelSpec(matrix, operations)
    net3 = NASBench101Network(spec, query_file=f'{PATH}/nasbench_only108.tfrecord')
    print(net3.query_by_key('test_accuracy'))
    print(net3(x).shape)
