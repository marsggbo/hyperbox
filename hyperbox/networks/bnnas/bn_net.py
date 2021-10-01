
import numpy as np

import torch
import torch.nn as nn

from hyperbox.mutables.ops import Conv2d, Linear, BatchNorm2d
from hyperbox.mutables.spaces import ValueSpace, OperationSpace
from hyperbox.utils.utils import load_json, hparams_wrapper
from hyperbox.networks.base_nas_network import BaseNASNetwork

from hyperbox.networks.bnnas.bn_blocks import blocks_dict, InvertedResidual, conv_bn, conv_1x1_bn


@hparams_wrapper
class BNNet(BaseNASNetwork):
    def __init__(
        self,
        first_stride: int=1,
        first_channels: int=24,
        width_mult: int=1,
        channels_list: list=[32, 40, 80, 96, 192, 320],
        num_blocks: list=[2, 2, 4, 4, 4, 1],
        strides_list: list=[2, 2, 2, 1, 2, 1],
        num_classes: int=1000,
        search_depth: bool=True,
        is_only_train_bn: bool=True,
        mask: dict=None
    ):
        super(BNNet, self).__init__()
        self.num_layers = len(channels_list)
        channels = int(first_channels * width_mult)
        self.first_conv = nn.Sequential(
            conv_bn(3, 40, first_stride),
            InvertedResidual(40, channels, 3, 1, 1, 1)
        )

        self.block_group_info = []
        self.features = nn.ModuleDict()
        c_in = channels
        for layer_id in range(self.num_layers):
            c_out = int(channels_list[layer_id] * width_mult)
            stride = strides_list[layer_id]
            n_blocks = num_blocks[layer_id]
            self.block_group_info.append([i for i in range(n_blocks)])

            ops = nn.Sequential()
            for i in range(n_blocks):
                key=f"layer{layer_id}_{i}"
                if i != 0:
                    stride = 1
                op = OperationSpace(
                    [block(c_in, c_out, stride) for block in blocks_dict.values()],
                    mask=mask, key=key
                )
                c_in = c_out
                ops.add_module(key, op)
            self.features[f"layer{layer_id}"] = ops

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(c_out, num_classes)
        )

        # dynamic depth
        self.runtime_depth = []
        for idx, block_group in enumerate(self.block_group_info):
            self.runtime_depth.append(
                ValueSpace(
                    list(range(1, len(block_group)+1)), key=f"depth{idx+1}", mask=mask)
            )
        self.runtime_depth = nn.Sequential(*self.runtime_depth)

        if is_only_train_bn:
            print(f"Only train BN.")
            self.freeze_except_last_bn()

    def forward(self, x):
        bs = x.shape[0]
        x = self.first_conv(x)

        if self.search_depth:
            for stage_id, key in enumerate(self.features.keys()):
                block_group = self.block_group_info[stage_id]
                depth = self.runtime_depth[stage_id].value
                active_idx = block_group[:depth]
                for idx in active_idx:
                    x = self.features[key][idx](x)
        else:
            for key in self.features:
                x = self.features[key](x)
        
        x = self.avgpool(x)
        x = x.view(bs, -1)
        x = self.classifier(x)
        return x

    def bn_metrics(self):
        values = 0
        for name, module in self.named_modules():
            if isinstance(module, OperationSpace):
                index = np.argmax(module.mask.numpy())
                op = module.candidates[index]
                value = op.conv[-1].weight.detach().mean()
                values += value
        return values

    def freeze_except_last_bn(self):
        for name, module in self.named_modules():
            module.requires_grad_ = False

        for name, module in self.named_modules():
            if isinstance(module, OperationSpace):
                for op in module.candidates:
                    op.conv[-1].requires_grad_ = True


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator
    x = torch.rand(2,3,64,64)
    net = BNNet()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    print(f"Supernet size: {net.arch_size((2,3,64,64), 1, 1)}")
    rm = RandomMutator(net)
    for i in range(10):
        rm.reset()
        print(f"Subnet size: {net.arch_size((2,3,64,64), 1, 1)}")
        print(net.bn_metrics())
    net.freeze_except_last_bn()
    y = net(x)
