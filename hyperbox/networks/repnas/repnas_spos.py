import os
import pickle
import re

import torch
import torch.nn as nn
from hyperbox.mutables.spaces import OperationSpace

# from .rep_ops import *
from hyperbox.networks.repnas.rep_ops import *
from hyperbox.networks.repnas.utils import *
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.utils.utils import load_json 

class RepNAS(BaseNASNetwork):
    """based on shufflenetv2 one shot
    single path one shot based SuperNet
    """

    def __init__(
        self,
        input_size=32,
        first_conv_channels=16,
        last_conv_channels=256,
        n_classes=10,
        affine=False,
        mask=None,
    ):
        super(RepNAS, self).__init__()

        assert input_size % 32 == 0
        # with open(os.path.join(os.path.dirname(__file__), op_flops_path), "rb") as fp:
        #     self._op_flops_dict = pickle.load(fp)

        self.mask = load_json(mask)        
        self.stage_blocks = [4, 4, 8, 4]  # [4, 4, 8, 4]
        self.stage_channels = [32, 80, 160, 320]  # [32, 80, 160, 320]  # divided by 2
        self._parsed_flops = dict()
        self._input_size = input_size
        self._feature_map_size = input_size
        self._first_conv_channels = first_conv_channels
        self._last_conv_channels = last_conv_channels
        self._n_classes = n_classes
        self._affine = affine

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 1, 1, bias=False),
            # to train cifar10 change stride=2 to stride=1
            nn.BatchNorm2d(first_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )

        p_channels = first_conv_channels
        features = []
        for idx, (num_blocks, channels) in enumerate(zip(self.stage_blocks, self.stage_channels)):
            features.extend(self._make_blocks(idx, num_blocks, p_channels, channels))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(self._feature_map_size)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, idx, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels
            oup = channels

            choice_block = OperationSpace(
                [
                    DBBORIGIN(inp, oup, kernel_size=3, stride=stride),
                    DBBAVG(inp, oup, kernel_size=3, stride=stride),
                    DBB1x1(inp, oup, stride=stride),
                    DBB1x1kxk(inp, oup, kernel_size=3, stride=stride),
                ],
                return_mask=False,
                mask = self.mask,
                key="idx{}_block{}_stride{}".format(idx, i, stride),
            )
            result.append(choice_block)

            if stride == 2:
                self._feature_map_size //= 2
        return result

    def forward(self, x):
        bs = x.size(0)
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    # def get_candidate_flops(self, candidate):
    #     conv1_flops = self._op_flops_dict["conv1"][(3, self._first_conv_channels,
    #                                                 self._input_size, self._input_size, 2)]
    #     # Should use `last_conv_channels` here, but megvii insists that it's `n_classes`. Keeping it.
    #     # https://github.com/megvii-model/SinglePathOneShot/blob/36eed6cf083497ffa9cfe7b8da25bb0b6ba5a452/src/Supernet/flops.py#L313
    #     rest_flops = self._op_flops_dict["rest_operation"][(self.stage_channels[-1], self._n_classes,
    #                                                         self._feature_map_size, self._feature_map_size, 1)]
    #     total_flops = conv1_flops + rest_flops
    #     for k, m in candidate.items():
    #         parsed_flops_dict = self._parsed_flops[k]
    #         if isinstance(m, dict):  # to be compatible with classical nas format
    #             total_flops += parsed_flops_dict[m["_idx"]]
    #         else:
    #             total_flops += parsed_flops_dict[torch.max(m, 0)[1]]
    #     return total_flops

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    """
    DBBORIGIN
    DBBAVG
    DBB1x1
    DBB1x1kxk
    """
    from hyperbox.mutator import RandomMutator, DartsMutator

    m = RepNAS()
    rm = RandomMutator(m)
    rm.reset()

    # replace(m)

    input = torch.zeros(5, 3, 32, 32)
    # print(m)
    print(m(input).shape)
