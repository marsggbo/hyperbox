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


class RepBlock(nn.Module):
    def __init__(self, idx, i_block, inp, outp, stride, mask):
        super(RepBlock, self).__init__()
        self.stride = stride
        self.use_skip_connect = stride==1 and inp==outp

        self.block = OperationSpace(
            [
                DBBORIGIN(inp, outp, kernel_size=3, stride=stride),
                DBBAVG(inp, outp, kernel_size=3, stride=stride),
                DBB1x1(inp, outp, stride=stride),
                DBB1x1kxk(inp, outp, kernel_size=3, stride=stride),
            ],
            return_mask=False,
            mask=mask,
            key="idx{}_block{}_stride{}".format(idx, i_block, stride),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_skip_connect:
            out = out + x
        return out


class RepNAS(BaseNASNetwork):
    """based on shufflenetv2 one shot
    single path one shot based SuperNet
    """

    def __init__(
        self,
        first_conv_channels=24,
        last_conv_channels=512,
        n_classes=10,
        affine=True,
        mask=None,
    ):
        super(RepNAS, self).__init__(mask)

        self.mask = load_json(mask)
        self.stage_blocks = [2, 4, 4, 4, 4, 1]  # [4, 4, 8, 4]
        self.stage_channels = [32, 40, 80, 96, 192, 320]  # [32, 80, 160, 320]  # divided by 2
        self.stage_strides = [1, 2, 2, 2, 2, 2]
        self._parsed_flops = dict()
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
        for idx, (num_blocks, channels, stride) in enumerate(
            zip(self.stage_blocks, self.stage_channels, self.stage_strides)):
            features.extend(self._make_blocks(idx, num_blocks, p_channels, channels, stride))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=affine),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_channels, n_classes, bias=False),
        )

        self._initialize_weights()

    def _make_blocks(self, idx, blocks, in_channels, channels, stride):
        result = []
        for i_block in range(blocks):
            stride = stride if i_block == 0 else 1
            inp = in_channels if i_block == 0 else channels
            outp = channels

            result.append(RepBlock(idx, i_block, inp, outp, stride, self.mask))

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
    import numpy as np
    from hyperbox.mutator import RandomMutator, DartsMutator

    net = RepNAS()
    net.eval()
    rm = DartsMutator(net)
    rm.reset()
    net = RepNAS(mask=rm._cache)

    x = torch.zeros(2, 3, 32, 32)
    y1 = net(x)
    replace(net)
    net.eval()
    y2 = net(x)
    print(y1,y2)
    print(np.allclose(y1.detach().numpy(), y2.detach().numpy(), atol=1e-5))