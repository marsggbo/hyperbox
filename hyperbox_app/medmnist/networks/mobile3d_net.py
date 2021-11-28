import torch
import torch.nn as nn
import math

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.utils.utils import load_json

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox_app.covid19.networks.kornia_transform import DataAugmentation
from hyperbox_app.covid19.networks.mobile3d_ops import *
from hyperbox_app.covid19.networks.mobile_utils import *


__all__ = [
    'Mobile3DNet',
    'DAMobile3DNet'
]


class Mobile3DNet(BaseNASNetwork):
    def __init__(
        self, c_in=3,
        width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1, num_classes=1000,
        dropout_rate=0, bn_param=(0.1, 1e-3), mask=None
    ):
        """
        Parameters
            ----------
            width_stages: str
                width (output channels) of each cell stage in the block
            n_cell_stages: str
                number of cells in each cell stage
            stride_strages: str
                stride of each cell stage in the block
            width_mult : int
                the scale factor of width
        """
        super(Mobile3DNet, self).__init__(mask)
        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        first_conv = ConvLayer(c_in, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                calibrate_op = CalibrationLayer(input_channel, width, stride)
                blocks.append(calibrate_op)
                op_candidates = [OPS['3x3_MBConv3'](width, width, 1),
                                 OPS['3x3_MBConv4'](width, width, 1),
                                 OPS['3x3_MBConv6'](width, width, 1),
                                 OPS['5x5_MBConv3'](width, width, 1),
                                 OPS['5x5_MBConv4'](width, width, 1),
                                 OPS['7x7_MBConv3'](width, width, 1),
                                 OPS['7x7_MBConv4'](width, width, 1),
                                 OPS['Identity'](width, width, 1),
                                 OPS['Zero'](width, width, 1),
                                 ]
                if stride == 1 and input_channel == width:
                    # if it is not the first one
                    op_candidates += [OPS['Zero'](input_channel, width, stride)]
                if self.mask: op_candidates = [op_candidates[self.mask[f"s{stage_cnt}_c{i}"].argmax()]]
                conv_op = OperationSpace(op_candidates, return_mask=True, key="s{}_c{}".format(stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(inverted_residual_block)
                input_channel = width
            stage_cnt += 1

        # feature mix layer
        last_channel = make_devisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        feature_mix_layer = ConvLayer(input_channel, last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act', )
        classifier = LinearLayer(last_channel, num_classes, dropout_rate=dropout_rate)

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.classifier = classifier

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        self.init_model()

    def forward(self, x, **kwargs):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    def init_model(self, model_init='he_fout', init_div_groups=False):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def reset_first_block(self, c_in):
        self.first_conv = ConvLayer(
            c_in, input_channel, kernel_size=3, stride=2,
            use_bn=True, act_func='relu6', ops_order='weight_bn_act')

    @property
    def arch(self):
        arch = ''
        for module in self.blocks:
            if isinstance(module, MobileInvertedResidualBlock):
                index = module.mobile_inverted_conv.mask.cpu().detach().numpy().argmax()
                arch +=f'{index}-'
        return arch


class DAMobile3DNet(BaseNASNetwork):
    def __init__(
        self,
        c_in=3, width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1, num_classes=1000,
        dropout_rate=0, bn_param=(0.1, 1e-3),

        rotate_degree=30, crop_size=[(32,128,128), (32,256,256)],
        affine_degree=0, affine_scale=(1.1, 1.5), affine_shears=20,
        mask=None
    ):
        super(DAMobile3DNet, self).__init__(mask)
        self.network = Mobile3DNet(
            c_in, width_stages, n_cell_stages, stride_stages, width_mult,
            num_classes, dropout_rate, bn_param)
        self.augmentation = DataAugmentation(
            rotate_degree, crop_size, affine_degree, affine_scale, affine_shears)

    def forward(self, x, to_aug=False):
        if to_aug:
            x = self.augmentation(x)
        x = self.network(x)
        return x

if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, RandomMutator, OnehotMutator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = DAMobile3DNet(1, num_classes=10).to(device)
    # net = Mobile3DNet(1, num_classes=10).to(device)
    dm = OnehotMutator(net)
    for i in range(10):
        dm.reset()
        # if i > 5:
        #     net = net.eval()
        x = torch.rand(10,1,64,300,300).to(device)
        y = net(x)
        print(y.argmax(-1))
