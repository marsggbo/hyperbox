import torch
import torch.nn as nn
import math

from hyperbox.mutables.spaces import OperationSpace, ValueSpace
from hyperbox.utils.utils import load_json

from hyperbox.utils.utils import hparams_wrapper
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox_app.medmnist.networks.kornia_transform import DataAugmentation
from hyperbox_app.medmnist.networks.mobile3d_ops import *
from hyperbox_app.medmnist.networks.mobile_utils import *


__all__ = [
    'Mobile3DNet',
    'DAMobile3DNet'
]

class SharePart3D(BaseNASNetwork):
    def __init__(
        self,
        in_channels=3,
        width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1,
        bn_param=(0.1, 1e-3),
        suffix='share',
        mask=None
    ):
        super(SharePart3D, self).__init__(mask)
        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)
        # first conv
        first_conv = ConvLayer(in_channels, input_channel, kernel_size=3, stride=2, use_bn=True, act_func='relu6', ops_order='weight_bn_act')
        # first block
        first_block_conv = OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)
        first_block = first_block_conv

        input_channel = first_cell_width

        blocks = [first_block]

        stage_cnt = 0
        self.runtime_depth = []
        n_cell_real_stages = []
        for idx, n_cell in enumerate(n_cell_stages):
            vs = ValueSpace(candidates=list(range(1, n_cell+1)), mask=self.mask, key=f"{suffix}_cell_depth{idx}")
            n_cell_real_stages.append(vs.value)
            self.runtime_depth.append(vs)
        self.runtime_depth = nn.Sequential(*self.runtime_depth)

        self.cell_group_info = []
        _cell_index = 1
        for width, n_cell, s in zip(width_stages, n_cell_real_stages, stride_stages):
            self.cell_group_info.append([_cell_index + i for i in range(n_cell)])
            _cell_index += n_cell
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                calibrate_op = CalibrationLayer(input_channel, width, stride)
                # blocks.append(calibrate_op)
                op_candidates = [OPS['3x3_MBConv3'](width, width, 1),
                                 OPS['3x3_MBConv4'](width, width, 1),
                                #  OPS['3x3_MBConv6'](width, width, 1),
                                 OPS['5x5_MBConv3'](width, width, 1),
                                #  OPS['5x5_MBConv4'](width, width, 1),
                                 OPS['7x7_MBConv3'](width, width, 1),
                                #  OPS['7x7_MBConv4'](width, width, 1),
                                 OPS['Identity'](width, width, 1),
                                #  OPS['Zero'](width, width, 1),
                                 ]
                # if stride == 1 and input_channel == width:
                #     # if it is not the first one
                #     op_candidates += [OPS['Zero'](input_channel, width, stride)]
                conv_op = OperationSpace(op_candidates, mask=self.mask, return_mask=True, key="{}_stage{}_cell{}".format(suffix, stage_cnt, i))
                # shortcut
                if stride == 1 and input_channel == width:
                    # if not first cell
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(conv_op, shortcut, op_candidates)
                blocks.append(nn.Sequential(calibrate_op, inverted_residual_block))
                input_channel = width
            stage_cnt += 1

        self.last_channel = input_channel
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])
        self.init_model()

    def forward(self, x, save_features=False, *args, **kwargs):
        bs = x.shape[0]
        if save_features: self.features = []

        x = self.first_conv(x)
        x = self.blocks[0](x)

        if self.mask is not None:
            for block in self.blocks[1:]:
                x = block(x)
                if save_features: self.features.append(x.view(bs,-1))
        else:
            for stage_id, cell_group in enumerate(self.cell_group_info):
                depth = self.runtime_depth[stage_id].value
                active_idx = cell_group[:depth]
                for idx in active_idx:
                    x = self.blocks[idx](x)
                    if save_features: self.features.append(x.view(bs,-1))
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

    def reset_first_block(self, in_channels):
        self.first_conv = ConvLayer(
            in_channels, input_channel, kernel_size=3, stride=2,
            use_bn=True, act_func='relu6', ops_order='weight_bn_act')

    @property
    def arch(self):
        arch = ''
        for module in self.blocks:
            if isinstance(module, MobileInvertedResidualBlock):
                index = module.mobile_inverted_conv.mask.cpu().detach().numpy().argmax()
                arch +=f'{index}-'
        return arch


class BranchPart3D(BaseNASNetwork):
    def __init__(
        self,
        in_channels=3,
        width_stages=[24,40,80,96,192,320],
        n_cell_stages=[4,4,4,4,4,1],
        stride_stages=[2,2,2,1,2,1],
        width_mult=1,
        bn_param=(0.1, 1e-3),
        dropout_rate=0,
        num_classes=10,
        suffix='branch',
        mask=None
    ):
        super(BranchPart3D, self).__init__(mask)
        self.feat = SharePart3D(in_channels, width_stages, n_cell_stages,
            stride_stages, width_mult, bn_param, suffix, self.mask)

        # feature mix layer
        self.last_channel = make_devisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.feature_mix_layer = ConvLayer(self.feat.last_channel, self.last_channel, kernel_size=1, use_bn=True, act_func='relu6', ops_order='weight_bn_act')

        self.global_avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.classifier = LinearLayer(self.last_channel, num_classes, dropout_rate=dropout_rate)

    def forward(self, x, save_features=False, *args, **kwargs):
        bs = x.shape[0]
        if save_features: self.features = []
        x = self.feat(x)
        # if save_features: self.features.append(x.view(bs,-1))
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        # if save_features: self.features.append(x.view(bs,-1))
        x = self.classifier(x)
        if save_features: self.features.append(x.view(bs,-1))
        return x


@hparams_wrapper
class Mobile3DEnsemble(BaseNASNetwork):
    def __init__(
        self,
        in_channels=3,
        share_width_stages=[24,40,80],
        share_n_cell_stages=[4,4,4],
        share_stride_stages=[1,1,2],
        share_width_mult=1,
        share_bn_param=(0.1, 1e-3),
        branch_width_stages=[96,192,320],
        branch_n_cell_stages=[4,4,4],
        branch_stride_stages=[2,2,2],
        branch_width_mult=1,
        branch_bn_param=(0.1, 1e-3),
        dropout_rate: float = 0,
        num_branches: int = 3,
        num_classes=10,
        mask=None
    ):
        super(Mobile3DEnsemble, self).__init__(mask)
        self.shared_part = SharePart3D(
            in_channels, share_width_stages, share_n_cell_stages, share_stride_stages,
            share_width_mult, share_bn_param, 'share', mask=self.mask
        )
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(BranchPart3D(
                self.shared_part.last_channel, branch_width_stages,
                branch_n_cell_stages, branch_stride_stages, branch_width_mult,
                branch_bn_param, dropout_rate, num_classes, f'branch{i}', mask=self.mask
            ))

    def forward(self, x, save_features=True, *args, **kwargs):
        out = self.shared_part(x)
        outs = []
        for branch in self.branches:
            outs.append(branch(out, save_features))
        return torch.stack(outs)
        # return torch.stack(outs).mean(0)


if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, RandomMutator, OnehotMutator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # net = SharePart3D().to(device)
    # net = BranchPart3D().to(device)
    net = Mobile3DEnsemble().to(device)
    net.eval()
    # net = Mobile3DNet(1, num_classes=10).to(device)
    dm = OnehotMutator(net)
    for i in range(10):
        dm.reset()
        subnet = Mobile3DEnsemble(mask=dm.export()).to(device)
        # if i > 5:
        #     net = net.eval()
        x = torch.rand(10,3,28,28,28).to(device)
        y = subnet(x)
        print(y.argmax(-1))
