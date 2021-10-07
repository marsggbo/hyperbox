from typing import Union, Optional, List
import copy

import torch
import torch.nn as nn

from hyperbox.mutables import spaces, ops, layers
from hyperbox.utils.utils import load_json, hparams_wrapper

from hyperbox.networks.utils import val2list, make_divisible
from hyperbox.networks.pytorch_modules import Hsigmoid, Hswish, ResidualBlock
from hyperbox.networks.base_nas_network import BaseNASNetwork


@hparams_wrapper
class OFAMobileNetV3(BaseNASNetwork):
    CHANNEL_DIVISIBLE = 8

    def __init__(
        self,
        kernel_size_list: List[int] = [3, 5, 7],
        expand_ratio_list: List[float] = [3, 4, 6],
        depth_list: List[int] = [2, 3, 4],
        base_stage_width: List[int] = [16, 16, 24, 40, 80, 112, 160, 960, 1280],
        stride_stages: List[int] = [1, 2, 2, 2, 1, 2],
        act_stages: List[str] = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish'],
        se_stages: List[bool] = [False, False, True, False, True, True],
        width_mult: float = 1.0,
        num_classes: int = 1000,
        first_stride: int = 1, # 1: CIFAR10 2: Imagenet
        mask=None,
    ):
        super(OFAMobileNetV3, self).__init__()
        self.mask = load_json(mask)
        self.kernel_size_list = sorted(val2list(kernel_size_list, 1))
        self.expand_ratio_list = sorted(val2list(expand_ratio_list, 1))
        self.depth_list = sorted(val2list(depth_list, 1))
    
        final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, self.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(base_stage_width[-1] * self.width_mult, self.CHANNEL_DIVISIBLE)

        n_block_list = [1] + [max(self.depth_list)] * 5
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(base_width * self.width_mult, self.CHANNEL_DIVISIBLE)
            width_list.append(width)

        # first stem layer
        first_channels, first_block_dim = width_list[0], width_list[1]
        if first_stride==2:
            conv = nn.Conv2d(3, first_channels, kernel_size=3, stride=2, padding=1, bias=False) # imagenet
        else:
            conv = nn.Conv2d(3, first_channels, kernel_size=3, stride=1, padding=0, bias=False) # cifar10
        self.stem_layer = nn.Sequential(
            conv,
            nn.BatchNorm2d(first_channels),
            Hswish()
        )

        # first block
        first_block_conv = layers.MBConvLayer(
            first_channels, first_block_dim, kernel_size=3, stride=stride_stages[0], groups=1,
            expand_ratio=1, act_func=act_stages[0], use_se=se_stages[0])
        first_block = ResidualBlock(
            first_block_conv,
            nn.Identity() if first_channels == first_block_dim else None
        )

        blocks = [first_block]

        # inverted residual blocks
        self.block_group_info = []
        _block_index = 1
        feature_dim = first_block_dim

        for width, n_blocks, _stride, act_func, use_se in zip(
                width_list[2:], n_block_list[1:], stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_blocks)])
            _block_index += n_blocks

            output_channel = width
            for i in range(n_blocks):
                if i == 0:
                    stride = _stride
                else:
                    stride = 1
                kernel_size = spaces.ValueSpace(self.kernel_size_list, key=f'ks_block{_block_index-n_blocks+i}', mask=self.mask)
                # kernel_size = 3
                expand_ratio = spaces.ValueSpace(self.expand_ratio_list, key=f'er_block{_block_index-n_blocks+i}', mask=self.mask)
                mobile_inverted_conv = layers.MBConvLayer(
                    feature_dim, output_channel, kernel_size, stride, 1, expand_ratio, act_func, use_se
                )
                if stride == 1 and feature_dim == output_channel:
                    shortcut = nn.Identity()
                else:
                    shortcut = None
                blocks.append(ResidualBlock(mobile_inverted_conv, shortcut))
                feature_dim = output_channel
        self.blocks = nn.Sequential(*blocks)

        # final expand layer, feature mix layer & classifier
        self.final_expand_layer = nn.Sequential(
            nn.Conv2d(feature_dim, final_expand_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(final_expand_width),
            Hswish()
        )
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(final_expand_width, last_channel, kernel_size=1, bias=False),
            Hswish()
        )
        self.classifier = nn.Linear(last_channel, num_classes)

        # dynamic depth
        self.runtime_depth = []
        for idx, block_group in enumerate(self.block_group_info):
            self.runtime_depth.append(
                spaces.ValueSpace(list(range(1, len(block_group)+1)), key=f"depth{idx+1}", mask=self.mask)
            )
        self.runtime_depth = nn.Sequential(*self.runtime_depth)

    def forward(self, x, return_feature=False):
        features = []
        x = self.stem_layer(x)
        x = self.blocks[0](x)
        # inverted residual blocks
        for stage_id, block_group in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id].value
            active_idx = block_group[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)
        if return_feature: features.append(x)
        x = self.final_expand_layer(x)
        if return_feature: features.append(x)
        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = self.feature_mix_layer(x)
        if return_feature: features.append(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if return_feature: return (x, features)
        return x

    @property
    def arch(self):
        arch = ''
        for name, m in self.named_modules():
            if isinstance(m, spaces.Mutable):
                if 'ks' in m.key:
                    key = 'k'
                elif 'er' in m.key:
                    key = 'W'
                elif 'depth' in m.key:
                    key = 'D'
                else:
                    continue
                arch += f"{key}{m.value}-"
        return arch

    def build_archs_for_valid(
        self,
        depth_list: List = [2,3,4],
        expand_ratio_list: List = [4,6],
        kernel_size_list: List = [3,5,7]
    ):
        is_compliant = lambda _input, _origin: all([x in _origin for x in _input])
        assert is_compliant(depth_list, list(range(1,max(self.depth_list)+1))), f"all elements of your depth input {depth_list} must be in {list(range(1,max(self.depth_list)+1))}"
        assert is_compliant(expand_ratio_list, self.expand_ratio_list), f"all elements of your er input {expand_ratio_list} must be in {self.expand_ratio_list}"
        assert is_compliant(kernel_size_list, self.kernel_size_list), f"all elements of your ks input {kernel_size_list} must be in {self.kernel_size_list}"
        self.archs_to_valid = {}
        for d in depth_list:
            for e in expand_ratio_list:
                for k in kernel_size_list:
                    mask = self.gen_mask(d, e, k)
                    self.archs_to_valid[f"d{d}-e{e}-k{k}"] = mask
        return self.archs_to_valid

    def gen_mask(self, depth, expand_ratio, kernel_size):
        mask = {}
        for m in self.modules():
            if isinstance(m, spaces.Mutable):
                if 'mc' in m.key:
                    mask_item = m.mask
                    mask[m.key] = mask_item
                    continue
                if 'ks' in m.key:
                    mask_item = (torch.tensor(self.kernel_size_list) - kernel_size)==0
                elif 'er' in m.key:
                    mask_item = (torch.tensor(self.expand_ratio_list) - expand_ratio)==0
                elif 'depth' in m.key:
                    mask_item = (torch.arange(1, max(self.depth_list)+1) - depth)==0
                assert mask_item.shape==m.mask.shape,\
                    f"{mask_item.shape} failed to match the original shape {m.mask.shape} of {m.key}"
                mask[m.key] = mask_item
        return mask

    def build_search_space(self, mutator):
        import re
        from itertools import product
        key2num = lambda key: int(re.findall(r'\d{1,3}', key)[0])
        num2id = lambda num: (num-1)//4
        d2e = {2:3, 3:4, 4:6}
        
        depth_list = sorted(self.depth_list)
        ks_list = sorted(self.kernel_size_list)
        er_list = sorted(self.expand_ratio_list)
        assert len(depth_list)==len(er_list),\
            f'the length of depth_list should be equal to that of er_list'
        combinations = [depth_list]*len(self.block_group_info)
        combinations = list(product(*combinations))
        combinations = list(product(*[combinations, ks_list]))

        masks = {}
        for c in combinations:
            depths, k = c
            expand_ratios = [d2e[d] for d in depths]
            mask = {}
            for m in mutator.mutables:
                key = m.key
                num = key2num(key)
                layer_id = num2id(num)
                if 'ks' in key:
                    mask[key] = (torch.tensor(ks_list)-k)==0
                elif 'er' in key:
                    mask[key] = (torch.tensor(er_list)-expand_ratios[layer_id])==0
                elif 'depth' in key:
                    mask[key] = (torch.arange(1, max(depth_list)+1)-depths[layer_id])==0
            masks[str(c)] = mask
        return masks


if __name__ == '__main__':
    import json
    from hyperbox.mutator import RandomMutator
    x = torch.rand(2,3,64,64)
    net = OFAMobileNetV3()
    rm = RandomMutator(net)
    rm.reset()
    masks = net.build_search_space(rm)
    from hyperbox.utils.utils import TorchTensorEncoder
    with open('ofa_mbv3_searchspace.json','w') as f:
        json.dump(masks, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)
    for i in range(10):
        m.reset()
        r = net.arch_size((2,3,32,32), True, True)
        y = net(x)
    print(y.shape)
