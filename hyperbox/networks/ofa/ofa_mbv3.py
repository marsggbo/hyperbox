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
        base_stage_width: List[int] = [16, 16, 24, 40, 80, 112, 160, 960, 1280], # indices in [1,6] are searchable
        stride_stages: List[int] = [1, 2, 2, 2, 1, 2],
        act_stages: List[str] = ['relu', 'relu', 'relu', 'h_swish', 'h_swish', 'h_swish'],
        se_stages: List[bool] = [False, False, True, False, True, True],
        width_mult: float = 1.0,
        num_classes: int = 1000,
        first_stride: int = 1, # 1: CIFAR10 2: Imagenet
        to_search_depth: bool = False,
        mask=None,
    ):
        super(OFAMobileNetV3, self).__init__(mask)
        self.mask = load_json(mask)
        self.kernel_size_list = sorted(val2list(kernel_size_list, 1))
        self.expand_ratio_list = sorted(val2list(expand_ratio_list, 1))
        self.depth_list = sorted(val2list(depth_list, 1))
    
        final_expand_width = make_divisible(base_stage_width[-2] * self.width_mult, self.CHANNEL_DIVISIBLE)
        last_channel = make_divisible(base_stage_width[-1] * self.width_mult, self.CHANNEL_DIVISIBLE)

        num_searchable_stages = len(stride_stages) - 1
        n_block_list = [1] + [max(self.depth_list)] * num_searchable_stages
        width_list = []
        for base_width in base_stage_width[:-2]:
            width = make_divisible(base_width * self.width_mult, self.CHANNEL_DIVISIBLE)
            width_list.append(width)

        # first stem layer
        first_channels, first_block_dim = width_list[0], width_list[1]
        if first_stride==2:
            conv = nn.Conv2d(3, first_channels, kernel_size=3, stride=2, padding=1, bias=False) # imagenet
        else:
            conv = nn.Conv2d(3, first_channels, kernel_size=3, stride=1, padding=1, bias=False) # cifar10
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(final_expand_width, last_channel, kernel_size=1, bias=False),
            Hswish(),
            nn.Flatten(1)
        )
        self.classifier = nn.Linear(last_channel, num_classes)

        # dynamic depth
        if self.to_search_depth:
            self.runtime_depth = []
            for idx, block_group in enumerate(self.block_group_info):
                self.runtime_depth.append(
                    spaces.ValueSpace(list(range(1, len(block_group)+1)), key=f"depth{idx+1}", mask=self.mask)
                )
            self.runtime_depth = nn.Sequential(*self.runtime_depth)

    def forward(self, x):
        x = self.stem_layer(x)
        if self.to_search_depth:
            x = self.blocks[0](x)
            # inverted residual blocks
            for stage_id, block_group in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id].value
                active_idx = block_group[:depth]
                for idx in active_idx:
                    x = self.blocks[idx](x)
        else:
            x = self.blocks(x)
        x = self.final_expand_layer(x)
        x = self.avg_pool(x)
        x = self.feature_mix_layer(x)
        x = self.classifier(x)
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

    def gen_random_arch(self, mutator=None):
        if mutator is not None:
            mask = mutator._cache
        else:
            mask = {}
            for m in self.modules():
                if isinstance(m, spaces.Mutable):
                    key = m.key
                    idx = torch.randint(0, len(m), (1,))[0]
                    n_classes = torch.tensor(len(m))
                    val = torch.nn.functional.one_hot(idx, n_classes).bool()
                    mask[key] = val
        for key in mask:
            if '_subMB' in key:
                mask[key].data = mask[key.split('_subMB')[0]].data
        return mask
        

if __name__ == '__main__':
    import json
    import types
    from hyperbox.mutator import RandomMutator
    from torch.profiler import profile, record_function, ProfilerActivity
    device = 'cuda'
    x = torch.rand(8,3,224,224).to(device)
    net = OFAMobileNetV3(to_search_depth=False).to(device)
    rm = RandomMutator(net)
    rm.reset()
    # masks = net.build_search_space(rm)
    # from hyperbox.utils.utils import TorchTensorEncoder
    # with open('ofa_mbv3_searchspace.json','w') as f:
    #     json.dump(masks, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)

    def forward_profile(self, x):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    # schedule=torch.profiler.schedule(wait=1, warmup=2, active=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result/stem_layer', worker_name='worker0'),
                    record_shapes=True, profile_memory=True, with_stack=True) as prof:
            with record_function("stem_layer"):
                x = self.stem_layer(x)
        print('stem_layer\n', prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        if self.to_search_depth:
            x = self.blocks[0](x)
            # inverted residual blocks
            for stage_id, block_group in enumerate(self.block_group_info):
                depth = self.runtime_depth[stage_id].value
                active_idx = block_group[:depth]
                for idx in active_idx:
                    x = self.blocks[idx](x)
        else:
            for idx, block in enumerate(self.blocks):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],        
                    # schedule=torch.profiler.schedule(wait=1, warmup=2, active=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./result/block{idx}', worker_name='worker0'),
                    record_shapes=True, profile_memory=True, with_stack=True) as prof:
                    with record_function(f"block{idx}"):
                        x = block(x)
                print(f'block{idx}', prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    # schedule=torch.profiler.schedule(wait=1, warmup=2, active=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result/final_feature', worker_name='worker0'),
                    record_shapes=True, profile_memory=True, with_stack=True) as prof:
            with record_function("final_feature"):
                x = self.final_expand_layer(x)
                x = self.avg_pool(x)
                x = self.feature_mix_layer(x)
        print('final_feature\n',prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    # schedule=torch.profiler.schedule(wait=1, warmup=2, active=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result/classifier', worker_name='worker0'),
                    record_shapes=True, profile_memory=True, with_stack=True) as prof:
            with record_function("classifier"):
                x = self.classifier(x)
        print('classifier\n', prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        return x
    net.forward_profile = types.MethodType(forward_profile, net)
    for i in range(7):
        rm.reset()
        # r = net.arch_size((2,3,32,32), True, True)
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #             # schedule=torch.profiler.schedule(wait=1, warmup=2, active=2),
        #             # on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
        #             record_shapes=True, profile_memory=True, with_stack=True) as prof:
        #         net(x)
        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_self_cuda_time_total", row_limit=20))
        if i <=5:
            net(x)
        else:
            net.forward_profile(x)
        print(f"Iteration {i} finished")
        print("*"*50)