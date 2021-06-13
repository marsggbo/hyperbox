
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .base_module import FinegrainedModule
from .utils import build_activation, sub_filter_start_end


__all__ = [
    'BaseConvNd',
    'Conv1d',
    'Conv2d',
    'Conv3d'
]


class BaseConvNd(FinegrainedModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False,
        *args,
        **kwargs
        ):
        '''Base Conv Module
        Args:
            auto_padding: if set to true, will set a proper padding size to make output size same as the input size.
                For example, if kernel size is 3, the padding size is 1;
                if kernel_size is (3,7), the padding size is (1, 3)
        '''
        super(BaseConvNd, self).__init__()
        conv_kwargs = {key:getattr(self, key, None) for key in ['in_channels', 'out_channels', 'kernel_size',
            'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']}
        self.init_ops(conv_kwargs)
        self.is_search = self.isSearchConv()
        self.conv_dim = self.__class__.__name__[-2:]

    def init_ops(self, conv_kwargs):
        '''Generate Conv operation'''
        raise NotImplementedError

    def isSearchConv(self):
        '''Search flag
        Supported arguments
            - search_in_channel
            - search_out_channel
            - search_kernel_size
            - search_stride
            - search_dilation
            - search_groups
        '''
        self.search_in_channel = False
        self.search_out_channel = False
        self.search_kernel_size = False
        self.search_stride = False
        self.search_dilation = False
        self.search_groups = False
        # self.search_bias = False
        if len(self.value_spaces)==0:
            return False

        if 'in_channels' in self.value_spaces:
            self.search_in_channel = True
        if 'out_channels' in self.value_spaces:
            self.search_out_channel = True
        if 'kernel_size' in self.value_spaces:
            kernel_candidates = self.value_spaces['kernel_size'].candidates
            assert self.kernel_size==max(kernel_candidates)
            max_k = self.kernel_size
            # Todo: 与`transform_kernel_size`搭配使用，目前未使用
            for i, k in enumerate(sorted(kernel_candidates)[:-1]):
                self.register_parameter(f'{max_k}to{k}_kernelMatrix', Parameter(torch.rand(max_k**2, k**2)))
            self.search_kernel_size = True
        if 'stride' in self.value_spaces:
            self.search_stride = True
        if 'dilation' in self.value_spaces:
            self.search_dilation = True
        if 'groups' in self.value_spaces:
            self.search_groups = True
        # if 'bias' in self.value_spaces:
        #     self.search_bias = True

        return True

    ###########################################
    # forward implementation
    # - forward_conv
    #   - transform_kernel_size
    ###########################################

    def forward(self, x):
        out = None
        if not self.is_search:
            out = self.conv(x)
        else:
            out = self.forward_conv(self.conv, x)
        return out

    def forward_conv(self, module, x):
        filters = module.weight.contiguous()
        bias = module.bias
        in_channels = None
        out_channels = None
        stride = self.value_spaces['stride'].value if self.search_stride else self.stride
        groups = self.value_spaces['groups'].value if self.search_groups else self.groups
        dilation = self.value_spaces['dilation'].value if self.search_dilation else self.dilation
        padding = self.padding

        if self.search_in_channel:
            in_channels = self.value_spaces['in_channels'].value
        if self.search_out_channel:
            out_channels = self.value_spaces['out_channels'].value
            if self.bias:
                bias = bias[:out_channels]
        filters = filters[:out_channels, :in_channels, ...]
        if self.search_kernel_size:
            filters = self.transform_kernel_size(filters)
        if self.auto_padding:
            kernel_size = filters.shape[2:]
            padding = []
            for k in kernel_size:
                padding.append(k//2)

        if isinstance(self.conv, nn.Conv1d):
            return F.conv1d(x, filters, bias, stride, padding, dilation, groups)
        if isinstance(self.conv, nn.Conv2d):
            return F.conv2d(x, filters, bias, stride, padding, dilation, groups)
        if isinstance(self.conv, nn.Conv3d):
            return F.conv3d(x, filters, bias, stride, padding, dilation, groups)

    def transform_kernel_size(self, filters):
        # Todo: support different types of kernel size transformation methods by `transform_kernel_size` function
        sub_kernel_size = self.value_spaces['kernel_size'].value
        start, end = sub_filter_start_end(self.kernel_size, sub_kernel_size)
        if isinstance(self.conv, nn.Conv1d): filters = filters[:, :, start:end]
        if isinstance(self.conv, nn.Conv2d): filters = filters[:, :, start:end, start:end]
        if isinstance(self.conv, nn.Conv3d): filters = filters[:, :, start:end, start:end, start:end]
        return filters

    def sort_weight_bias(self, module):
        if self.search_in_channel:
            vc = self.value_spaces['in_channels']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_channel:
            vc = self.value_spaces['out_channels']
            module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
            if self.bias: module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # conv
        weight = self.conv.weight
        bias = self.conv.bias

        if self.search_in_channel:
            in_channels = self.value_spaces['in_channels'].value
            weight = weight[:, :in_channels, ...]
        if self.search_out_channel:
            out_channels = self.value_spaces['out_channels'].value
            weight = weight[:out_channels, :, ...]
            if bias is not None: bias = bias[:out_channels]
        if self.search_kernel_size:
            kernel_size = self.value_spaces['kernel_size'].value
            start, end = sub_filter_start_end(self.kernel_size, kernel_size)
            shape_size = len(weight.shape)
            if shape_size == 3:
                # 1D conv
                weight = weight[:, :, start:end]
            elif shape_size == 4:
                # 2D conv
                weight = weight[:, :, start:end, start:end]
            else:
                # 3D conv
                weight = weight[:, :, start:end, start:end, start:end]
        parameters = [weight, bias]
        params = sum([p.numel() for p in parameters if p is not None])
        return params


class Conv1d(BaseConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False
        ):
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def init_ops(self, conv_kwargs):
        '''Generate Conv operation'''
        self.conv = nn.Conv1d(**conv_kwargs)


class Conv2d(BaseConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False
        ):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def init_ops(self, conv_kwargs):
        '''Generate Conv operation'''
        self.conv = nn.Conv2d(**conv_kwargs)


class Conv3d(BaseConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[str, int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False
        ):
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode)

    def init_ops(self, conv_kwargs):
        '''Generate Conv operation'''
        self.conv = nn.Conv3d(**conv_kwargs)
