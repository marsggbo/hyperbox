
from typing import Union, Optional, Set, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter

from hyperbox.mutables.spaces import ValueSpace
from hyperbox.mutables.ops.base_module import FinegrainedModule
from hyperbox.mutables.ops.utils import sub_filter_start_end, is_searchable


__all__ = [
    'BaseConvNd',
    'Conv1d',
    'Conv2d',
    'Conv3d'
]


class BaseConvNd(FinegrainedModule):
    KERNEL_TRANSFORM_MODE = 1 # None or 1 or other settings

    def init(
        self,
        in_channels: Union[int, tuple, ValueSpace],
        out_channels: Union[int, tuple, ValueSpace],
        kernel_size: Union[int, tuple, ValueSpace],
        stride: Union[int, tuple, ValueSpace],
        padding: Union[str, int, tuple, ValueSpace],
        dilation: Union[int, ValueSpace],
        groups: Union[int, ValueSpace],
        bias: bool,
        padding_mode: str,
        auto_padding: bool = False,
        device=None,
        dtype=None,
        transposed: bool = False,
        output_padding: Tuple[int, ...] = 0,
        **kwargs
    ):
        '''Base Conv Module
        Args:
            auto_padding: if set to true, will set a proper padding size to make output size same as the input size.
                For example, if kernel size is 3, the padding size is 1;
                if kernel_size is (3,7), the padding size is (1, 3)
        '''
        super(BaseConvNd, self).__init__()
        self.conv_dim = self.__class__.__name__[-2]
        assert self.conv_dim in ['1', '2', '3'], 'conv_dim must be 1, 2 or 3'
        self.conv_dim = int(self.conv_dim)
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        _in_channels = in_channels.max_value if isinstance(in_channels, ValueSpace) else in_channels
        _out_channels = out_channels.max_value if isinstance(out_channels, ValueSpace) else out_channels
        _kernel_size = kernel_size.max_value if isinstance(kernel_size, ValueSpace) else kernel_size
        _stride = stride.max_value if isinstance(stride, ValueSpace) else stride
        _padding = padding.max_value if isinstance(padding, ValueSpace) else padding
        _dilation = dilation.min_value if isinstance(dilation, ValueSpace) else dilation
        _groups = groups
        if isinstance(groups, ValueSpace):
            if isinstance(in_channels, ValueSpace):
                if groups is not in_channels:
                    print('groups must be the same as in_channels when in_channels is ValueSpace')
                    groups = in_channels
                _groups = groups.max_value
            else:
                _groups = groups.min_value

        self.format_args(_kernel_size, _stride, _padding, _dilation)
        self.conv_kwargs = kwargs
        self.conv_kwargs.update({
            'in_channels': _in_channels,
            'out_channels': _out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': _groups,
            'bias': bias,
            'padding_mode': padding_mode,
            'device': device,
            'dtype': dtype,
            # 'transposed': transposed,
            # 'output_padding': self.output_padding,
        })
        self.is_search = self.isSearchConv()
    
    def init_kernel_transform_matrix(self, kernel_size):
        if not isinstance(kernel_size, ValueSpace) or \
            self.KERNEL_TRANSFORM_MODE is None:
                return
        # register scaling parameters
        # 7to5_matrix, 5to3_matrix
        kernel_size = self.value_spaces['kernel_size'].candidates_original
        scale_params = {}
        for i in range(len(kernel_size) - 1):
            ks_small = kernel_size[i]
            ks_larger = kernel_size[i + 1]
            param_name = '%dto%d' % (ks_larger, ks_small)
            scale_params['%s_matrix' % param_name] = Parameter(torch.eye(ks_small ** self.conv_dim))
        for name, param in scale_params.items():
            self.register_parameter(name, param)

    def format_args(self, *args, **kwargs):
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

        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False

        if  is_searchable(getattr(self.value_spaces, 'in_channels', None)):
            self.search_in_channel = True
        if  is_searchable(getattr(self.value_spaces, 'out_channels', None)):
            self.search_out_channel = True
        if  is_searchable(getattr(self.value_spaces, 'kernel_size', None)):
            kernel_candidates = self.value_spaces['kernel_size'].candidates
            max_k = self.kernel_size
            # Todo: 与`transform_kernel_size`搭配使用，目前未使用
            self.search_kernel_size = True
        if  is_searchable(getattr(self.value_spaces, 'stride', None)):
            self.search_stride = True
        if  is_searchable(getattr(self.value_spaces, 'dilation', None)):
            self.search_dilation = True
        if  is_searchable(getattr(self.value_spaces, 'groups', None)):
            self.search_groups = True
        # if  is_searchable(getattr(self.value_spaces, 'bias', None)):
        #     self.search_bias = True

        return True

    ###########################################
    # forward implementation
    # - forward_conv
    #   - transform_kernel_size
    ###########################################

    def forward_conv(self, x):
        filters = self.weight.contiguous()
        bias = self.bias
        in_channels = self.in_channels
        out_channels = self.out_channels
        stride = self.value_spaces['stride'].value if self.search_stride else self.stride
        groups = self.value_spaces['groups'].value if self.search_groups else self.groups
        dilation = self.value_spaces['dilation'].value if self.search_dilation else self.dilation
        padding = self.padding

        if self.search_in_channel:
            in_channels = self.value_spaces['in_channels'].value
            filters = filters[:, :in_channels, ...]
        if self.search_out_channel:
            out_channels = self.value_spaces['out_channels'].value
            if self.bias is not None:
                bias = bias[:out_channels]
            filters = filters[:out_channels, ...]
        if self.search_kernel_size:
            filters = self.transform_kernel_size(filters)
        if self.search_groups and not self.search_in_channel:
            filters = self.get_filters_by_groups(filters, in_channels, groups).contiguous()
        if self.auto_padding:
            kernel_size = filters.shape[2:]
            padding = []
            for k in kernel_size:
                padding.append(k//2)
        return self.conv(x, filters, bias, stride, padding, dilation, groups)

    def get_filters_by_groups(self, filters, in_channels, groups):
        '''Get filters when searching for #of groups'''
        if groups == 1:
            return filters
        sub_in_channels = in_channels // groups
        sub_ratio = filters.size(1) // sub_in_channels

        filter_crops = []
        sub_filters = torch.chunk(filters, groups, dim=0)
        for i, sub_filter in enumerate(sub_filters):
            part_id = i % sub_ratio
            start = part_id * sub_in_channels
            filter_crops.append(sub_filter[:, start:start + sub_in_channels, :, :])
        filters = torch.cat(filter_crops, dim=0)
        return filters

    def transform_kernel_size(self, filters):        
        if self.KERNEL_TRANSFORM_MODE is None:
            # print('vanilla transform_kernel_size')
            sub_kernel_size = self.value_spaces['kernel_size'].value
            start, end = sub_filter_start_end(self.kernel_size, sub_kernel_size)
            if self.conv_dim==1: filters = filters[:, :, start:end]
            if self.conv_dim==2: filters = filters[:, :, start:end, start:end]
            if self.conv_dim==3: filters = filters[:, :, start:end, start:end, start:end]
        else:
            max_kernel_size = self.kernel_size
            if isinstance(max_kernel_size, (tuple, list)):
                max_kernel_size = max(max_kernel_size)
            sub_kernel_size = self.value_spaces['kernel_size'].value
            ks_set = self.value_spaces['kernel_size'].candidates
            if sub_kernel_size < max_kernel_size:
                start_filter = filters
                for i in range(len(ks_set)-1, 0, -1):
                    src_ks = ks_set[i]
                    if src_ks <= sub_kernel_size:
                        break
                    target_ks = ks_set[i - 1]
                    start, end = sub_filter_start_end(src_ks, target_ks)
                    if self.conv_dim==1: _input_filter = start_filter[:, :, start:end]
                    elif self.conv_dim==2: _input_filter = start_filter[:, :, start:end, start:end]
                    elif self.conv_dim==3: _input_filter = start_filter[:, :, start:end, start:end, start:end]
                    _input_filter = _input_filter.contiguous()
                    _input_filter = _input_filter.view(_input_filter.size(0), _input_filter.size(1), -1)
                    _input_filter = _input_filter.view(-1, _input_filter.size(2))
                    _input_filter = F.linear(
                        _input_filter, getattr(self, '%dto%d_matrix' % (src_ks, target_ks)),
                    )
                    _input_filter = _input_filter.view(filters.size(0), filters.size(1), target_ks ** self.conv_dim)
                    _input_filter = _input_filter.view(filters.size(0), filters.size(1), *([target_ks]*self.conv_dim))
                    start_filter = _input_filter
                filters = start_filter
        return filters

    def sort_weight_bias(self, module):
        if self.search_in_channel:
            vc = self.value_spaces['in_channels']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_channel:
            vc = self.value_spaces['out_channels']
            module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
            if self.bias is not None:
                module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # conv
        weight = self.weight
        bias = self.bias

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


class Conv1d(nn.Conv1d, BaseConvNd):
    def __init__(
        self,
        in_channels: Union[int, tuple, ValueSpace],
        out_channels: Union[int, tuple, ValueSpace],
        kernel_size: Union[_size_1_t, ValueSpace],
        stride: Union[_size_1_t, ValueSpace] = 1,
        padding: Union[str, _size_1_t, ValueSpace] = 0,
        dilation: Union[_size_1_t, ValueSpace] = 1,
        groups: Union[int, ValueSpace] = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        auto_padding: bool = False,
        **kwargs
    ):
        self.init(in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, auto_padding, device, dtype, **kwargs)
        nn.Conv1d.__init__(self, **self.conv_kwargs)
        self.init_kernel_transform_matrix(kernel_size)

    def forward(self, x):
        out = None
        if not self.is_search:
            padding = self.padding
            if self.auto_padding:
                kernel_size = self.weight.shape[2:]
                padding = []
                for k in kernel_size:
                    padding.append(k//2)
                self.padding = padding
            out = nn.Conv1d.forward(self, x)
        else:
            out = self.forward_conv(x)
        return out

    def format_args(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
    ):
        '''Generate Conv operation'''
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = padding if isinstance(padding, str) else _single(padding)
        self.dilation = _single(dilation)
        self.output_padding = _single(0)
        self.conv = F.conv1d


class Conv2d(nn.Conv2d, BaseConvNd):
    def __init__(
        self,
        in_channels: Union[int, tuple, ValueSpace],
        out_channels: Union[int, tuple, ValueSpace],
        kernel_size: Union[int, ValueSpace],
        stride: Union[int, ValueSpace] = 1,
        padding: Union[str, int, ValueSpace] = 0,
        dilation: Union[int, ValueSpace] = 1,
        groups: Union[int, ValueSpace] = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False,
        device=None,
        dtype=None,
        **kwargs
    ):
        self.init(in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, auto_padding, device, dtype, **kwargs)
        nn.Conv2d.__init__(self, **self.conv_kwargs)
        self.init_kernel_transform_matrix(kernel_size)

    def format_args(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
    ):
        '''Generate Conv operation'''
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.output_padding = _pair(0)
        self.conv = F.conv2d


class Conv3d(nn.Conv3d, BaseConvNd):
    def __init__(
        self,
        in_channels: Union[int, ValueSpace],
        out_channels: Union[int, ValueSpace],
        kernel_size: Union[int, ValueSpace],
        stride: Union[int, ValueSpace] = 1,
        padding: Union[int, ValueSpace] = 0,
        dilation: Union[int, ValueSpace] = 1,
        groups: Union[int, ValueSpace] = 1,
        bias: Union[str, ValueSpace] = True,
        padding_mode: str = 'zeros',
        auto_padding: bool = False,
        device=None,
        dtype=None,
        **kwargs
    ):
        self.init(in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, padding_mode, auto_padding, device, dtype, **kwargs)
        nn.Conv3d.__init__(self, **self.conv_kwargs)
        self.init_kernel_transform_matrix(kernel_size)

    def format_args(
        self, 
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        bias: bool = True
    ):
        '''Generate Conv operation'''
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.dilation = _triple(dilation)
        self.output_padding = _triple(0)
        self.conv = F.conv3d


if __name__ == '__main__':
    from time import time
    import torch
    from hyperbox.mutator import RandomMutator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    steps = 50
    # 1d
    oc = ValueSpace(candidates=[3,10])
    ks = ValueSpace(candidates=[3,5,7])
    op = Conv1d(in_channels=3, out_channels=oc, kernel_size=ks, bias=True, auto_padding=True).to(device)
    rm = RandomMutator(op)
    print('*'*80)
    x = torch.rand(2,3,16).to(device)
    start = time()
    for i in range(steps):
        rm.reset()
        y = op(x)
        # print(y.shape)
    end = time()
    print(f"testing 1d {op}, cost {end-start:.2f} s")

    # 2d
    oc = ValueSpace(candidates=[9,18])
    ks = ValueSpace(candidates=[3,5,7])
    groups = ValueSpace(candidates=[1,3])
    op = Conv2d(in_channels=9, out_channels=36, kernel_size=ks, groups=groups).to(device)
    rm = RandomMutator(op)
    print('*'*80)
    x = torch.rand(2,9,16,16).to(device)
    start = time()
    for i in range(steps):
        rm.reset()
        y = op(x)
        # print(y.shape)
    end = time()
    print(f"testing 2d {op}, cost {end-start:.2f} s")

    # 3d
    oc = ValueSpace(candidates=[3,10])
    ks = ValueSpace(candidates=[3,5,7])
    op = Conv3d(in_channels=3, out_channels=oc, kernel_size=ks).to(device)
    rm = RandomMutator(op)
    print('*'*80)
    x = torch.rand(2,3,16,16,16).to(device)
    start = time()
    for i in range(steps):
        rm.reset()
        y = op(x)
        # print(y.shape)
    end = time()
    print(f"testing 3d {op}, cost {end-start:.2f} s")
