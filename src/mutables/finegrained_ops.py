import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from mutables.mutables import ValueChoice
from utils.average_meter import AverageMeter
from utils.utils import hparams_wrapper

from .masker import __MASKERS__
from .utils import build_activation, sub_filter_start_end


__all__ = [
    'FinegrainedModule',
    'FinegrainedConv2d',
    'FinegrainedConv3d',
    'FinegrainedLinear',
    'FinegrainedBN2d',
    'FinegrainedBN1d',
    'bind_module_to_valueChoice',
    'sortChannels',
    'set_running_statistics',
]


@hparams_wrapper
class FinegrainedModule(nn.Module):
    def __init__(self):
        super(FinegrainedModule, self).__init__()

    def getValueChoices(self, kwargs):
        valueChoices = nn.ModuleDict()
        for key, value in kwargs.items():
            if isinstance(value, ValueChoice):
                valueChoices[key] = value
                if value.index is not None:
                    _v = value.candidates[value.index]
                elif len(value.mask) != 0:
                    index = torch.tensor(value.mask).argmax()
                    _v = value.candidates[index]
                else:
                    _v = value.max_value
                setattr(self, key, _v)
            else:
                setattr(self, key, value)
        return valueChoices

    def __deepcopy__(self, memo):
        try:
            new_instance = self.__class__(**self.hparams)
            device = next(self.parameters()).device
            new_instance.load_state_dict(self.state_dict())
            return new_instance.to(device)
        except Exception as e:
            print(str(e))


class FinegrainedConv2d(FinegrainedModule):
    def __init__(self,
        in_channels, out_channels, kernel_size,
        stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False
        ):
        '''Initialize FinegrainedConv2d
            Example
            >>> import torch, mutables, mutator
            >>> out_channels = mutables.ValueChoice([8,16,24])
            >>> kernel_size = mutables.ValueChoice([3,5])
            >>> x = torch.rand(2,3,16,16)
            >>> conv = FinegrainedConv2d(3, out_channels, kernel_size))
            >>> myMutator = mutator.RandomMutator(conv, None)
            >>> for i in range(2):
            >>>     myMutator.reset()
            >>>     print(myMutator._cache)
            >>>     y = conv(x)
            >>>     print(y.shape)
        '''
        super(FinegrainedConv2d, self).__init__()
        self.valueChoices = self.getValueChoices(self.hparams)
        self.init_ops()
        self.searchConv = self.toSearchConv()
        self.conv_dim = self.__class__.__name__[-2:]

    def init_ops(self):
        '''Generate Conv operation'''
        conv_kwargs = {key:getattr(self, key, None) for key in ['in_channels', 'out_channels', 'kernel_size',
            'stride', 'padding', 'dilation', 'groups', 'bias']}
        conv = nn.Conv2d(**conv_kwargs)
        self.conv = conv

    def toSearchConv(self):
        '''search flag
            - searchConv
            - search_in_channel
            - search_out_channel
            - search_kernel
        '''
        self.search_in_channel = False
        self.search_out_channel = False
        self.search_kernel = False
        if len(self.valueChoices)==0:
            return False
        cout, cin, k, k = self.conv.weight.shape
        if 'in_channels' in self.valueChoices:
            self.search_in_channel = True
        if 'out_channels' in self.valueChoices:
            self.search_out_channel = True
        if 'kernel_size' in self.valueChoices:
            kernel_candidates = self.valueChoices['kernel_size'].candidates
            assert self.kernel_size==max(kernel_candidates)
            max_k = self.kernel_size
            # Todo: 与`transform_kernel_size`搭配使用，目前未使用
            for i, k in enumerate(sorted(kernel_candidates)[:-1]):
                self.register_parameter(f'{max_k}to{k}_kernelMatrix', Parameter(torch.rand(max_k**2, k**2)))
            self.search_kernel = True
        return True

    ###########################################
    # forward implementation
    # - forward_conv
    #   - transform_kernel_size
    ###########################################

    def forward(self, x):
        out = None
        if len(self.valueChoices)==0:
            out = self.conv(x)
        else:
            out = self.forward_conv(self.conv, x)
        return out

    def forward_conv(self, module, x):
        filters = module.weight.contiguous()
        bias = module.bias
        in_channels = None
        out_channels = None
        if self.search_in_channel:
            in_channels = self.valueChoices['in_channels'].value
        if self.search_out_channel:
            out_channels = self.valueChoices['out_channels'].value
            if self.bias:
                bias = bias[:out_channels]
        filters = filters[:out_channels, :in_channels, ...]
        if self.search_kernel:
            # Todo: support different types of kernel size transformation methods by `transform_kernel_size` function
            kernel_size = self.valueChoices['kernel_size'].value
            start, end = sub_filter_start_end(self.kernel_size, kernel_size)
            if isinstance(self.conv, nn.Conv2d): filters = filters[:, :, start:end, start:end]
            if isinstance(self.conv, nn.Conv3d): filters = filters[:, :, start:end, start:end, start:end]
        if isinstance(self.conv, nn.Conv2d):
            return F.conv2d(x, filters, bias, self.stride, self.padding, self.dilation, self.groups)
        if isinstance(self.conv, nn.Conv3d):
            return F.conv3d(x, filters, bias, self.stride, self.padding, self.dilation, self.groups)

    def transform_kernel_size(self, kernel_size, filters):
        # Todo: support different types of kernel size transformation methods by `transform_kernel_size` function
        raise NotImplementedError

    def sort_weight_bias(self, module):
        if self.search_in_channel:
            vc = self.valueChoices['in_channels']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_channel:
            vc = self.valueChoices['out_channels']
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
            in_channels = self.valueChoices['in_channels'].value
            weight = weight[:, :in_channels, ...]
        if self.search_out_channel:
            out_channels = self.valueChoices['out_channels'].value
            weight = weight[:out_channels, :, ...]
            if bias is not None: bias = bias[:out_channels]
        if self.search_kernel:
            kernel_size = self.valueChoices['kernel_size'].value
            start, end = sub_filter_start_end(self.kernel_size, kernel_size)
            if self.conv_dim=='2D':
                weight = weight[:, :, start:end, start:end]
            else:
                weight = weight[:, :, start:end, start:end, start:end]
        parameters = [weight, bias]
        size = sum([p.numel() for p in parameters if p is not None])
        return size


class FinegrainedConv3d(FinegrainedConv2d):
    def __init__(self,
        in_channels, out_channels, kernel_size,
        stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=True
        ):
        '''Initialize FinegrainedConv3d
        '''
        super(FinegrainedConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
            use_bn, eps, momentum, affine, track_running_stats, act_func, ops_order)

    def init_ops(self):
        conv_kwargs = [getattr(self, key) for key in ['in_channels', 'out_channels', 'kernel_size',
            'stride', 'padding', 'dilation', 'groups', 'bias']]
        conv = nn.Conv3d(conv_kwargs)
        self.conv = conv


class FinegrainedLinear(FinegrainedModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(FinegrainedLinear, self).__init__()
        self.valueChoices = self.getValueChoices(self.hparams)
        self.init_ops()
        self.searchLinear = self.toSearchLinear()

    def init_ops(self):
        '''Generate linear operation'''
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    def toSearchLinear(self):
        '''search flag
            searchConv
            search_in_features
            search_out_features
        '''
        self.search_in_features = False
        self.search_out_features = False
        if len(self.valueChoices)==0:
            return False
        cout, cin = self.linear.weight.shape
        if 'in_features' in self.valueChoices:
            self.search_in_features = True
        if 'out_features' in self.valueChoices:
            self.search_out_features = True
        return True

    ###########################################
    # forward implementation
    # - forward_linear
    #   - get_active_weight_bias
    ###########################################

    def forward(self, x):
        out = None
        if len(self.valueChoices)==0:
            out = self.linear(x)
        else:
            out = self.forward_linear(self.linear, x)
        return out

    def forward_linear(self, module, x):
        weight, bias = self.get_active_weight_bias(module)
        y = F.linear(x, weight, bias)
        return y

    def get_active_weight_bias(self, module):
        weight = module.weight.contiguous()
        bias = module.bias
        in_features = None
        out_features = None
        if self.search_in_features:
            in_features = self.valueChoices['in_features'].value
        if self.search_out_features:
            out_features = self.valueChoices['out_features'].value
            if self.bias:
                bias = bias[:out_features]
        weight = weight[:out_features, :in_features]
        return weight, bias

    def sort_weight_bias(self, module):
        if self.search_in_features:
            vc = self.valueChoices['in_features']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_features:
            vc = self.valueChoices['out_features']
            module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
            if self.bias:
                module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # linear
        weight = self.linear.weight
        bias = self.linear.bias
        in_features = None
        out_features = None

        if self.search_in_features:
            in_features = self.valueChoices['in_features'].value
        if self.search_out_features:
            out_features = self.valueChoices['out_features'].value
        weight = weight[:in_features, :out_features]
        if bias is not None: bias = bias[:out_features]
        parameters = [weight, bias]
        size = sum([p.numel() for p in parameters if p is not None])
        return size


class FinegrainedBN2d(FinegrainedModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(FinegrainedBN2d, self).__init__()
        self.valueChoices = self.getValueChoices(self.hparams)
        self.bn = self.init_ops()
        self.searchBN = isinstance(num_features, ValueChoice)

    def init_ops(self):
        bn_kwargs = {key:getattr(self, key, None) 
            for key in ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats']}
        bn = nn.BatchNorm2d(**bn_kwargs)
        return bn

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # bn
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        if 'num_features' in self.valueChoices:
            num_features = self.valueChoices['num_features'].value
            bn_weight = bn_weight[:num_features]
            bn_bias = bn_bias[:num_features]
        size = sum([p.numel() for p in [bn_weight, bn_bias] if p is not None])
        return size

    ###########################################
    # forward implementation
    # - forward_bn
    ###########################################

    def forward(self, x):
        out = None
        if len(self.valueChoices)==0 or self.valueChoices['num_features'].index is not None:
            out = self.bn(x)
        else:
            out = self.forward_bn(self.bn, x)
        return out

    def forward_bn(self, module, x):
        num_features = getattr(self.valueChoices, 'num_features', self.num_features)
        if isinstance(num_features, ValueChoice):
            num_features = num_features.value
        exponential_average_factor = 0.0

        if module.training and module.track_running_stats:
            if module.num_batches_tracked is not None:
                module.num_batches_tracked += 1
                if module.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = module.momentum
        return F.batch_norm(
            x, module.running_mean[:num_features], module.running_var[:num_features], module.weight[:num_features],
            module.bias[:num_features], module.training or not module.track_running_stats,
            exponential_average_factor, module.eps,
        )


    def sort_weight_bias(self, module):
        vc = self.valueChoices['num_features']
        module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
        module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)
        if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            module.running_mean.data = torch.index_select(module.running_mean.data, 0, idx)
            module.running_var.data = torch.index_select(module.running_var.data, 0, idx)


class FinegrainedBN1d(FinegrainedBN2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(FinegrainedBN1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def init_ops(self):
        bn_kwargs = {key:getattr(self, key, None) 
            for key in ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats']}
        bn = nn.BatchNorm1d(**bn_kwargs)
        return bn


def isValueChoice(module):
    return isinstance(module, ValueChoice)

def bind_module_to_valueChoice(net):
    '''若输入通道是ValueChoice，则绑定该Module'''
    def appendBoundName(module, key, module_name):
        if isValueChoice(getattr(module.valueChoices, key, None)):
            module.valueChoices[key].bindModuleNames.append(name)
            module.valueChoices[key].device = next(module.parameters()).device
    names = []
    for name, module in net.named_modules():
        if isinstance(module, FinegrainedModule):
            if isinstance(module, (FinegrainedConv2d, FinegrainedConv3d)):
                appendBoundName(module, 'in_channels', name)
            elif isinstance(module, FinegrainedLinear):
                appendBoundName(module, 'in_features', name)
            elif isinstance(module, (FinegrainedBN2d, FinegrainedBN1d)):
                appendBoundName(module, 'num_features', name)
            names.append(name)
    return names

def sortChannels(net, masker=None):
    def setSortIdx(module, key, module_name, weight_name):
        if isValueChoice(getattr(module.valueChoices, key, None)) and \
            module_name == module.valueChoices[key].lastBindModuleName:
            op = getattr(module, weight_name) # e.g., FinegrainedConv2d.conv, FinegrainedBN2d.bn
            module.valueChoices[key].sortIdx = masker(op)
            # module.sort_weight_bias(op)

    if isinstance(net, nn.DataParallel):
        net = net.module

    if masker is None:
        masker = __MASKERS__['l1']()
    elif isinstance(masker, str):
        masker = __MASKERS__[masker]()

    for name, module in net.named_modules():
        if isinstance(module, FinegrainedModule):
            if isinstance(module, (FinegrainedConv2d, FinegrainedConv3d)):
                setSortIdx(module, 'in_channels', name, 'conv')
            elif isinstance(module, FinegrainedLinear):
                setSortIdx(module, 'in_features', name, 'linear')
            elif isinstance(module, (FinegrainedBN2d, FinegrainedBN1d)):
                setSortIdx(module, 'num_features', name, 'bn')


def set_running_statistics(model, data_loader):
    bn_mean = {}
    bn_var = {}

    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_mean[name] = AverageMeter(name)
            bn_var[name] = AverageMeter(name)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)
                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    if len(bn_mean) == 0:
        # skip if there is no batch normalization layers in the network
        return

    with torch.no_grad():
        device = next(forward_model.parameters()).device
        for images, labels in data_loader:
            images = images.to(device)
            forward_model(images)

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)
