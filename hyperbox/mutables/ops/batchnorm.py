
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbox.mutables.spaces import ValueSpace

from .base_module import FinegrainedModule
from .utils import is_searchable


__all__ = [
    'BaseBatchNorm',
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d'
]


class BaseBatchNorm(FinegrainedModule):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BaseBatchNorm, self).__init__()
        bn_kwargs = {key:getattr(self, key, None) 
            for key in ['num_features', 'eps', 'momentum', 'affine', 'track_running_stats']}
        self.init_ops(bn_kwargs)
        self.searchBN = is_searchable(getattr(self.value_spaces, 'num_features', None))

    def init_ops(self, bn_kwargs: dict):
        raise NotImplementedError

    ###########################################
    # property
    ###########################################

    @property
    def params(self):
        '''The number of the trainable parameters'''
        # bn
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        if 'num_features' in self.value_spaces:
            num_features = self.value_spaces['num_features'].value
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
        if len(self.value_spaces)==0 or self.value_spaces['num_features'].index is not None:
            out = self.bn(x)
        else:
            out = self.forward_bn(self.bn, x)
        return out

    def forward_bn(self, module, x):
        num_features = getattr(self.value_spaces, 'num_features', self.num_features)
        if isinstance(num_features, ValueSpace):
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
        vc = self.value_spaces['num_features']
        module.weight.data = torch.index_select(module.weight.data, 0, vc.sortIdx)
        module.bias.data = torch.index_select(module.bias.data, 0, vc.sortIdx)
        if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            module.running_mean.data = torch.index_select(module.running_mean.data, 0, idx)
            module.running_var.data = torch.index_select(module.running_var.data, 0, idx)


class BatchNorm1d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def init_ops(self, bn_kwargs: dict):
        self.bn = nn.BatchNorm1d(**bn_kwargs)


class BatchNorm2d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def init_ops(self, bn_kwargs: dict):
        self.bn = nn.BatchNorm2d(**bn_kwargs)


class BatchNorm3d(BaseBatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm3d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def init_ops(self, bn_kwargs: dict):
        self.bn = nn.BatchNorm3d(**bn_kwargs)
