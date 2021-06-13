
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_module import FinegrainedModule


__all__ = [
    'Linear'
]


class Linear(FinegrainedModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(Linear, self).__init__()
        self.value_spaces = self.getValueSpaces(self.hparams)
        self.init_ops()
        self.is_search = self.isSearchLinear()

    def init_ops(self):
        '''Generate linear operation'''
        self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

    def isSearchLinear(self):
        '''search flag
            search
            search_in_features
            search_out_features
        '''
        self.search_in_features = False
        self.search_out_features = False
        if len(self.value_spaces)==0:
            return False
        cout, cin = self.linear.weight.shape
        if 'in_features' in self.value_spaces:
            self.search_in_features = True
        if 'out_features' in self.value_spaces:
            self.search_out_features = True
        return True

    ###########################################
    # forward implementation
    # - forward_linear
    #   - get_active_weight_bias
    ###########################################

    def forward(self, x):
        out = None
        if not self.is_search:
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
            in_features = self.value_spaces['in_features'].value
        if self.search_out_features:
            out_features = self.value_spaces['out_features'].value
            if self.bias:
                bias = bias[:out_features]
        weight = weight[:out_features, :in_features]
        return weight, bias

    def sort_weight_bias(self, module):
        if self.search_in_features:
            vc = self.value_spaces['in_features']
            module.weight.data = torch.index_select(module.weight.data, 1, vc.sortIdx)
        if self.search_out_features:
            vc = self.value_spaces['out_features']
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
            in_features = self.value_spaces['in_features'].value
        if self.search_out_features:
            out_features = self.value_spaces['out_features'].value
        weight = weight[:in_features, :out_features]
        if bias is not None: bias = bias[:out_features]
        parameters = [weight, bias]
        size = sum([p.numel() for p in parameters if p is not None])
        return size
