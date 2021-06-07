from typing import Optional, Union
import torch.nn as nn

from utils.utils import load_json


class BaseNASNetwork(nn.Module):
    def __init__(self, mask: Optional[Union[str, dict]]=None):
        super(BaseNASNetwork, self).__init__()
        if mask is None:
            self.is_search=True
        elif isinstance(mask, str):
            mask = load_json(mask)
            self.is_search = False
        elif isinstance(mask, dict):
            self.is_search = False
        self._mask = mask

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, new_mask):
        self._mask = new_mask

    @property
    def arch(self):
        '''return the current arch encoding'''
        raise NotImplementedError

    def export_model_from_mask(self, mask: Optional[dict]=None):
        # Todo
        if mask is not None:
            self.mask = mask
        # iteratate all mutables

    def load_state_dict(self, state_dict, **kwargs):
        model_dict = self.state_dict()
        for key in state_dict:
            if 'total_ops' in key or \
                'total_params' in key or \
                'module_used' in key:
                continue
            else:
                model_dict[key] = state_dict[key]
        super(self.__class__, self).load_state_dict(model_dict, **kwargs)

    def build_subnet(self, mask):
        '''build subnet by the given mask'''
        raise NotImplementedError

    def load_subnet_state_dict(self, state_dict, **kwargs):
        '''load subnet state dict from the given state_dict'''
        def sub_filter_start_end(kernel_size, sub_kernel_size):
            center = kernel_size // 2
            dev = sub_kernel_size // 2
            start, end = center - dev, center + dev + 1
            assert end - start == sub_kernel_size
            return start, end

        model_dict = self.state_dict()
        for key in state_dict:
            if 'total_ops' in key or \
                'total_params' in key or \
                'module_used' in key or \
                'mask' in key:
                continue
            if model_dict[key].shape == state_dict[key].shape:
                model_dict[key] = state_dict[key]
            else:
                shape = model_dict[key].shape
                if len(shape) == 1:
                    # e.g., bias, BN
                    model_dict[key].data = state_dict[key].data[:shape[0]]
                if len(shape) == 2:
                    # e.g., linear weight
                    _out, _in = shape
                    model_dict[key].data = state_dict[key].data[:_out, :_in]
                if len(shape) == 4:
                    # e.g., conv weight
                    _out, _in, k, k = shape
                    k_larger = state_dict[key].shape[-1]
                    start, end = sub_filter_start_end(k_larger, k)
                    model_dict[key].data = state_dict[key].data[:_out, :_in, start:end, start:end]
        super(self.__class__, self).load_state_dict(model_dict, **kwargs, strict=False)