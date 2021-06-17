from typing import Optional, Union, Tuple, List
import torch.nn as nn

from hyperbox.utils.utils import load_json
from hyperbox.utils.calc_model_size import flops_size_counter


class BaseNASNetwork(nn.Module):
    def __init__(self, mask: Optional[Union[str, dict]]=None):
        super(BaseNASNetwork, self).__init__()
        if mask is None or mask == '':
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

    def arch_size(
        self,
        datasize: Optional[Union[Tuple, List]] = None,
        convert: bool = True,
        verbose: bool = False
    ):
        size = datasize
        assert size is not None, \
            "Please specify valid data size, e.g., size=self.arch_size(datasize=(1,3,32,32))"
        result = flops_size_counter(self, size, convert, verbose)
        mflops, mb_size = list(result.values())
        return mflops, mb_size   

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
                dim = len(model_dict[key].shape)
                if dim == 1:
                    # e.g., bias
                    model_dict[key].data = state_dict[key].data[:shape[0]]
                if dim == 2:
                    # e.g., linear weight
                    _out, _in = shape
                    model_dict[key].data = state_dict[key].data[:_out, :_in]
                if dim >= 3:
                    # e.g., conv weight
                    _out, _in, k = shape[:3]
                    k_larger = state_dict[key].shape[-1]
                    start, end = sub_filter_start_end(k_larger, k)
                    if dim == 3: # conv1d
                        model_dict[key].data = state_dict[key].data[:_out, :_in, start:end]
                    elif dim == 4: #conv2d
                        model_dict[key].data = state_dict[key].data[:_out, :_in, start:end, start:end]
                    else:
                        model_dict[key].data = state_dict[key].data[:_out, :_in, start:end, start:end, start:end]
        super(self.__class__, self).load_state_dict(model_dict, **kwargs, strict=False)
