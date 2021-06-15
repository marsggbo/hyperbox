from typing import Optional, Union

import torch
import torch.nn as nn

from ..spaces import Mutable

def is_searchable(
    obj: Optional[Union[None, Mutable]]
    ):
    '''Check whether the Space obj is searchable'''
    if (obj is None) or (not obj.is_search):
        return False
    return True

def sub_filter_start_end(kernel_size, sub_kernel_size):
	center = kernel_size // 2
	dev = sub_kernel_size // 2
	start, end = center - dev, center + dev + 1
	assert end - start == sub_kernel_size
	return start, end

def build_activation(act_func, inplace=True):
    if act_func is None:
        return None
    elif act_func.lower() == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func.lower() == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func.lower() == 'tanh':
        return nn.Tanh()
    elif act_func.lower() == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('do not support: %s' % act_func)
