import numpy as np
import omegaconf
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbox.utils.average_meter import AverageMeter
from hyperbox.mutables.ops import BaseConvNd, BaseBatchNorm, Linear, FinegrainedModule
from hyperbox.mutables.masker import __MASKERS__
from hyperbox.mutables.spaces import ValueSpace
from hyperbox.mutator.fixed_mutator import FixedArchitecture

from hyperbox.networks.pytorch_modules import Hsigmoid, Hswish


__all__ = [
    'isValueSpace', 'bind_module_to_ValueSpace', 'sortChannels', 'set_running_statistics',
    'val2list', 'make_divisible', 'build_activation', 'get_same_padding'
]


def isValueSpace(module):
    return isinstance(module, ValueSpace)

def bind_module_to_ValueSpace(net):
    '''若输入通道是ValueSpace，则绑定该Module'''
    def appendBoundName(module, key, module_name):
        if isValueSpace(getattr(module.value_spaces, key, None)):
            module.value_spaces[key].bindModuleNames.append(name)
            module.value_spaces[key].device = next(module.parameters()).device
    names = []
    for name, module in net.named_modules():
        if isinstance(module, FinegrainedModule):
            if isinstance(module, BaseConvNd):
                appendBoundName(module, 'in_channels', name)
            elif isinstance(module, Linear):
                appendBoundName(module, 'in_features', name)
            elif isinstance(module, BaseBatchNorm):
                appendBoundName(module, 'num_features', name)
            names.append(name)
    return names

def sortChannels(net, masker=None):
    def setSortIdx(module, key, module_name, weight_name):
        if isValueSpace(getattr(module.value_spaces, key, None)) and \
            module_name == module.value_spaces[key].lastBindModuleName:
            op = getattr(module, weight_name) # e.g., Conv2d.conv, BatchNorm2d.bn
            module.value_spaces[key].sortIdx = masker(op)
            # module.sort_weight_bias(op)

    if isinstance(net, nn.DataParallel):
        net = net.module

    if masker is None:
        masker = __MASKERS__['l1']()
    elif isinstance(masker, str):
        masker = __MASKERS__[masker]()

    for name, module in net.named_modules():
        if isinstance(module, FinegrainedModule):
            if isinstance(module, (Conv2d, Conv3d)):
                setSortIdx(module, 'in_channels', name, 'conv')
            elif isinstance(module, Linear):
                setSortIdx(module, 'in_features', name, 'linear')
            elif isinstance(module, (BatchNorm2d, BatchNorm1d)):
                setSortIdx(module, 'num_features', name, 'bn')

def set_running_statistics(model, data_loader, mutator=None, mask=None):
    bn_mean = {}
    bn_var = {}

    device = next(model.parameters()).device
    try:
        forward_model = model.copy() # only support `BaseNASNetwork`
    except:
        forward_model = copy.deepcopy(model) # normal `nn.Module`
    forward_model.to(device)
    if mutator is not None and model.mask is None:
        '''
        if the model needs to be seached (model.mask is None),
        then we need to build a mutator to get the same subnet by setting the same mask
        '''
        try:
            new_mutator = FixedArchitecture(model, mask)
            new_mutator.sample_by_mask(mask)
        except:
            new_mutator = mutator.__class__(forward_model)
            new_mutator.sample_by_mask(mutator._cache)
    for name, m in forward_model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if not m.affine:
                continue
            bn_mean[name] = AverageMeter(name)
            bn_var[name] = AverageMeter(name)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    if len(x.shape)==3:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True)  # 1, C, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True)
                    elif len(x.shape)==4:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                    elif len(x.shape)==5:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)  # 1, C, 1, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)

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
        for images, labels in data_loader:
            images = images.to(device)
            forward_model(images)

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)

def val2list(val, repeat_time=1):
	if isinstance(val, (list, np.ndarray, omegaconf.listconfig.ListConfig)):
		return val
	elif isinstance(val, tuple):
		return list(val)
	else:
		return [val for _ in range(repeat_time)]

def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def min_divisible_value(n1, v1):
	""" make sure v1 is divisible by n1, otherwise decrease v1 """
	if v1 >= n1:
		return n1
	while n1 % v1 != 0:
		v1 -= 1
	return v1

def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2

def build_activation(act_func, inplace=True):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'h_swish':
        return Hswish(inplace=inplace)
    elif act_func == 'h_sigmoid':
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == 'none':
        return None
    else:
        raise ValueError('do not support: %s' % act_func)

def extract_net_from_ckpt(ckpt: str) -> dict:
    ckpt = torch.load(ckpt, map_location='cpu')
    net_weight = ckpt['state_dict'] # keys: 'network.stem', 'network.cells', 'mutator...'

    # extract state_dict of network
    to_copy_weight = {}
    for key, value in net_weight.items():
        if 'network.' in key:
            to_copy_weight[key.replace('network.', '', 1)] = value
    return to_copy_weight