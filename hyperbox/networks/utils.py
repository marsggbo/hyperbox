
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbox.utils.average_meter import AverageMeter
from hyperbox.mutables.ops import BaseConvNd, BaseBatchNorm, Linear, FinegrainedModule
from hyperbox.mutables.masker import __MASKERS__
from hyperbox.mutables.spaces import ValueSpace


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
