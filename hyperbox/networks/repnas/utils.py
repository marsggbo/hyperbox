from hyperbox.mutables.spaces import OperationSpace
from numpy.lib.arraysetops import isin
import torch
import torch.nn as nn
# from .rep_ops import *
from hyperbox.networks.repnas.rep_ops import *


def fuse(candidates, weights, kernel_size=3):
    k_list = []
    b_list = []

    for i in range(len(candidates)):
        op = candidates[i]
        weight = weights[i].float()
        if op.__class__.__name__ == "DBB1x1kxk":
            if hasattr(op.dbb_1x1_kxk, 'idconv1'):
                k1 = op.dbb_1x1_kxk.idconv1.get_actual_kernel()
            else:
                k1 = op.dbb_1x1_kxk.conv1.weight

            k1, b1 = transI_fusebn(k1, op.dbb_1x1_kxk.bn1)
            k2, b2 = transI_fusebn(op.dbb_1x1_kxk.conv2.weight, op.dbb_1x1_kxk.bn2)

            k, b = transIII_1x1_kxk(k1, b1, k2, b2, groups=op.groups)
        elif op.__class__.__name__ == "DBB1x1":
            k, b = transI_fusebn(op.conv.weight, op.bn)
            k = transVI_multiscale(k, kernel_size)
        elif op.__class__.__name__ == "DBBORIGIN":
            k, b = transI_fusebn(op.conv.weight, op.bn)
        elif op.__class__.__name__ == "DBBAVG":
            ka = transV_avg(op.out_channels, op.kernel_size, op.groups)
            k2, b2 = transI_fusebn(ka.to(op.dbb_avg.avgbn.weight.device), op.dbb_avg.avgbn)

            if hasattr(op.dbb_avg, 'conv'):
                k1, b1 = transI_fusebn(op.dbb_avg.conv.weight, op.dbb_avg.bn)
                k, b = transIII_1x1_kxk(k1, b1, k2, b2, groups=op.groups)
            else:
                k, b = k2, b2
        else:
            raise "TypeError: Not In DBBAVG DBB1x1kxk DBB1x1 DBBORIGIN."
        k_list.append(k.detach() * weight)
        b_list.append(b.detach() * weight)

    return transII_addbranch(k_list, b_list)


def replace(net):
    for name, module in net.named_modules():
        if isinstance(module, OperationSpace):
            candidates = []
            weights = []
            for idx, weight in enumerate(module.mask):
                if weight:
                    candidates.append(module.candidates_original[idx])
                    weights.append(weight)
            ks = max([c_.kernel_size for c_ in candidates])
            k, b = fuse(candidates, weights, ks)
            first = module.candidates_original[0]
            inc = first.in_channels
            ouc = first.out_channels
            s = first.stride
            p = ks//2
            g = first.groups
            reparam = nn.Conv2d(in_channels=inc, out_channels=ouc, kernel_size=ks,
                                stride=s, padding=p, dilation=1, groups=g)
            reparam.weight.data = k
            reparam.bias.data = b

            module.candidates_original = [reparam]
            module.candidates = torch.nn.ModuleList([reparam])
            module.mask = torch.tensor([True])
