import os

import torch
import torch.nn as nn

from omegaconf.listconfig import ListConfig
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import *

from hyperbox.mutables import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork

__all__ = [
    'DataAugmentation',
    'DAOperation3D'
]


def prob_list_gen(func, *args, **kwargs):
    return [func(p=p, *args, **kwargs) for p in [i*0.25 for i in range(4)]] + [nn.Identity()]

def DAOperation3D(affine_degree=30, affine_scale=(1.1, 1.5), affine_shears=20, rotate_degree=30, crop_size=128):
    # RandomHorizontalFlip3D(same_on_batch=False, p=0.5),
    # RandomVerticalFlip3D(same_on_batch=False, p=0.5),
    # RandomRotation3D(rotate_degree, same_on_batch=False, p=0.5),
    # RandomAffine3D(affine_degree, same_on_batch=False, p=0.5),
    # RandomCrop3D(crop_size, pad_if_needed=False, same_on_batch=False, p=0.5),
    # RandomEqualize3D(same_on_batch=False, p=0.5)
    ops = {}
    ops['dflip'] = prob_list_gen(RandomDepthicalFlip3D, same_on_batch=False)
    ops['hflip'] = prob_list_gen(RandomHorizontalFlip3D, same_on_batch=False)
    ops['vflip'] = prob_list_gen(RandomVerticalFlip3D, same_on_batch=False)
    ops['equal'] = prob_list_gen(RandomEqualize3D, same_on_batch=False)
    # ops['rotate'] = prob_list_gen(RandomRotation3D, same_on_batch=False, degrees=rotate_degree)
    ops['affine'] = prob_list_gen(
        RandomAffine3D,
        same_on_batch=False, degrees=affine_degree, scale=affine_scale, shears=affine_shears)
    if isinstance(crop_size, int):
        rcrop = prob_list_gen(RandomCrop3D, same_on_batch=False, size=crop_size)
    elif isinstance(crop_size, (ListConfig, list, tuple)):
        rcrop = []
        for size in crop_size:
            rcrop.extend(prob_list_gen(RandomCrop3D, same_on_batch=False, size=size))
    ops['rcrop'] = rcrop
    return ops


class DataAugmentation(BaseNASNetwork):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(
        self,
        rotate_degree=30, crop_size=[(32,128,128), (16,128,128)],
        affine_degree=0, affine_scale=(1.1, 1.5), affine_shears=20,
        # norm_mean=[0.6075, 0.4564, 0.4182], norm_std=[0.2158, 0.1871, 0.1826],
        mask=None
    ):
        super().__init__(mask)
        self.ops = DAOperation3D(affine_degree, affine_scale, affine_shears, rotate_degree, crop_size)
        transforms = []
        for key, value in self.ops.items():
            transforms.append(OperationSpace(candidates=value, key=key, mask=self.mask))
        self.transforms = nn.Sequential(*transforms)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x)  # BxCXDxHxW
        return x_out

    @property
    def arch(self):
        _arch = []
        for op in self.transforms:
            mask = op.mask
            if 'bool' in str(mask.dtype):
                index = mask.int().argmax()
            else:
                index = mask.float().argmax()
            _arch.append(f"{op.candidates[index]}")
        _arch = '\n'.join(_arch)
        return _arch

if __name__ == '__main__':
    from hyperbox.mutator import DartsMutator, RandomMutator
    x = torch.rand(2,1,6,300,300)
    op = DataAugmentation(crop_size=[(2,128,128), (3,256,256), (6,200,200)])
    m = RandomMutator(op)
    # m = DartsMutator(op)
    for i in range(10):
        m.reset()
        print(op.arch)
        y = op(x)
        print(y.shape)

