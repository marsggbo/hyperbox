
from .base_transforms import BaseTransforms
from .torch_transforms import TorchTransforms
from .cutout import Cutout

import importlib
alb = importlib.util.find_spec('albumentations')
if alb is not None:
    from .albumentation_transforms import AlbumentationsTransforms

def get_transforms(name, kwargs: dict):
    if name == 'torch':
        from .torch_transforms import TorchTransforms
        T = TorchTransforms(**kwargs)
    elif name == 'albumentation':
        from .albumentation_transforms import AlbumentationsTransforms
        T = AlbumentationsTransforms(**kwargs)
    return T
