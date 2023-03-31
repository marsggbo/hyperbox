
import numpy as np
import torch
import torch.nn as nn

from hyperbox.mutables.spaces import ValueSpace
from hyperbox.utils.utils import hparams_wrapper

__all__ = [
    'FinegrainedModule'
]


@hparams_wrapper
class FinegrainedModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FinegrainedModule, self).__init__()
        # The decorator @hparams_wrapper can automatically save all input arguments to
        # ``hparams`` attribute
        self.value_spaces = self.getValueSpaces(self.hparams)

    def getValueSpaces(self, kwargs):
        value_spaces = nn.ModuleDict()
        for key, value in kwargs.items():
            if key in ['weight', 'bias']:
                if hasattr(self, key): delattr(self, key)
                key = '_' + key
            if isinstance(value, ValueSpace):
                value_spaces[key] = value
                if value.index is not None:
                    _v = value.candidates_original[value.index]
                elif len(value.mask) != 0:
                    if isinstance(value.mask, torch.Tensor):
                        index = value.mask.clone().detach().cpu().numpy().argmax()
                    else:
                        index = np.array(value.mask).argmax()
                    _v = value.candidates_original[index]
                else:
                    _v = value.max_value
                setattr(self, key, _v)
            else:
                setattr(self, key, value)
        return value_spaces

    def __deepcopy__(self, memo):
        try:
            new_instance = self.__class__(**self.hparams)
            device = next(self.parameters()).device
            new_instance.load_state_dict(self.state_dict())
            return new_instance.to(device)
        except Exception as e:
            print(str(e))
