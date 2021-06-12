# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbox.mutables import InputChoice, LayerChoice, ValueChoice

from .default_mutator import Mutator

__all__ = [
    'OnehotMutator',
]


class OnehotMutator(Mutator):
    def __init__(self, model, cfg=None):
        super().__init__(model)
        self.choices = nn.ParameterDict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                # self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length+1))
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
            if isinstance(mutable, ValueChoice):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.length))
                mutable.mask = self.choices[mutable.key].data
            elif isinstance(mutable, InputChoice):
                self.choices[mutable.key] = nn.Parameter(1.0E-3 * torch.randn(mutable.n_candidates))

    def device(self):
        for v in self.choices.values():
            return v.device

    def sample_search(self):
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice):
                # result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()[:-1]
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()
            elif isinstance(mutable, ValueChoice):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()
                mutable.mask.data = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).data
            elif isinstance(mutable, InputChoice):
                result[mutable.key] = F.gumbel_softmax(self.choices[mutable.key], hard=True, dim=-1).bool()
        return result

    def sample_final(self):
        return self.sample_search()