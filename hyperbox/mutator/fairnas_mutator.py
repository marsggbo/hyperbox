import math

import numpy as np
import torch
import torch.nn.functional as F

from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace

from hyperbox.mutator.default_mutator import Mutator

__all__ = [
    'FairNASMutator',
]



def lcm(s):
    '''
    gcd: greated common divisor
    lcm: least common multiple
    '''
    assert len(s) >= 2, 'the input of `lcm` must be a list of at least 2 elements'
    lcm = 1
    for x in s:
        lcm = lcm * x // math.gcd(lcm, x)
    return lcm


class FairNASMutator(Mutator):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model)
        self.search_space = self.parse_search_space()
        self.idx = 0

    def parse_search_space(self):
        self.mutable_length = {}
        for mutable in self.mutables:
            self.mutable_length[mutable.key] = mutable.length
        self.lcm = lcm(list(self.mutable_length.values()))
        search_space = {}
        for mutable in self.mutables:
            search_space[mutable.key] = list(range(self.mutable_length[mutable.key])) * (self.lcm // self.mutable_length[mutable.key])
        return search_space

    def shuffle_search_space(self):
        self.idx = 0
        for key, value in self.search_space.items():
            np.random.shuffle(value)

    def sample_search(self):
        if self.idx >= self.lcm:
            self.shuffle_search_space()
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, OperationSpace):
                gen_index = torch.tensor(self.search_space[mutable.key][self.idx])
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = result[mutable.key].detach()
            elif isinstance(mutable, InputSpace):
                if mutable.n_chosen is None:
                    gen_index = torch.tensor(self.search_space[mutable.key][self.idx])
                    result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.n_candidates).view(-1).bool()
                else:
                    perm = torch.randperm(mutable.n_candidates)
                    mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
                    result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
                mutable.mask = result[mutable.key].detach()
            elif isinstance(mutable, ValueSpace):
                gen_index = torch.tensor(self.search_space[mutable.key][self.idx])
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
        self.idx += 1
        return result

    def sample_final(self):
        return self.sample_search()


if __name__ == '__main__':
    from hyperbox.networks.nasbench201.nasbench201 import NASBench201Network
    net = NASBench201Network()
    fm = FairNASMutator(net)
    for i in range(24):
        if fm.idx == fm.lcm:
            print('new round\n=====================')
        fm.reset()
        net.sync_mask_for_all_cells(fm._cache)
        print(f"{i}: {net.arch_encoding} {fm.idx}")
