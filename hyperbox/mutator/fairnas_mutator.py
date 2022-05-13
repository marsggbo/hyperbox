import copy
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
    def __init__(
        self, model,
        is_singlepath: bool=True,
        parse_flag: str='max', # 'lcm' or 'max'
        *args, **kwargs
    ):
        '''
        Args:
            is_singlepath: whether to generate singlepath or multipath
            parse_flag: 'lcm' or 'max'
                lcm: least common multiple
            >>> original_FairNAS = FairNASMutator(model, is_singlepath=True, parse_flag='max')
        '''
        super(FairNASMutator, self).__init__(model)
        self.is_singlepath = is_singlepath
        self.parse_flag = parse_flag
        self.base_search_space = {
            mutable.key: list(range(mutable.length)) for mutable in self.mutables}
        self.search_space = self.generate_search_space(
            self.base_search_space, self.is_singlepath, self.parse_flag)

    def generate_search_space(
        self, search_space, is_singlepath, parse_flag
    ):
        self.idx = 0
        if is_singlepath:
            self.search_space = self.generate_singlepath_space(search_space, parse_flag)
        else:
            self.search_space = self.generate_multipath_space(search_space, parse_flag)
        self.shuffle_search_space(self.search_space)
        self.search_space = self.prune_repeat_path(self.search_space)
        return self.search_space

    def shuffle_search_space(self, search_space):
        for key, value in search_space.items():
            np.random.shuffle(value)

    def prune_repeat_path(self, search_space):
        pruned_search_space = {key: [] for key, value in search_space.items()}
        memory = ()
        idx = 0
        for idx in range(self.num_path):
            path = self.path2string(self.get_path_by_idx(idx))
            if path not in memory:
                memory += (path,)
                for key in search_space:
                    pruned_search_space[key].append(search_space[key][idx])
        self.num_path = len(memory)
        return pruned_search_space

    def generate_singlepath_space(self, search_space=None, parse_flag='lcm'):
        self.idx = 0
        if search_space is None:
            result = copy.deepcopy(self.base_search_space)
        else:
            result = copy.deepcopy(search_space)

        # generate fair singlepath
        if parse_flag == 'lcm':
            result = self.generate_lcm_search_space(result)
        elif parse_flag == 'max':
            result = self.generate_max_search_space(result)
        return result

    def generate_multipath_space(self, search_space=None, parse_flag='lcm'):
        '''generate multipath space

        multipath lcm mode:
            origin space        multipath space                      fair multipath space
        [
            [0,1,2,3],          [ [0], [1,3], [2] ],                 [ [0],   [1,3], [2],   [0],   [1,3], [2]   ],
            [0,1,2],     ==>    [ [0,1], [2] ],       ==> (lcm=6)    [ [0,1], [2],   [0,1], [2],   [0,1], [2]   ],
            [0,1],              [ [0,1] ],                           [ [0,1], [0,1], [0,1], [0,1], [0,1], [0,1] ],
            [0,1,2,3]           [ [1,3], [0], [2] ]                  [ [1,3], [0],   [2],   [1,3], [0],   [2]   ]
        ]
        multipath max mode:
            origin space        multipath space                      fair multipath space
        [
            [0,1,2,3],          [ [0], [1,3], [2] ],                 [ [0],   [1,3], [2]   ],
            [0,1,2],     ==>    [ [0,1], [2] ],       ==> (max=3)    [ [0,1], [2],   [0,1] ],
            [0,1],              [ [0,1] ],                           [ [0,1], [0,1], [0,1] ],
            [0,1,2,3]           [ [1,3], [0], [2] ]                  [ [1,3], [0],   [2]   ]
        ]
        '''
        self.idx = 0
        if search_space is None:
            search_space_cp = copy.deepcopy(self.base_search_space)
        else:
            search_space_cp = copy.deepcopy(search_space)
        result = dict()
        # generate multipath
        for key, space in search_space_cp.items():
            tmp_space = []
            length = len(space)
            np.random.shuffle(space)
            while length > 1:
                sample_num = np.random.randint(1, length)
                tmp_space.append([space.pop() for i in range(sample_num)])
                length -= sample_num
            if length == 1:
                tmp_space.append([space.pop()])
            result[key] = tmp_space

        # generate fair multipath
        if parse_flag == 'lcm':
            result = self.generate_lcm_search_space(result)
        elif parse_flag == 'max':
            result = self.generate_max_search_space(result)
        # print(f"generate multipath space: {result}\num_path: {self.num_path}")
        return result

    def generate_lcm_search_space(self, search_space):
        self.num_path = lcm([len(space) for key, space in search_space.items()])
        for key, space in search_space.items():
            search_space[key] = space * (self.num_path // len(space))
        return search_space

    def generate_max_search_space(self, search_space):
        self.num_path = max([len(space) for key, space in search_space.items()])
        for key, space in search_space.items():
            if len(space) < self.num_path:
                num_to_sample = self.num_path - len(space)
                replace = False if len(space) >= num_to_sample else True
                search_space[key] = space + np.random.choice(space, num_to_sample, replace=replace).tolist()
        return search_space

    def sample_search(self):
        if self.idx >= self.num_path:
            self.search_space = self.generate_search_space(self.base_search_space, self.is_singlepath, self.parse_flag)
        result = dict()
        for mutable in self.mutables:
            if isinstance(mutable, (OperationSpace, ValueSpace)):
                gen_index = torch.tensor(self.search_space[mutable.key][self.idx])
                if self.is_singlepath:
                    result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                else:
                    result[mutable.key] = torch.zeros(mutable.length, dtype=torch.long).scatter_(0, gen_index, 1).bool()
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
        self.idx += 1
        return result

    def sample_final(self):
        return self.sample_search()

    def get_path_by_idx(self, idx):
        path = {key: value[idx] for key, value in self.search_space.items()}
        return path

    def path2string(self, path):
        name = ''
        for key, value in path.items():
            # name += f"{{{key}:{value}}}"
            name += f"{value}_"
        return name


if __name__ == '__main__':
    from hyperbox.networks.nasbench201.nasbench201 import NASBench201Network

    def test(net, fm):
        print('\n========test========')
        for i in range(60):
            if fm.idx == fm.num_path:
                print('new round\n=====================')
            fm.reset()
            print(f"{i+1}: {net.arch_encoding} {fm.idx}")

    # singlepath lcm
    net = NASBench201Network()
    fm_sl = FairNASMutator(net, is_singlepath=True, parse_flag='lcm')
    test(net, fm_sl)

    # singlepath max
    net = NASBench201Network()
    fm_sm = FairNASMutator(net, is_singlepath=True, parse_flag='max')
    test(net, fm_sm)

    # multipath lcm
    net = NASBench201Network()
    fm_ml = FairNASMutator(net, is_singlepath=False, parse_flag='lcm')
    test(net, fm_ml)

    # multipath max
    net = NASBench201Network()
    fm_mm = FairNASMutator(net, is_singlepath=False, parse_flag='max')
    test(net, fm_mm)
