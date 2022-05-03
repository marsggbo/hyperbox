import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.mutator.default_mutator import Mutator
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import lazy_property


__all__ = [
    'FewshotMutator',
]

log = get_logger(__name__)


class FewshotMutator(Mutator):
    def __init__(
        self,
        model,
        training_epochs: int = 100,
        split_mutable_indices: Union[int, list]=2,
        save_dir: Optional[str] = None,
        to_save_sub_supernets: bool = True,
        *args, **kwargs
    ):
        super().__init__(model)
        self.training_epochs = training_epochs
        self.to_save_sub_supernets = to_save_sub_supernets
        if save_dir is None:
            self.save_dir = os.path.join(os.getcwd(), 'fewshot_sub_supernets')
        else:
            self.save_dir = save_dir
        to_split = self.parse_split_mutable_indices(split_mutable_indices)
        self.split_mutable_indices, self.split_mutable_keys = to_split
        self.sub_supernet_pools, self.num_sub_supernets = self.split_search_space(self.split_mutable_keys)
        assert training_epochs % self.num_sub_supernets ==0, f"training_epochs ({training_epochs}) must be divisible by num_sub_supernets ({self.num_sub_supernets})."
        self.training_interval = training_epochs // self.num_sub_supernets
        self.crt_sub_sueprnet_idx = 0
        self.crt_sub_supernet = self.sub_supernet_pools[self.crt_sub_sueprnet_idx]
        self.crt_mutable_key = self.crt_sub_supernet['key']
        log.info(f"there are {self.num_sub_supernets} sub-supernets, each has {self.training_interval} training epochs")

    def parse_split_mutable_indices(self, split_mutable_indices):
        if isinstance(split_mutable_indices, int):
            assert split_mutable_indices <= len(self), 'split_mutable_indices must be less than or equal to the number of mutables'
            split_mutable_indices = np.random.choice(len(self), size=split_mutable_indices, replace=False).tolist()
        assert len(split_mutable_indices) <= len(self), 'split_mutable_indices must be less than or equal to the number of mutables'
        split_mutable_keys = [self[i].key for i in split_mutable_indices]
        return split_mutable_indices, split_mutable_keys

    def split_search_space(self, split_keys):
        num_sub_supernets = 0
        sub_supernet_pools = []
        for idx, key in enumerate(split_keys):
            length = self[key].length
            for idy in range(length):
                mask = F.one_hot(torch.tensor(idy), num_classes=length).view(-1).bool()
                sub_supernet = {'key': key, 'mask': mask}
                sub_supernet_pools.append(sub_supernet)
                num_sub_supernets += 1
        return sub_supernet_pools, num_sub_supernets

    def sample_search(self):
        if self.training_interval == 0:
            # switch to another sub-supernet
            self.training_interval = self.training_epochs // self.num_sub_supernets
            self.crt_sub_sueprnet_idx += 1
            self.crt_sub_supernet = self.sub_supernet_pools[self.crt_sub_sueprnet_idx]
            self.crt_mutable_key = self.crt_sub_supernet['key']
            if self.to_save_sub_supernets:
                self.save_sub_supernet()
        result = dict()
        for mutable in self.mutables:
            if mutable.key == self.crt_mutable_key:
                mask = self.crt_sub_supernet['mask']
                result[mutable.key] = mask
                mutable.mask = mask
                continue
            elif isinstance(mutable, OperationSpace):
                gen_index = torch.randint(high=mutable.length, size=(1, ))
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = result[mutable.key].detach()
            elif isinstance(mutable, InputSpace):
                if mutable.n_chosen is None:
                    result[mutable.key] = torch.randint(high=2, size=(mutable.n_candidates,)).view(-1).bool()
                else:
                    perm = torch.randperm(mutable.n_candidates)
                    mask = [i in perm[:mutable.n_chosen] for i in range(mutable.n_candidates)]
                    result[mutable.key] = torch.tensor(mask, dtype=torch.bool)  # pylint: disable=not-callable
                mutable.mask = result[mutable.key].detach()
            elif isinstance(mutable, ValueSpace):
                gen_index = torch.randint(high=mutable.length, size=(1, ))
                result[mutable.key] = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
                mutable.mask = F.one_hot(gen_index, num_classes=mutable.length).view(-1).bool()
        # self.training_interval -= 1 # should be called outside after each epoch
        return result

    def sample_final(self):
        return self.sample_search()

    @property
    def sub_supernet_name(self):
        # the key of fixed mutable
        key = self.crt_mutable_key
        # the mask of fixed mutable
        mask = self.crt_sub_supernet['mask'].cpu().detach().int().numpy().tolist()
        mask = ''.join(str(i) for i in mask)
        name = f"sub_supernet(fixed={key},mask={mask})"
        return name

    def save_sub_supernet(self):
        name = self.sub_supernet_name
        path = os.path.join(self.save_dir, name)
        log.info(f"saving sub-supernet to {path}")
        # torch.save(self.model.state_dict(), f'{name}.pth')


if __name__ == '__main__':
    from hyperbox.networks.nasbench201 import NASBench201Network
    net = NASBench201Network()
    epochs = 60
    fm = FewshotMutator(net, training_epochs=epochs, split_mutable_indices=[3,1,0,2])
    for epoch in range(epochs):
        fm.reset()
        print(fm.sub_supernet_name, net.arch_encoding)
        fm.training_interval -= 1
