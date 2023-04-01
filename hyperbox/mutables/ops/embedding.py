from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hyperbox.mutables.ops.base_module import FinegrainedModule
from hyperbox.mutables.ops.utils import is_searchable
from hyperbox.mutables.spaces import ValueSpace


class Embedding(nn.Embedding, FinegrainedModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: Union[int, ValueSpace],
        device=None,
        dtype=None,
        *args, **kwargs,
    ) -> None:
        if isinstance(embedding_dim, ValueSpace):
            _embedding_dim = embedding_dim.max_value
        else:
            _embedding_dim = embedding_dim
        super(Embedding, self).__init__(
            num_embeddings, _embedding_dim, *args, **kwargs)
        self.is_search = self.isSearchEmbedding()

    def forward(self, input: Tensor) -> Tensor:
        if not self.is_search:
            x = F.embedding(input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            embedding_dim = self.value_spaces["embedding_dim"].value
            weight = self.weight[:, :embedding_dim]
            x = F.embedding(input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        return x

    def isSearchEmbedding(self):
        '''search flag
            search
            search_embedding_dim
        '''
        self.search_embedding_dim = False
        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False
        if is_searchable(getattr(self.value_spaces, 'embedding_dim', None)):
            self.search_embedding_dim = True
        return True

    @property
    def params(self):
        weight = self.weight
        if self.search_embedding_dim:
            weight = weight[:, :self.value_spaces['embedding_dim'].value]
        size = weight.numel()
        return size


if __name__ == "__main__":
    from hyperbox.mutator import RandomMutator
    for i in range(10):
        embed_dim = ValueSpace([10, 20])
        embedding = Embedding(
            10, embed_dim, max_norm=1.0, norm_type=2.0,
            scale_grad_by_freq=True, sparse=True)    
        rm = RandomMutator(embedding)
        print(embedding)
        for j in range(4):
            rm.reset()
            x = torch.randint(0, 10, (2, 30))
            y = embedding(x)
            print(y.shape)
