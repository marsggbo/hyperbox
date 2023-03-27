from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hyperbox.mutables.ops.base_module import FinegrainedModule
from hyperbox.mutables.ops.utils import is_searchable
from hyperbox.mutables.spaces import ValueSpace


class LayerNorm(nn.LayerNorm, FinegrainedModule):
    def __init__(
        self,
        normalized_shape: Union[int, ValueSpace, List[int]], # Todo: support to search Tensor.Size
        eps: float = 1e-5, 
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        *args, **kwargs,
    ) -> None:
        r"""
        
        Examples::
            >>> # NLP Example
            >>> batch, sentence_length, embedding_dim = 20, 5, 10
            >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
            >>> layer_norm = nn.LayerNorm(embedding_dim)
            >>> # Activate module
            >>> layer_norm(embedding)
            >>>
            >>> # NLP Searchable Example
            >>> from hyperbox.mutables.spaces import ValueSpace
            >>> from hyperbox.mutables.ops import LayerNorm, Embedding
            >>> batch, sentence_length, vocab_size = 20, 5, 10000
            >>> embedding_dim = ValueSpace([10, 20])
            >>> x = torch.randint(0, vocab_size, (batch, sentence_length))
            >>> input_embeddimg_layer = nn.Sequential(
                Embedding(1000, embedding_dim),
                Linear(embedding_dim))
            >>> x = input_embeddimg_layer(x)
            >>>
            >>> # Image Example
            >>> N, C, H, W = 20, 5, 10, 30
            >>> input = torch.randn(N, C, H, W)
            >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
            >>> # as shown in the image below
            >>> layer_norm = nn.LayerNorm([C, H, W])
            >>> output = layer_norm(input)
            >>> layer_norm = nn.LayerNorm([H, W])
            >>> output = layer_norm(input)
            >>> layer_norm = nn.LayerNorm([W])
            >>> output = layer_norm(input)
        """
        if isinstance(normalized_shape, ValueSpace):
            _normalized_shape = normalized_shape.max_value
        else:
            _normalized_shape = normalized_shape
        super(LayerNorm, self).__init__(
            _normalized_shape, eps, elementwise_affine, *args, **kwargs)
        self.is_search = self.isSearchLayerNorm()

    def forward(self, input: Tensor) -> Tensor:
        if not self.is_search:
            x = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            normalized_shape = self.value_spaces["normalized_shape"].value
            weight = self.weight[:normalized_shape]
            if self.bias is not None:
                bias = self.bias[:normalized_shape]
            x = F.layer_norm(
                input, (normalized_shape,), weight, bias, self.eps)
        return x

    def isSearchLayerNorm(self):
        '''search flag
            search
            search_normalized_shape
        '''
        self.search_normalized_shape = False
        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False
        if is_searchable(getattr(self.value_spaces, 'normalized_shape', None)):
            self.search_normalized_shape = True
        return True


if __name__ == "__main__":
    # NLP Searchable Example
    from hyperbox.mutables.ops import Embedding
    from hyperbox.mutator import RandomMutator
    
    batch, sentence_length, vocab_size = 20, 5, 10000
    for i in range(10):
        if i % 2 == 0:
            embedding_dim = ValueSpace([10, 20])
        elif i % 2 == 1:
            embedding_dim = 10 + i
        x = torch.randint(0, vocab_size, (batch, sentence_length))
        input_embeddimg_layer = nn.Sequential(
                Embedding(1000, embedding_dim),
                LayerNorm(embedding_dim)
        )
        rm = RandomMutator(input_embeddimg_layer)
        for j in range(4):
            rm.reset()
            x = torch.randint(0, 10, (2, 30))
            y = input_embeddimg_layer(x)
            # print(y.shape)
