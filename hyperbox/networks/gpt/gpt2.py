from typing import Optional, Union
from functools import partial
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from hyperbox.mutables.ops import MultiheadAttention, Embedding, Linear, LayerNorm
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutables.spaces import ValueSpace


class CustomGELU(nn.Module):
    """GELU implementation taken from the `transformers`.
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
        the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        """Run forward pass."""
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        return x


class Block(nn.Module):
    """Decoder block.

        Parameters
        ----------
        n_embed : int
            Dimensionality of the embeddings.

        n_head : int
            Number of attention heads.

        n_positions : int
            Maximum number of tokens.

        attn_pdrop : float
            Probability of dropout on attention weights.

        resid_pdrop : float
            Probability of dropout after applying the MLP.

        layer_norm_epsilon : float
            Hyperparameter of layer normalization.

        Attributes
        ----------
        ln_1, ln_2 : LayerNorm
            Layer norms.

        attention : nn.MultiHeadAttention
            Attention module.

        mlp : nn.Sequential
            Multilayer perceptron.
    """

    def __init__(
        self,
        n_embed: Union[int, ValueSpace],
        n_head: Union[int, ValueSpace],
        n_positions: int,
        attn_pdrop: float,
        resid_pdrop: float,
        layer_norm_epsilon: float,
        block_index: int = None,
        transform_params_method: str = "Large2Small",
        mask: dict = None,
        *args, **kwargs,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.ln_2 = LayerNorm(n_embed, eps=layer_norm_epsilon)

        self.attention = MultiheadAttention(
            embed_dim=n_embed,
            num_heads=n_head,
            dropout=attn_pdrop,
            bias=True,
            batch_first=True,
            transform_params_method=transform_params_method
        )
        self.register_buffer(
            "mask",
            (1 - torch.tril(torch.ones(n_positions, n_positions))).to(
                dtype=torch.bool
            ),
        )

        if isinstance(n_embed, ValueSpace):
            key = n_embed.key
            candidates = (n_embed * 0.5 + n_embed * 4).candidates_original
            n_embed_middle = ValueSpace(candidates=candidates, key=f"{key}_block{block_index}", mask=mask)
        else:
            n_embed_middle = int(n_embed * 4)
        self.mlp = nn.Sequential(
            Linear(n_embed, n_embed_middle, bias=True),
            CustomGELU(),
            Linear(n_embed_middle, n_embed, bias=True),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        """Run forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape `(batch_size, n_tokens, n_embed)`.

            Returns
            -------
            torch.Tensor
                Output tensor of shape `(batch_size, n_tokens, n_embed)`.
        """
        batch_size, n_tokens, n_embed = x.shape

        x_ = self.ln_1(x)  # (batch_size, n_tokens, n_embed)

        mask = self.mask[:n_tokens, :n_tokens]  # (n_tokens, n_tokens)

        attn_out, _ = self.attention(
            x_, x_, x_, attn_mask=mask, need_weights=False
        )  # (batch_size, n_tokens, n_embed)
        x = x + attn_out  # (batch_size, n_tokens, n_embed)
        x = x + self.mlp(self.ln_2(x))  # (batch_size, n_tokens, n_embed)

        return x


class InputEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: Union[int, ValueSpace],
        n_positions: int,
        embd_pdrop: float,
    ):
        super(InputEmbedding, self).__init__()
        self.n_positions = n_positions
        self.token_emb = Embedding(vocab_size, n_embed)
        self.pos_emb = Embedding(n_positions, n_embed)

        self.drop = nn.Dropout(embd_pdrop)

    def forward(self, idx):
        batch_size, n_tokens = idx.shape
        device = idx.device

        if n_tokens > self.n_positions:
            raise ValueError("There are too many tokens in the input")

        positions = torch.arange(n_tokens, device=device)  # (n_tokens,)

        token_emb = self.token_emb(idx)  # (batch_size, n_tokens, n_embed)
        pos_emb = self.pos_emb(positions)[None, ...]  # (1, n_tokens, n_embed)
        x = self.drop(token_emb + pos_emb)  # (batch_size, n_tokens, n_embed)
        return x


class GPT2(BaseNASNetwork):
    """Entire GPT model.

        Parameters
        ----------
        vocab_size : int
            Number of tokens in the vocabulary.

        n_layer : int
            Number of decoder blocks to include.

        n_embed : int
            Dimensionality of the embeddings.

        n_head : int
            Number of attention heads.

        n_positions : int
            Maximum number of tokens.

        attn_pdrop : float
            Probability of dropout on attention weights.

        embd_pdrop : float
            Probability of dropout on the sum of embeddings.

        resid_pdrop : float
            Probability of dropout after applying the MLP.

        layer_norm_epsilon : float
            Hyperparameter of layer normalization.

        Attributes
        ----------
        token_emb : nn.Embedding
            Token embeddings.

        pos_emb : nn.Embedding
            Positional embedding.

        drop : nn.Dropout
            Dropout module to be applied on the sum of embeddings.

        blocks : nn.Sequential
            List of decoder blocks.

        ln : LayerNorm
            Layer norm applied before applying `head`.

        head : nn.Linear
            Final linear layer.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_embed: Union[int, list],
        n_head: Union[int, list],
        n_positions: int,
        attn_pdrop: float,
        embd_pdrop: float,
        resid_pdrop: float,
        layer_norm_epsilon: float,
        transform_params_method: str = 'Large2Small',
        mask: dict = None,
        **kwargs
    ):
        super().__init__()
        if isinstance(n_embed, list):
            _n_embed = ValueSpace(n_embed, key=f'n_embed', mask=self.mask)
        else:
            _n_embed = n_embed
        self.embedding = InputEmbedding(vocab_size, _n_embed, n_positions, embd_pdrop)

        self.blocks = []
        for layer_idx in range(n_layer):
            if isinstance(n_head, list):
                _n_head = ValueSpace(n_head, key=f'n_head_{layer_idx}', mask=self.mask)
            else:
                _n_head = n_head
            self.blocks.append(
                Block(
                    n_embed=_n_embed,
                    n_head=_n_head,
                    n_positions=n_positions,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    layer_norm_epsilon=layer_norm_epsilon,
                    block_index=layer_idx,
                    mask=self.mask
                )
            )
        self.blocks = nn.Sequential(*self.blocks)
        self.ln = LayerNorm(_n_embed, eps=layer_norm_epsilon)
        self.head = Linear(_n_embed, vocab_size, bias=False)

    def forward(self, idx):
        """Run forward pass.

            Parameters
            ----------
            idx : torch.Tensor
                Integer tensor of shape `(batch_size, n_tokens)` where each
                element is in the range `[0, vocab_size)`.

            Returns
            -------
            logits : torch.Tensor
                Tensor of shape `(batch_size, n_tokens, vocab_size)`.
        """
        x = self.embedding(idx)
        x = self.blocks(x)  # (batch_size, n_tokens, n_embed)
        x = self.ln(x)  # (batch_size, n_tokens, n_embed)
        logits = self.head(x)  # (batch_size, n_tokens, vocab_size)

        return logits


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator
    x = torch.randint(0, 100, (2, 12))
    # net = GPT(100, 12, 256, 8, 512, 0.1, 0.1, 0.1, 1e-5)
    # net = GPT(100, 12, 768, [8, 16], 512, 0.1, 0.1, 0.1, 1e-5)
    # net = GPT(100, 12, [256, 768], 8, 512, 0.1, 0.1, 0.1, 1e-5)
    net = GPT2(100, 12, [256, 768], [8, 16], 512, 0.1, 0.1, 0.1, 1e-5)
    rm = RandomMutator(net)
    for i in range(10):
        rm.reset()
        y = net(x)
        print(net.arch, y.shape)
