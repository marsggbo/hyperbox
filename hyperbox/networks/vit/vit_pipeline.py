from typing import Dict, List, Tuple, Union, Optional, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.mutables import spaces, ops


__all__ = [
    'Attention',
    'Transformer',
    'VisionTransformer',
    'ViT',
    'ViT_S',
    'ViT_B',
    'ViT_H',
    'ViT_G',
    'ViT_10B',
]


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def keepPositiveList(l):
    return [x for x in l if x > 0]


# model classes
class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int, # input dimension
        hidden_dim: int, # hidden dimension
        search_ratio: List[float] = [0.5, 1.], # search ratio for hidden dimension (hidden_dim)
        dropout: float = 0., # dropout rate
        suffix: str = '', # suffix for the name of the module
        mask= None, # mask for the search space (mutables)
    ):
        super().__init__()
        self.mask = mask
        hidden_dim_list = keepPositiveList([int(r*hidden_dim) for r in search_ratio])
        hidden_dim_list = spaces.ValueSpace(hidden_dim_list, key=f"{suffix}_hidden_dim", mask=self.mask)
        self.net = nn.Sequential(
            ops.Linear(dim, hidden_dim_list),
            nn.GELU(),
            nn.Dropout(dropout),
            ops.Linear(hidden_dim_list, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int, # input dimension
        search_ratio: List[float], # search ratio for hidden dimension
        heads: int = 8, # number of attention heads
        dim_head: int = 64, # dimension of each attention head
        dropout: float = 0., # dropout
        suffix: str = None, # suffix for naming
        mask: dict = None, # mask for the search space (mutables)
    ):
        super().__init__()
        self.mask = mask
        self.heads_list = keepPositiveList([int(r*heads) for r in search_ratio])
        self.dim_head_list = keepPositiveList([int(r*dim_head) for r in search_ratio])
        self.scale_list = [dh**-0.5 for dh in self.dim_head_list]
        
        self.inner_dim_list = []
        self.heads_idx_map = {}
        self.dim_head_idx_map = {}
        count = 0
        for h_idx, h in enumerate(self.heads_list):
            for d_idx, d in enumerate(self.dim_head_list):
                inner_dim = h * d
                self.inner_dim_list.append(inner_dim)
                self.heads_idx_map[count] = h_idx
                self.dim_head_idx_map[count] = d_idx
                count += 1
        self.inner_dim_list = spaces.ValueSpace(self.inner_dim_list, key=f"{suffix}_inner_dim", mask=self.mask)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        qkv_dim_list = [x*3 for x in self.inner_dim_list.candidates_original]
        qkv_dim_list = spaces.ValueSpace(qkv_dim_list, key=f"{suffix}_inner_dim", mask=self.mask) # coupled with self.inner_dim_list
        self.to_qkv = ops.Linear(dim, qkv_dim_list, bias = False)

        self.to_out = nn.Sequential(
            ops.Linear(self.inner_dim_list, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    @property
    def heads(self):
        idx = self.inner_dim_list.index
        if idx is None:
            idx = self.inner_dim_list.mask.float().argmax().item()
        idx = self.heads_idx_map[idx]
        return self.heads_list[idx]
    
    @property
    def scale(self):
        idx = self.inner_dim_list.index
        if idx is None:
            idx = self.inner_dim_list.mask.float().argmax().item()
        idx = self.dim_head_idx_map[idx]
        return self.scale_list[idx]

    @property
    def dim_head(self):
        idx = self.inner_dim_list.index
        if idx is None:
            idx = self.inner_dim_list.mask.float().argmax().item()
        idx = self.dim_head_idx_map[idx]
        return self.dim_head_list[idx]


class VitEmbedding(nn.Module):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]], # image size
        patch_size: Union[int, Tuple[int, int]], # patch size
        channels: int, # number of input channels, 3 for RGB images
        dim: int, # dimension of the model
        emb_dropout: float = 0.1, # embedding dropout rate
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x


class VitBlock(nn.Module):
    def __init__(
        self,
        dim: int, # dimension of the model
        heads: int, # number of heads
        dim_head: int, # dimension of each head
        mlp_dim: int, # dimension of the feedforward layer
        search_ratio: list, # search ratio for hidden_dim and inner_dim
        dropout: float= 0., # dropout rate
        suffix: str = None, # suffix for the block, r.g., 1, 2, 3, etc.
        mask: dict = None, # mask for the search space (mutables)
    ):
        super().__init__()
        self.search_ratio = search_ratio
        self.mask = mask
        self.suffix = suffix
        attKey = f"att_{suffix}"
        ffKey = f"ff_{suffix}"
        self.attn = PreNorm(dim, Attention(
            dim, heads = heads, dim_head = dim_head, search_ratio=self.search_ratio,
            dropout = dropout, suffix = attKey, mask = self.mask))
        self.ff = PreNorm(dim, FeedForward(
            dim, mlp_dim, search_ratio=self.search_ratio, dropout = dropout,
            suffix = ffKey, mask = self.mask))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class VitClsHead(nn.Module):
    def __init__(
        self,
        pool: str, # 'cls' or 'mean'
        dim: int, # dimension of the model
        num_classes: int, # number of classes
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
        

class VisionTransformer(BaseNASNetwork):
    def __init__(
        self, *, 
        image_size: Union[int, Tuple[int, int]], # image size
        patch_size: Union[int, Tuple[int, int]], # patch size
        num_classes: int, # number of classes
        dim: int, # dim of model (hidden dim), each layer has the same dim
        depth: int, # depth of transformer
        heads: int, # number of heads
        mlp_dim: int, # hidden dim of mlp
        search_ratio: Optional[List[float]] = None,
        pool: str = 'cls', # 'cls' or 'mean'
        channels: int = 3, # number of input channels
        dim_head: int = 64, # dimension of each attention head
        dropout: float = 0., # dropout rate
        emb_dropout: float = 0., # embedding dropout rate
        to_search_depth: bool = False,
        mask: dict = None, # mask for the search space (mutables)
    ):
        super().__init__(mask = mask)
        self.to_search_path = to_search_depth
        
        self.vit_embed = VitEmbedding(image_size, patch_size, channels, dim, emb_dropout)

        if self.to_search_path:
            runtime_depth = [v for v in range(1, depth + 1)]    
            self.run_depth = spaces.ValueSpace(runtime_depth, key='run_depth', mask=self.mask)

        vit_blocks = [
            VitBlock(
                dim=dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                search_ratio=search_ratio, dropout=dropout, suffix=i, mask=self.mask
            ) for i in range(depth)]
        self.vit_blocks = nn.Sequential(*vit_blocks)

        self.vit_cls_head = VitClsHead(pool, dim, num_classes)

    def forward(self, x):
        out = self.vit_embed(x)
        if self.to_search_path:
            vit_blocks = list(self.vit_blocks.children())
            runtime_depth = self.run_depth.value
            vit_blocks = vit_blocks[:runtime_depth]
            vit_blocks = nn.Sequential(*vit_blocks)
            out = vit_blocks(out)
        else:
            out = self.vit_blocks(out)
        out = self.vit_cls_head(out)
        return out


_vit_s = dict(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=4,
    heads=6,
    dim_head=64, # intermediate_size=12*64=768
    mlp_dim=1024,
    search_ratio=[0.5, 0.75, 1],
    pool='cls',
    channels=3,
    dropout=0.1,
    emb_dropout=0.1,
)

_vit_b = dict(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=768,
    depth=12,
    heads=12,
    dim_head=64, # intermediate_size=12*64=768
    mlp_dim=3072,
    search_ratio=[0.5, 0.75, 1],
    pool='cls',
    channels=3,
    dropout=0.1,
    emb_dropout=0.1,
)

_vit_h = dict(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=1280,
    depth=32,
    heads=16,
    dim_head=80,
    mlp_dim=5120,
    search_ratio=[0.5, 0.75, 1],
    pool='cls',
    channels=3,
    dropout=0.1,
    emb_dropout=0.1,
)

_vit_g = dict(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=1664,
    depth=48,
    heads=16,
    dim_head=104,
    mlp_dim=8192,
    search_ratio=[0.5, 0.75, 1],
    pool='cls',
    channels=3,
    dropout=0.1,
    emb_dropout=0.1,
)

_vit_10b = dict(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=4096,
    depth=50,
    heads=16,
    dim_head=256,
    mlp_dim=16384,
    search_ratio=[0.5, 0.75, 1],
    pool='cls',
    channels=3,
    dropout=0.1,
    emb_dropout=0.1,
)

ViT = partial(VisionTransformer, **_vit_b)
ViT_S = partial(VisionTransformer, **_vit_s)
ViT_B = partial(VisionTransformer, **_vit_b)
ViT_H = partial(VisionTransformer, **_vit_h)
ViT_G = partial(VisionTransformer, **_vit_g)
ViT_10B = partial(VisionTransformer, **_vit_10b)


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator
    # device = 'cpu'
    device = 'mps'
    # device = 'cuda'
    net = ViT(image_size = 224, patch_size = 16, num_classes = 1000, dim = 1024, 
        depth = 6, heads = 1, dim_head=1024, mlp_dim = 2048)
    x = torch.rand(2,3,224,224).to(device)
    net = net.to(device)
    rm = RandomMutator(net)
    steps = 2
    net.eval()
    for i in range(steps):
        rm.reset()
        if i==(steps-1): 
            print(rm._cache)
        # print(net.arch_size((2,3,224,224), convert=False, verbose=True))
        y = net(x)
        # print(y.shape, y.device)
        subnet = net.build_subnet(mask=rm._cache).to(device)
        subnet.eval()
        y2 = subnet(x)
        print((y-y2).abs().sum().item())