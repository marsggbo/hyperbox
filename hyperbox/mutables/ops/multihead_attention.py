from typing import Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from hyperbox.mutables.ops.base_module import FinegrainedModule
from hyperbox.mutables.ops.utils import is_searchable
from hyperbox.mutables.spaces import ValueSpace


class MultiheadAttention(FinegrainedModule):
    def __init__(
        self,
        embed_dim: Union[int, ValueSpace], # Total dimension of the model.
        num_heads: Union[int, ValueSpace], # Parallel attention heads. Note that ``embed_dim`` will be split
                   # across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: float=0., # Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: bool=True,  # If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: bool=False,   # If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: bool=False, # If specified, adds a new batch of zeros to the key and value sequences at dim=1.
        batch_first: bool=False, # If ``True``, then the input and output tensors are provided as (batch, seq, feature).
                           # Default: ``False`` (seq, batch, feature).
        transform_params_method: str='disable', # Method to transform parameters (disable, L2S, identity). Default: ``disable``.
                            # suppose orginal weights has shape [512, 256], and we want to truncate it to [256, 128]
                            # 1. `disable`: no transformation, directly using truncated parameters, e.g., new_weight = weight[:256, :128]
                            # 2. `Large2Small`: multiply two matrices left [256, 512] and right [256, 128], e.g., new_weight = left @ weight @ right
                            # 3. `TruncatedLinear`: first truncated and multiply a matrix A [128, 128] and plus a vector B [128]
                            #    , i.e., new_weight = weight[:256, :128] @ A + B
        device=None, dtype=None,
    ) -> None:
        r"""Allows the model to jointly attend to information from different representation subspaces.
        
            Examples::
                >>> multihead_attn = MultiheadAttention(embed_dim, num_heads)
                >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
        """
        super(MultiheadAttention, self).__init__()
        mha_kwargs = {
            key: getattr(self, key, None) for key in [
                'embed_dim', 'dropout', 'bias', 'add_bias_kv',
                'add_zero_attn', 'device', 'dtype']
        }
        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        self.is_search = self.isSearchMHA()
        self.init_weights(**mha_kwargs)

    def init_weights(self, embed_dim, dropout, bias, add_bias_kv, add_zero_attn, device, dtype):
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        if getattr(self.value_spaces, 'embed_dim', False) and self.transform_params_method != 'disable':
            self.init_transform_matrix()

        self._reset_parameters()

    def init_transform_matrix(self):
        # Todo: init transform matrix (TM)
        
        max_embed_dim = self.embed_dim
        embed_dim_candidates = sorted(self.value_spaces['embed_dim'].candidates_original)
        
        matrices = {}
        for i in range(len(embed_dim_candidates) - 1):
            dim_small = embed_dim_candidates[i]

            if self.transform_params_method == 'PoolConv':
                assert max_embed_dim % dim_small == 0, \
                    f'embed_dim ({max_embed_dim}) must be divisible by candidate dim ({dim_small})'
                downsample_ratio = max_embed_dim // dim_small
                downsample_times = int(math.log2(downsample_ratio))
                
                # in_proj_weight
                param_name = f'in_proj_weight_TM_{max_embed_dim}to{dim_small}'
                matrices[param_name] = nn.Sequential(
                    *[nn.Sequential(
                        # nn.AvgPool2d(2),
                        nn.Conv2d(1, 1, 3, 2, 1, bias=False)
                    ) for i in range(downsample_times)],
                )

                # out_proj_weight
                param_name = f'out_proj_weight_TM_{max_embed_dim}to{dim_small}'
                matrices[param_name] = nn.Sequential(
                    *[nn.Sequential(
                        # nn.AvgPool2d(2),
                        nn.Conv2d(1, 1, 3, 2, 1, bias=False)
                    ) for i in range(downsample_times)],
                )

                # in_proj_bias
                if self.in_proj_bias is not None:
                    param_name = f'in_proj_bias_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = nn.Sequential(
                        *[nn.Sequential(
                            # nn.AvgPool1d(2),
                            nn.Conv1d(1, 1, 3, 2, 1, bias=False)
                        ) for i in range(downsample_times)]
                    )
                
                # out_proj_bias
                if self.out_proj.bias is not None:
                    param_name = f'out_proj_bias_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = nn.Sequential(
                        *[nn.Sequential(
                            # nn.AvgPool1d(2),
                            nn.Conv1d(1, 1, 3, 2, 1, bias=False)
                        ) for i in range(downsample_times)]
                    )

                # bias_k
                if self.bias_k is not None:
                    param_name = f'bias_k_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = nn.Sequential(
                        *[nn.Sequential(
                            # nn.AvgPool1d(2),
                            nn.Conv1d(1, 1, 3, 2, 1, bias=False)
                        ) for i in range(downsample_times)]
                    )

                # bias_v
                if self.bias_v is not None:
                    param_name = f'bias_v_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = nn.Sequential(
                        *[nn.Sequential(
                            # nn.AvgPool1d(2),
                            nn.Conv1d(1, 1, 3, 2, 1, bias=False)
                        ) for i in range(downsample_times)]
                    )
            elif self.transform_params_method == 'Large2Small':
                # in_proj_weight
                left_param_name = f'in_proj_weight_left_TM_{max_embed_dim}to{dim_small}'
                right_param_name = f'in_proj_weight_right_TM_{max_embed_dim}to{dim_small}'
                matrices[left_param_name] = Parameter(torch.eye(dim_small, 3*max_embed_dim))
                matrices[right_param_name] = Parameter(torch.eye(max_embed_dim, 3*dim_small))

                # out_proj_weight
                left_param_name = f'out_proj_weight_left_TM_{max_embed_dim}to{dim_small}'
                right_param_name = f'out_proj_weight_right_TM_{max_embed_dim}to{dim_small}'
                matrices[left_param_name] = Parameter(torch.eye(dim_small, max_embed_dim))
                matrices[right_param_name] = Parameter(torch.eye(max_embed_dim, dim_small))

                # in_proj_bias
                if self.in_proj_bias is not None:
                    param_name = f'in_proj_bias_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(3*max_embed_dim, 3*dim_small))

                # out_proj_bias
                if self.out_proj.bias is not None:
                    param_name = f'out_proj_bias_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(max_embed_dim, dim_small))

                # bias_k
                if self.bias_k is not None:
                    param_name = f'bias_k_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(max_embed_dim, dim_small))

                # bias_v
                if self.bias_v is not None:
                    param_name = f'bias_v_TM_{max_embed_dim}to{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(max_embed_dim, dim_small))
            elif self.transform_params_method == 'TruncatedLinear':
                # in_proj_weight
                param_name = f'in_proj_weight_TM_{dim_small}'
                matrices[param_name] = Parameter(torch.eye(dim_small, dim_small))

                # out_proj_weight
                param_name = f'out_proj_weight_TM_{dim_small}'
                matrices[param_name] = Parameter(torch.eye(dim_small, dim_small))

                # in_proj_bias
                if self.in_proj_bias is not None:
                    param_name = f'in_proj_bias_TM_{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(3*dim_small, 3*dim_small))

                # out_proj_bias
                if self.out_proj.bias is not None:
                    param_name = f'out_proj_bias_TM_{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(dim_small, dim_small))

                # bias_k
                if self.bias_k is not None:
                    param_name = f'bias_k_TM_{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(dim_small, dim_small))

                # bias_v
                if self.bias_v is not None:
                    param_name = f'bias_v_TM_{dim_small}'
                    matrices[param_name] = Parameter(torch.eye(dim_small, dim_small))
            else:
                raise NotImplementedError
                
        for name, val in matrices.items():
            if isinstance(val, Parameter):
                self.register_parameter(name, val)
            elif isinstance(val, nn.Module):
                self.add_module(name, val)

    def forward(
        self,
        query: Tensor, # (BatchSize, TargetSeqLength, EmbedDim) if batch_first is True else (TargetSeqLength, BatchSize, EmbedDim)
        key: Tensor,   # (BatchSize, SourceSeqLength, EmbedDim) if batch_first is True else (SourceSeqLength, BatchSize, EmbedDim) 
        value: Tensor, # (BatchSize, SourceSeqLength, EmbedDim) if batch_first is True else (SourceSeqLength, BatchSize, EmbedDim) 
        key_padding_mask: Optional[Tensor] = None, # require a mask of shape (BatchSize, SourceSeqLength). 
                                                   # For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored.
                                                   # For a byte mask, a non-zero value indicates that the corresponding ``key`` value will be ignored.
        need_weights: bool = True, # If specified, returns ``attn_output_weights`` and ``attn_outputs``. Default: ``True``.
        attn_mask: Optional[Tensor] = None, # Must be shape (TargetSeqLength, SourceSeqLength) or (BatchSize*NumHeads, TargetSeqLength, SourceSeqLength).
                                            # For a binary mask, a ``True`` value indicates that the corresponding position is not allowed to attend.
                                            # For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend.
                                            # For a float mask, the mask values will be added to the attention weight.
    ):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self.is_search:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        else:            
            if self.search_num_heads:
                num_heads = self.value_spaces['num_heads'].value
            else:
                num_heads = self.num_heads
            if self.search_embed_dim:
                embed_dim = self.value_spaces['embed_dim'].value
            else:
                embed_dim = self.embed_dim
            in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias = self.transform_params(embed_dim)

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, embed_dim, num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def transform_params(self, embed_dim):
        if embed_dim == self.embed_dim:
            return self.in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v, self.out_proj.weight, self.out_proj.bias
        if self.transform_params_method == 'PoolConv':
            in_proj_weight = getattr(
                self, f'in_proj_weight_TM_{self.embed_dim}to{embed_dim}'
                )(self.in_proj_weight.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            out_proj_weight = getattr(
                self, f'out_proj_weight_TM_{self.embed_dim}to{embed_dim}'
                )(self.out_proj.weight.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            in_proj_bias = getattr(
                self, f'in_proj_bias_TM_{self.embed_dim}to{embed_dim}'
                )(self.in_proj_bias.unsqueeze(0)).squeeze(0) if self.in_proj_bias is not None else None
            out_proj_bias = getattr(
                self, f'out_proj_bias_TM_{self.embed_dim}to{embed_dim}'
                )(self.out_proj.bias.unsqueeze(0)).squeeze(0) if self.out_proj.bias is not None else None
            bias_k = getattr(
                self, f'bias_k_TM_{self.embed_dim}to{embed_dim}'
                )(self.bias_k.unsqueeze(0)).squeeze(0) if self.bias_k is not None else None
            bias_v = getattr(
                self, f'bias_v_TM_{self.embed_dim}to{embed_dim}'
                )(self.bias_v.unsqueeze(0)).squeeze(0) if self.bias_v is not None else None
        elif self.transform_params_method == 'Large2Small':
            # weights
            in_proj_weight = self.param_transform_via_TM(
                self.in_proj_weight,
                L_transform_matrix=getattr(self, f'in_proj_weight_left_TM_{self.embed_dim}to{embed_dim}'),
                R_transform_matrix=getattr(self, f'in_proj_weight_right_TM_{self.embed_dim}to{embed_dim}')).T
            out_proj_weight = self.param_transform_via_TM(
                self.out_proj.weight,
                L_transform_matrix=getattr(self, f'out_proj_weight_left_TM_{self.embed_dim}to{embed_dim}'),
                R_transform_matrix=getattr(self, f'out_proj_weight_right_TM_{self.embed_dim}to{embed_dim}'))

            # biases
            in_proj_bias = self.param_transform_via_TM(
                self.in_proj_bias, R_transform_matrix=getattr(self, f'in_proj_bias_TM_{self.embed_dim}to{embed_dim}')
            ) if self.in_proj_bias is not None else None
            out_proj_bias = self.param_transform_via_TM(
                self.out_proj.bias, R_transform_matrix=getattr(self, f'out_proj_bias_TM_{self.embed_dim}to{embed_dim}')
            ) if self.out_proj.bias is not None else None
            bias_k = self.param_transform_via_TM(
                self.bias_k, R_transform_matrix=getattr(self, f'bias_k_TM_{self.embed_dim}to{embed_dim}')
            ) if self.bias_k is not None else None
            bias_v = self.param_transform_via_TM(
                self.bias_v, R_transform_matrix=getattr(self, f'bias_v_TM_{self.embed_dim}to{embed_dim}')
            ) if self.bias_v is not None else None
        else:
            # weights
            in_proj_weight = self.in_proj_weight[:3 * embed_dim, :embed_dim]
            out_proj_weight = self.out_proj.weight[:embed_dim, :embed_dim]

            # biases
            in_proj_bias = self.in_proj_bias[:3 * embed_dim] if self.in_proj_bias is not None else None
            out_proj_bias = self.out_proj.bias[:embed_dim] if self.out_proj.bias is not None else None
            bias_k = self.bias_k[:, :, :embed_dim] if self.bias_k is not None else None
            bias_v = self.bias_v[:, :, :embed_dim] if self.bias_v is not None else None
            if self.transform_params_method == 'TruncatedLinear':
                in_proj_weight = self.param_transform_via_TM(
                    in_proj_weight, R_transform_matrix=getattr(self, f'in_proj_weight_TM_{embed_dim}'))
                out_proj_weight = self.param_transform_via_TM(
                    out_proj_weight, R_transform_matrix=getattr(self, f'out_proj_weight_TM_{embed_dim}'))
                if in_proj_bias is not None:
                    in_proj_bias = self.param_transform_via_TM(
                        in_proj_bias, R_transform_matrix=getattr(self, f'in_proj_bias_TM_{embed_dim}'))
                if out_proj_bias is not None:
                    out_proj_bias = self.param_transform_via_TM(
                        out_proj_bias, R_transform_matrix=getattr(self, f'out_proj_bias_TM_{embed_dim}'))
                if bias_k is not None:
                    bias_k = self.param_transform_via_TM(
                        bias_k, R_transform_matrix=getattr(self, f'bias_k_TM_{embed_dim}'))
                if bias_v is not None:
                    bias_v = self.param_transform_via_TM(
                        bias_v, R_transform_matrix=getattr(self, f'bias_v_TM_{embed_dim}'))
        return in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias

    @staticmethod
    def param_transform_via_TM(origin_param, L_transform_matrix=None, R_transform_matrix=None):
        if L_transform_matrix is None and R_transform_matrix is not None:
            return torch.matmul(origin_param, R_transform_matrix)
        elif L_transform_matrix is not None and R_transform_matrix is None:
            return torch.matmul(L_transform_matrix, origin_param)
        else:
            return torch.matmul(torch.matmul(L_transform_matrix, origin_param), R_transform_matrix)      

    def isSearchMHA(self):
        self.search_num_heads = False
        self.search_embed_dim = False
        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False
        if is_searchable(getattr(self.value_spaces, 'num_heads', None)):
            self.search_num_heads = True
        if is_searchable(getattr(self.value_spaces, 'embed_dim', None)):
            self.search_embed_dim = True
        return True

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)


if __name__ == '__main__':
    import timeit
    from time import time
    from hyperbox.mutables.ops import Linear
    from hyperbox.networks.base_nas_network import BaseNASNetwork
    from hyperbox.mutator import RandomMutator
    
    class Net(BaseNASNetwork):
        def __init__(self, dim_in, dims, heads, transform_params_method='TruncatedLinear'):
            super().__init__()
            self.linear = Linear(dim_in, dims)
            self.mha = MultiheadAttention(dims, heads, 0.1, True,
                                          transform_params_method=transform_params_method)
            self.linear_time = []
            self.mha_time = []
        def forward(self, x):
            start = time()
            x = self.linear(x)
            end = time()
            self.linear_time.append(end-start)
            
            start = time()
            y = self.mha(x, x, x)
            end = time()
            self.mha_time.append(end-start)
            return y[0]

    for tpm in [
            'disable', 
            'Large2Small', 
            'PoolConv', 
            'TruncatedLinear'
        ]:
        print(f"testing {tpm}...")
        dims = ValueSpace([256, 512, 1024])
        heads = ValueSpace([4, 8, 16])
        # device = 'mps'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dim_in = 128
        net = Net(dim_in, dims, heads, tpm).to(device)
        total_params = sum(p.numel() for p in net.parameters())
        if tpm == 'disable':
            TM_params = 0
        else:
            TM_params = sum(p.numel() for name, p in net.named_parameters() if 'TM' in name)
        print(f"Testing {tpm}: Total params: {total_params}, TM params: {TM_params} ({TM_params/total_params:.2%})")
        rm = RandomMutator(net)
        x = torch.rand(16, 512, dim_in).to(device)

        steps = 100
        start = time()
        reset_times = []
        fw_times = []
        bw_times = []
        for i in range(steps):
            start1 = time()
            rm.reset()
            end1 = time()
            reset_times.append(end1-start1)
            start2 = time()
            y = net(x)
            end2 = time()
            fw_times.append(end2-start2)
            start3 = time()
            loss = y.sum()
            loss.backward()
            end3 = time()
            bw_times.append(end3-start3)
        end = time()
        print(f"Testing {tpm}: Time taken about {(end-start)/(steps-5):.6f} seconds")
        print(f"\treset time: {sum(reset_times[5:])/(steps-5):.6f}")
        print(f"\tforward time: {sum(fw_times[5:])/(steps-5):.6f}")
        print(f"\t\tlinear time: {sum(net.linear_time[5:])/(steps-5):.6f}")
        print(f"\t\tmha time: {sum(net.mha_time[5:])/(steps-5):.6f}")
        print(f"\tbackward time: {sum(bw_times[5:])/(steps-5):.6f}")
        print('*'*20)
