import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbox.mutables.ops.base_module import FinegrainedModule
from hyperbox.mutables.ops.utils import is_searchable
from hyperbox.mutables.spaces import ValueSpace


class GroupNorm(nn.GroupNorm, FinegrainedModule):

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        _num_groups = num_groups.max_value if isinstance(num_groups, ValueSpace) else num_groups
        _num_channels = num_channels.max_value if isinstance(num_channels, ValueSpace) else num_channels
        super(GroupNorm, self).__init__(_num_groups, _num_channels, eps, affine)
        self.is_search = self.isSearchGroupNorm()

    def isSearchGroupNorm(self):
        if all([not vs.is_search for vs in self.value_spaces.values()]):
            return False

        if is_searchable(getattr(self.value_spaces, 'num_groups', None)):
            self.search_num_groups = True
        else:
            self.search_num_groups = False
        if is_searchable(getattr(self.value_spaces, 'num_channels', None)):
            self.search_num_channels = True
        else:
            self.search_num_channels = False
        return True

    def forward(self, input):
        if not self.is_search:
            x = F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
        else:
            n_groups = self.value_spaces['num_groups'].value if self.search_num_groups else self.num_groups
            n_channels = self.value_spaces['num_channels'].value if self.search_num_channels else self.num_channels
            return F.group_norm(input, n_groups, self.weight[:n_channels], self.bias[:n_channels], self.eps)

    @property
    def params(self):
        weight = self.weight
        bias = self.bias
        if self.search_num_channels:
            weight = weight[:self.value_spaces['num_channels'].value]
            bias = bias[:self.value_spaces['num_channels'].value] if bias is not None else None
        parameters = [weight, bias]
        params = sum([p.numel() for p in parameters if p is not None])
        return params


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator
    from hyperbox.mutables.ops import Conv2d
    input = torch.randn(20, 6, 10, 10)

    #### no search
    ##### Separate 6 channels into 3 groups
    print("no search")
    m = nn.GroupNorm(3, 6)
    output = m(input)
    print(output.shape)
    ##### Separate 6 channels into 6 groups (equivalent with InstanceNorm)
    m = nn.GroupNorm(6, 6)
    output = m(input)
    print(output.shape)
    ##### Put all 6 channels into a single group (equivalent with LayerNorm)
    m = nn.GroupNorm(1, 6)
    output = m(input)
    print(output.shape)
    
    
    #### search only n_groups
    print("search only n_groups")
    n_groups = ValueSpace([2, 3, 6, 1], key='num_groups')
    m = GroupNorm(n_groups, 6)
    rm = RandomMutator(m)
    for i in range(5):
        rm.reset()
        output = m(input)
        print(output.shape)
    
    # search only n_channels
    print("search only n_channels")
    n_channels = ValueSpace([12, 18, 24], key='num_channels')
    m = nn.Sequential(
        Conv2d(6, n_channels, 3, 1, 1),
        GroupNorm(3, n_channels)
    )
    rm = RandomMutator(m)
    for i in range(10):
        rm.reset()
        output = m(input)
        print(output.shape)
    
    # search n_groups and n_channels
    print("search n_groups and n_channels")
    n_groups = ValueSpace([2, 3, 6, 1], key='num_groups2')
    n_channels = ValueSpace([12, 18, 24], key='num_channels2')
    m = nn.Sequential(
        Conv2d(6, n_channels, 3, 1, 1),
        GroupNorm(n_groups, n_channels)
    )
    rm = RandomMutator(m)
    for i in range(10):
        rm.reset()
        output = m(input)
        print(output.shape)