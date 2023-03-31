import torch
from hyperbox.mutables import ops
from hyperbox.mutables.spaces import ValueSpace
from hyperbox.mutator import RandomMutator

from hyperbox.utils.calc_model_size import flops_size_counter

def test_groups(cin, cout, group_list):
    print(f"test_groups: cin={cin}, cout={cout}, group_list={group_list}\n")
    groups = ValueSpace(candidates=group_list)
    x = torch.rand(2, cin, 64, 64)
    conv = ops.Conv2d(cin, cout, 3, 1, 1, groups=groups)
    m = RandomMutator(conv)
    for i in range(10):
        m.reset()
        print(f'\n*******step{i}********\n', conv)
        y = conv(x)

def test_cin_groups(cin_list, cout, group_list):
    print(f"test_cin_groups: cin_list={cin_list}, cout={cout}, group_list={group_list}\n")
    cin = ValueSpace(candidates=cin_list)
    groups = ValueSpace(candidates=group_list)
    x = torch.rand(2, 3, 64, 64)
    conv = torch.nn.Sequential(
        ops.Conv2d(3, cin, 3, 1, 1),
        ops.Conv2d(cin, cout, 3, 1, 1, groups=cin)
    )
    m = RandomMutator(conv)
    for i in range(10):
        m.reset()
        print(f'\n*******step{i}********\n', conv)
        y = conv(x)

def test_cout_groups(cin, cout_list, group_list):
    print(f"test_cout_groups: cin={cin}, cout_list={cout_list}, group_list={group_list}\n")
    cout = ValueSpace(candidates=cout_list)
    groups = ValueSpace(candidates=group_list)
    x = torch.rand(2, cin, 64, 64)
    conv = ops.Conv2d(cin, cout, 3, 1, 1, groups=groups)
    m = RandomMutator(conv)
    for i in range(10):
        m.reset()
        print(f'\n*******step{i}********\n', conv)
        y = conv(x)

def test_cin_cout_groups(cin_list, cout_list, group_list):
    print(f"test_cin_cout_groups: cin_list={cin_list}, cout_list={cout_list}, group_list={group_list}\n")
    cin = ValueSpace(candidates=cin_list)
    cout = ValueSpace(candidates=cout_list)
    groups = ValueSpace(candidates=group_list)
    x = torch.rand(2, 3, 64, 64)
    conv = torch.nn.Sequential(
        ops.Conv2d(3, cin, 3, 1, 1),
        ops.Conv2d(cin, cout, 3, 1, 1, groups=cin)
    )
    m = RandomMutator(conv)
    for i in range(10):
        m.reset()
        print(f'\n*******step{i}********\n', conv)
        y = conv(x)

def test_conv():
    print('testing conv flops and sizes')
    x = torch.rand(1,3,64,64)
    vs1 = ValueSpace([10,2])
    vs2 = ValueSpace([3,5,7])
    conv = ops.Conv2d(3, vs1, vs2,bias=False)
    bn = ops.BatchNorm2d(vs1)
    op = torch.nn.Sequential(
        # torch.nn.Conv2d(3,3,3,1),
        conv, bn
    )
    m = RandomMutator(op)
    m.reset()
    # print(op)
    # print(m._cache)
    # print(conv.weight.shape)
    # print(conv(x).shape)
    r = flops_size_counter(op, (1,3,8,8), False, False)
    # print(r)
    op = torch.nn.Sequential(
        ops.Conv2d(3,8,3,1),
        ops.BatchNorm2d(8)
    )
    r = flops_size_counter(op, (1,3,8,8), False, False)
    # print(conv(x).shape)

if __name__ == '__main__':
    test_conv()
    test_groups(32, 64, group_list=[1, 2, 4, 8, 16, 32])
    test_cin_groups([32, 64], 128, group_list=[32, 64])
    test_cout_groups(8, [16, 32], group_list=[1, 2, 4, 8])
    test_cin_cout_groups([8, 16], [32, 64], group_list=[1, 8, 16])
