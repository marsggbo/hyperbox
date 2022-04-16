import torch.nn as nn
import torch
from collections import OrderedDict

from hyperbox.mutables.spaces import OperationSpace
from hyperbox.networks.base_nas_network import BaseNASNetwork
from hyperbox.utils.utils import load_json

candidate_OP = ['id', 'ir_3x3_t3', 'ir_5x5_t6']
OPS = OrderedDict()
OPS['id'] = lambda inp, oup, stride: Identity(inp=inp, oup=oup, stride=stride)
OPS['ir_3x3_t3'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=3, stride=stride, k=3)
OPS['ir_5x5_t6'] = lambda inp, oup, stride: InvertedResidual(inp=inp, oup=oup, t=6, stride=stride, k=5)


class Identity(nn.Module):
    def __init__(self, inp, oup, stride):
        super(Identity, self).__init__()
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x



class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t, k=3, activation=nn.ReLU, use_se=False, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        hidden_dim = round(inp * t)
        if t == 1:
            self.conv = nn.Sequential(
                # dw            
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, k, stride, padding=k//2, groups=hidden_dim, 
                              bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)



class NASBenchMBNet(BaseNASNetwork):
    def __init__(
        self,
        arch_list: list=None,
        num_classes: int=10,
        stages: list=[2, 3, 3],
        init_channels: int=32,
        mask=None
    ):
        assert arch_list is None or mask is None, 'You cannot set both arch_list and mask'
        super(NASBenchMBNet, self).__init__(mask)
        if mask is not None:
            assert len(mask) == sum(stages)
        if arch_list is not None:
            assert len(arch_list) == sum(stages)
        self.arch_list = arch_list

        self.stem = nn.Sequential(
            nn.Conv2d(3, init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        arch_ = arch_list
        features = []
        channels = init_channels
        for stage_id, stage_len in enumerate(stages):
            for idx in range(stage_len):
                if idx == 0:
                    # stride = 2 
                    ops = [op_func(channels, channels*2, 2) for key, op_func in OPS.items()]
                    channels *= 2
                else:
                    ops = [op_func(channels, channels, 1) for key, op_func in OPS.items()]
                index = int(arch_[len(features)]) if arch_ is not None else None
                ops = OperationSpace(candidates=ops, key=f"stage{stage_id}_{idx}", mask=mask, index=index)
                features.append(ops)

        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            nn.Conv2d(channels, 1280, kernel_size=1, bias=False, stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(1280, num_classes)


    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.out(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out

    @property
    def arch(self):
        arch_list = []
        if self.mask is not None:
            for key, value in self.mask.items():
                arch_list.append(value.cpu().detach().numpy().argmax())
        else:
            for block in self.features:
                if isinstance(block, OperationSpace):
                    arch_list.append(block.mask.cpu().detach().numpy().argmax())
        return ''.join([str(x) for x in arch_list])


if __name__ == '__main__':
    from hyperbox.mutator import RandomMutator

    # test search and query
    import json
    with open('./nasbench_mbnet_cifar10.json', 'r') as f:
        data = json.load(f)
    net = NASBenchMBNet()
    rm = RandomMutator(net)
    for i in range(5):
        rm.reset()
        print(net.arch)
        print(data[net.arch])
        print('='*20)
    
    # test mask
    mask = rm._cache
    net = NASBenchMBNet(mask=mask)
    print(net.arch)

    # test arch
    arch = net.arch
    net = NASBenchMBNet(arch_list=arch)
    print(net.arch)
