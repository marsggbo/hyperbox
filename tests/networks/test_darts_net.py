
import torch

from hyperbox.networks.darts import DartsNetwork, DartsCell
from hyperbox.mutator.random_mutator import RandomMutator


if __name__ == '__main__':
    net = DartsNetwork(3,16,10,4,mask='./hyperbox/networks/darts/darts_mask.json')
    pass

    net = DartsNetwork(
        in_channels=3,
        channels=16,
        n_classes=10,
        n_layers=3,
        factory_func=DartsCell,
    ).cuda()
    print(net.arch)
    m = RandomMutator(net)
    m.reset()
    print(net.arch)
    m.reset()
    print(net.arch)
    m.reset()
    print(net.arch)
    x = torch.rand(2,3,64,64).cuda()
    output = net(x)
    print(output.shape)
    pass