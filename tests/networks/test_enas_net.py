
import torch

from hyperbox.networks.enas import ENASMicroNetwork
from hyperbox.mutator.random_mutator import RandomMutator

if __name__ == '__main__':
    net = ENASMicroNetwork(
        num_layers=2,
        num_nodes=3,
    ).cuda()
    m = RandomMutator(net)
    m.reset()
    x = torch.rand(2,3,64,64).cuda()
    output = net(x)
    print(output.shape)
    pass