import torch
import torch.nn as nn

from hyperbox.networks.mobilenet.mobile_net import MobileNet
from hyperbox.networks.mobilenet.mobile3d_net import Mobile3DNet

if __name__ == '__main__':
    from hyperbox.mutator.random_mutator import RandomMutator
    net = MobileNet()
    m = RandomMutator(net)
    m.reset()
    x = torch.rand(2,3,64,64)
    output = net(x)
    print(output.shape)
    
    
    net = Mobile3DNet()
    m = RandomMutator(net)
    m.reset()
    x = torch.rand(2,3,64,64,64)
    output = net(x)
    print(output.shape)