
import torch
from hyperbox.mutator import RandomMutator
from hyperbox.networks.mobilenet.mobile_net import MobileNet
from hyperbox.utils.calc_model_size import flops_size_counter


if __name__ == '__main__':

    net = MobileNet()
    mutator = RandomMutator(net)
    mutator.reset()

    x = torch.rand(1,3,128,128)
    result = flops_size_counter(net, (x,))
    ops, params = [result[k] for k in result]
    print(f"{ops} MFLOPS, {params} MB")

    x = torch.rand(1,3,64,64)
    result = flops_size_counter(net, (x,))
    ops, params = [result[k] for k in result]
    print(f"{ops} MFLOPS, {params} MB")
