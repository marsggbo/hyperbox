import json

import torch
import torch.nn as nn

from hyperbox.utils.utils import TorchTensorEncoder
from hyperbox.networks.bnnas import BNNet
from hyperbox.mutator import EAMutator


if __name__ == '__main__':
    device = 'cuda'
    net = BNNet(num_classes=10, search_depth=False)
    ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_bn_depth_adam0.001_sync_hete/2021-10-02_23-05-51/checkpoints/epoch=390_val/acc=27.8100.ckpt'
    ckpt = torch.load(ckpt, map_location='cpu')
    weights = {}
    for key in ckpt['state_dict']:
        weights[key.replace('network.', '')] = ckpt['state_dict'][key]

    net.load_state_dict(weights)
    net = net.to(device)
    ea = EAMutator(net, num_population=50)

    ea.start_evolve = True
    ea.init_population(ea.init_population_mode)
    for i in range(20):
        print(f"search epoch {i}")
        if i>0:
            ea.evolve()
        for j, arch in enumerate(ea.population.values()):
            ea.reset()
            arch['metric'] = net.bn_metrics().item()
        print([f"{pool['metric']:.4f}" for pool in ea.population.values()])
    with open('pareto_fronts.json', 'w') as f:
        json.dump(ea.population, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)
    pass