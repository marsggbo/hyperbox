import json

import torch
import torch.nn as nn

from hyperbox.utils.utils import TorchTensorEncoder
from hyperbox.networks.bnnas import BNNet
from hyperbox.mutator import EAMutator


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = BNNet(num_classes=10, search_depth=False)
    ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all_depth_adam0.001_sync_hete/2021-10-02_23-05-43/checkpoints/epoch=339_val/acc=43.1300.ckpt'
    ckpt = torch.load(ckpt, map_location='cpu')
    weights = {}
    for key in ckpt['state_dict']:
        weights[key.replace('network.', '')] = ckpt['state_dict'][key]

    net.load_state_dict(weights)
    net = net.to(device)
    ea = EAMutator(net, num_population=50, algorithm='top')

    ea.start_evolve = True
    ea.init_population(ea.init_population_mode)
    for i in range(30):
        print(f"\n\nsearch epoch {i}")
        if i>0:
            ea.evolve()
        for j, arch in enumerate(ea.population.values()):
            # ea.reset()
            ea.reset_cache_mask(arch['arch'])
            arch['metric'] = net.bn_metrics().item()
        metrics = [pool['metric'] for pool in ea.population.values()]
        metrics.sort()
        print('pop',metrics)
        metrics = [pool['metric'] for pool in ea.pareto_fronts.values()]
        metrics.sort()
        print('pare',metrics)
        visited = []
        for arch in ea.pareto_fronts.values():
            if arch['arch_code'] not in visited: visited.append(arch['arch_code'])
        print(len(visited))
        if i>4:
            pass
    # with open('pareto_fronts.json', 'w') as f:
    #     json.dump(ea.population, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)
    pass