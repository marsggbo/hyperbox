import json

import numpy as np
import torch
import torch.nn as nn

from hyperbox.utils.utils import TorchTensorEncoder
from hyperbox.networks.bnnas import BNNet
from hyperbox.mutator.ea_mutator import EAMutator, plot_pareto_fronts
from hyperbox.mutator.utils import NonDominatedSorting


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

    # method 1
    ea = EAMutator(net, num_population=50, algorithm='cars')
    # ea.load_ckpt('epoch2.pth')
    eval_func = lambda arch, net: net.bn_metrics().item()
    ea.search(20, eval_func, verbose=True, filling_history=True)
    size = np.array([pool['size'] for pool in ea.history.values()])
    metric = np.array([pool['metric'] for pool in ea.history.values()])
    indices = np.argsort(size)
    size, metric = size[indices], metric[indices]
    epoch = ea.crt_epoch
    pareto_lists = NonDominatedSorting(np.vstack( (size.reshape(-1), 1/metric.reshape(-1)) ))
    pareto_indices = pareto_lists[0] # e.g., [75,  87, 113, 201, 205]
    plot_pareto_fronts(size, metric, pareto_indices, 'model size (MB)', 'BN-based metric', figname=f'pareto_epoch{epoch}.pdf')

    # method 2
    # ea = EAMutator(net, num_population=50, algorithm='top')

    # ea.start_evolve = True
    # ea.init_population(ea.init_population_mode)
    # for i in range(30):
    #     print(f"\n\nsearch epoch {i}")
    #     if i>0:
    #         ea.evolve()
    #     for j, arch in enumerate(ea.population.values()):
    #         # ea.reset()
    #         ea.reset_cache_mask(arch['arch'])
    #         arch['metric'] = net.bn_metrics().item()
    #     metrics = [pool['metric'] for pool in ea.population.values()]
    #     metrics.sort()
    #     print('pop',metrics)
    #     metrics = [pool['metric'] for pool in ea.pareto_fronts.values()]
    #     metrics.sort()
    #     print('pare',metrics)
    #     visited = []
    #     for arch in ea.pareto_fronts.values():
    #         if arch['arch_code'] not in visited: visited.append(arch['arch_code'])
    #     print(len(visited))