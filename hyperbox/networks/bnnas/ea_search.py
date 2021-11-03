import json

import numpy as np
import torch
import torch.nn as nn

from hyperbox.utils.utils import TorchTensorEncoder, save_arch_to_json
from hyperbox.networks.bnnas import BNNet
from hyperbox.mutator.ea_mutator import EAMutator, plot_pareto_fronts
from hyperbox.mutator.utils import NonDominatedSorting


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = BNNet(num_classes=10, search_depth=False)
    # ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_bn_depth_adam0.001_sync_hete/2021-10-02_23-05-51/checkpoints/epoch=390_val/acc=27.8100.ckpt'
    # ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all_depth_adam0.001_sync_hete/2021-10-02_23-05-43/checkpoints/epoch=339_val/acc=43.1300.ckpt'
    # ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_bn_adam0.001_sync_hete/2021-10-06_06-29-41/checkpoints/epoch=392_val/acc=28.8200.ckpt'
    # ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all_adam0.001_sync_hete/2021-10-06_06-31-00/checkpoints/epoch=302_val/acc=44.0300.ckpt'
    # ckpt = '/datasets/xinhe/xinhe/hyperbox/logs/runs/bnnas_c10_all_bn/2021-10-22_04-03-29/checkpoints/epoch=14_val/acc=25.1800.ckpt'
    ckpt = '/home/xihe/xinhe/hyperbox/logs/runs/bnnas_c10_all20_bn20/2021-10-27_05-33-45/checkpoints/epoch=39_val/acc=39.3800.ckpt'
    ckpt = torch.load(ckpt, map_location='cpu')
    weights = {}
    for key in ckpt['state_dict']:
        weights[key.replace('network.', '')] = ckpt['state_dict'][key]

    net.load_state_dict(weights)
    net = net.to(device)

    # method 1
    mode = 'all20_bn20'
    search_algorithm = 'cars'
    ea = EAMutator(net, num_population=50, algorithm=search_algorithm)
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
    plot_pareto_fronts(
        size, metric, pareto_indices, 'model size (MB)', 'BN-based metric',
        figname=f'{mode}_pareto_searchepoch{epoch}_{search_algorithm}.pdf'
    )

    path = ckpt.split('checkpoints')[0]
    path = os.path.join(path, 'mask_json')
    if not os.path.exists(path):
        os.makedirs(path)
    for key, arch_info in ea.pareto_fronts.items():
        arch = arch_info['arch']
        arch_path = os.path.join(path, f'arch_{key}.json')
        save_arch_to_json(arch, arch_path)
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