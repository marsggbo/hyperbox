import json
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from hyperbox.utils.calc_model_size import flops_size_counter
from hyperbox.utils.utils import TorchTensorEncoder
from hyperbox.mutables.spaces import InputSpace, OperationSpace

from hyperbox.mutator.random_mutator import RandomMutator
from hyperbox.mutator.utils import CARS_NSGA

__all__ = [
    'EAMutator',
]


OBJECT_KEY_MAP = {
    0: 'flops',
    1: 'size',
}

TARGET_KEY_MAP = {
    0: 'metric',
    1: 'speed',
    2: 'acc'
}


class EAMutator(RandomMutator):
    def __init__(
        self,
        model: torch.nn.Module,
        target_keys: list = [0],
        object_keys: list = [0],
        warmup_epochs: int = 10,
        num_population: int = 50,     # total number of individuals in a population
        ratio_selection: int = 0.5,   # ratio of top individuals to select
        ratio_mutation: int = 0.2,    # ratio of new individuals generated by mutation
        ratio_crossover: int = 0.2,   # ratio of new individuals generated by crossover
        ratio_random: float = 0.1,    # ratio of new individuals generated by random
        prob_crossover: float = 0.2,  # probability of crossover
        prob_mutation: float = 0.1,   # probability of mutation
        init_population_mode: str = 'random', # or 'warmup'
        algorithm: str = 'cars',
        *args, **kwargs
    ):
        '''
        Args:
        '''
        super().__init__(model)
        self.algorithm = algorithm
        self.target_keys = [TARGET_KEY_MAP[k] for k in target_keys]
        self.object_keys = [OBJECT_KEY_MAP[k] for k in object_keys]
        self.num_population = num_population # 初始population数量
        self.warmup_epochs  = warmup_epochs
        self.init_population_mode  = init_population_mode # 初始化population的方式
        self.ratio_selection = ratio_selection
        self.ratio_mutation = ratio_mutation
        self.ratio_crossover = ratio_crossover
        self.ratio_random = ratio_random
        self.prob_crossover = prob_crossover
        self.prob_mutation  = prob_mutation

        self.start_evolve = False # when current epoch > warmup_epochs, set True
        self.is_initialized = False
        self.idx = -1 # 当前的arch索引. start from 0. -1 is invalid index
        self.num_crt_population = self.num_population # 最新的population数量
        self.random_state = np.random.RandomState(0)

        # population = {
            #  {'arch':arch, 'metric':0.92, 'flops':100MFLOPS, 'size':100MB}, # subnetwork
            # {'arch':arch, 'metric':0.97, 'flops':200MFLOPS, 'size':300MB}  # supernet
        # }
        self.population = {}   # population of each search epoch
        self.history = {}      # todo: 记录所有arch的结果
        self.pareto_fronts = {}

        self.crt_epoch = 0  # current search epoch

    def reset(self):
        """
        Reset the mutator by call the `sample_search` to resample (for search). Stores the result in a local
        variable so that `on_forward_operation_space` and `on_forward_input_space` can use the decision directly.
        """
        if not self.start_evolve:
            arch = self.sample_search()
            self.reset_cache_mask(arch)
        else:
            assert self.num_population == len(self.population), \
                f"{self.num_population}!={len(self.population)}"
            idx = self.idx % self.num_population
            self.reset_cache_mask(self.population[idx]['arch'])
            self.idx += 1

    def reset_cache_mask(self, arch: dict):
        self._cache = arch
        for mutable in self.mutables:
            mutable.mask = self._cache[mutable.key]

    def init_population(self, mode='random'):
        '''initialize population
        Args:
        - mode
            - random: randomly generate archs
            - warmup: generate first population by warup performance
        '''
        POOLS = {} if len(self.population) < self.num_population else self.population
        if mode == 'random':
            idx = 0
            while len(POOLS) < self.num_population:
                arch = self.sample_search()
                if self.add_new_arch(POOLS, idx, arch):
                    idx += 1
        elif mode == 'warmup':
            # sorted_pools = sorted(self.history.items(), key=lambda x: x[1]['speed'], reverse=True)
            sorted_pools = sorted(self.history.items(), key=lambda x: np.mean(x[1]['metric']), reverse=True)
            for idx, (key, value) in enumerate(sorted_pools):
                if idx >= self.num_population:
                    break
                value_cp = value.copy()
                value_cp.pop('metric', None)
                POOLS[idx] = value_cp
            # num_remaining = self.num_population - len(POOLS)
            while idx < self.num_population:
                arch = self.sample_search()
                if self.add_new_arch(POOLS, idx, arch):
                    idx += 1
        self.population = POOLS
        return POOLS

    def fitness(self, population, fitness_name='metric'):
        '''Fitness function for a population
        Args:
        - population: dict. 
            example: (each individual can be obtained by index)
            population = {
                0: {'arch':arch, 'metric':0.92, 'flops':100MFLOPS, 'size':100MB}, # arch1
                1: {'arch':arch, 'metric':0.97, 'flops':200MFLOPS, 'size':300MB}  # arch2
                ...
            }
        Return: np.array
        '''
        targets = []
        if 'metric' in self.target_keys:
            metric = np.array([individual['metric'] for key, individual in population.items()])
            targets.append(metric)
        if 'speed' in self.target_keys:
            speeds = np.array(self.fitness_increase_speed(population))
            targets.append(speeds)
        targets = np.vstack(targets) if targets else np.random.rand(len(population))
        return targets

    def objectives(self, population, obj_keys):
        '''根据obj_keys生成对应的object值 (比如flops, acc)
        Args:
            population: (dict) {0: {'arch':..., 'flops':23, 'size':12}}
            obj_keys: (list) ['flops', 'size']
        Return: np.array
        '''
        objs = []
        for key in obj_keys:
            try:
                objs.append([individual[key] for idx, individual in population.items()])
            except Exception as e:
                print(f'{e}\n{key} not found')
                objs.append([0 for idx in population])
        objs = np.vstack(objs)
        return objs

    def selection(self, population, num_selection, by_probability=False):
        '''Selection step:
        select `num_selection` individuals with probability of their fitness
        '''
        fitness = self.fitness(population)
        num_population = len(population)
        if by_probability:
            try:
                probs = fitness
                nan_indices = np.isnan(probs)
                probs[nan_indices] = probs[~nan_indices].mean()
                indices = np.random.choice(np.arange(num_population), num_selection, replace=False, p=probs/sum(probs))
            except Exception as e:
                print(e)
                indices = np.random.choice(np.arange(num_population), num_selection, replace=False)
        else:
            top_values, top_indices = torch.tensor(fitness).topk(num_selection)
            indices = top_indices.view(-1).tolist()
        return indices

    def crossover(self, arch1, arch2):
        '''Crossover step
        Generate a new arch by randomly crossover part of architectures of two parent individuals
        '''
        cross_arch = deepcopy(arch1)
        for key in arch1:
            if self.random_state.rand() < self.prob_crossover:
                cross_arch[key] = deepcopy(arch2[key])
        return cross_arch

    def mutation(self, arch):
        '''Mutation step
        Mutate a parent individual
        '''
        new_arch = deepcopy(arch)
        for key in arch:
            length = len(arch[key])
            if length > 1 and self.random_state.rand() < self.prob_mutation:
                delete_index = arch[key].float().argmax()
                select_range = list(range(delete_index)) + list(range(delete_index+1, length))
                index = torch.tensor(np.random.choice(select_range))
                new_arch[key] = F.one_hot(index, num_classes=length).view(-1).bool()
        return new_arch

    def encode_arch(self, arch):
        '''Encoding an architecture to a string.
        The encoding string is used to verify whether it has been generated before.
        If so, then there is no need to re-calculate the size and flops for this individual.
        '''
        code = []
        for key in arch:
            code.append(str(arch[key].int().argmax().item()))
        return ''.join(code)

    def add_new_arch(self, pools, idx, arch, input_size=(2,3,64,64)):
        '''Add a new generated architecture if not created before.
        '''
        arch_code = self.encode_arch(arch)
        if not self.is_arch_existed(self.history, arch):
            # if not existed in `self.history`, then calculate its flops and size
            self.reset_cache_mask(arch)
            flops_size = flops_size_counter(self.model, input_size)
            flops, size = flops_size['flops'], flops_size['size']
            if idx not in pools:
                pools[idx] = {}
            individual = {
                'arch': arch,
                'arch_code': arch_code,
                'flops': flops,
                'size': size
            }
            self.history[arch_code] = individual
            pools[idx] = deepcopy(individual)
            return True
        if not self.is_arch_existed(pools, arch):
            # existed in self.history not population
            pools[idx] = deepcopy(self.history[arch_code])
            return True
        return False

    def is_arch_existed(self, pools, new_arch):
        '''判断新arch是否已经在pools
        '''
        def is_repeat(arch, new_arch):
            '''判断两个arch是否相同
            '''
            for key in arch:
                if sum(torch.abs(arch[key].float()-new_arch[key].float())) != 0:
                    return False
            return True
        if len(pools)<1:
            return False
        for idx in pools:
            if idx == -1:
                continue
            arch_code = pools[idx]['arch_code']
            new_arch_code = self.encode_arch(new_arch)
            if arch_code == new_arch_code:
                return True
            # arch = pools[idx]['arch']
            # if is_repeat(arch, new_arch):
            #     return True
        return False

    def evolve(self):
        '''进化
        '''
        try:
            self.update_pareto_fronts(key='metric', reverse=True)
            targets = self.fitness(self.pareto_fronts)
            num_selection = int(round(self.ratio_selection * self.num_population))
            if self.algorithm == 'cars':
                objectives = self.objectives(self.pareto_fronts, self.object_keys)
                indices = CARS_NSGA(targets, objectives, num_selection)
            else:
                indices = self.selection(self.pareto_fronts, num_selection)
        except Exception as e:
            print(e)
        pools = {}
        for i, idx in enumerate(indices):
            pools[i] = deepcopy(self.pareto_fronts[idx])
        self.population = pools
        self.expand_population()

    def expand_population(self):
        '''扩充population
        '''
        calc_num = lambda ratio, total_num: int(round(total_num*ratio))
        offsprings = self.population
        crt_num_population = len(offsprings)
        idx = len(self.population)
        num_mutation = calc_num(self.ratio_mutation, self.num_population)
        num_crossover = calc_num(self.ratio_crossover, self.num_population)
        num_random = calc_num(self.ratio_random, self.num_population)
        num_offspring = num_mutation + num_crossover + num_random

        while num_mutation > 0:
            index = np.random.randint(0, crt_num_population)
            arch = self.mutation(self.population[index]['arch'])
            if self.add_new_arch(offsprings, idx, arch):
                idx += 1; num_mutation -= 1
        while num_crossover > 0:
            index1 = np.random.randint(0, crt_num_population)
            remain_spaces = list(range(index1)) + list(range(index1+1, crt_num_population))
            index2 = np.random.choice(remain_spaces)
            arch = self.crossover(
                self.population[index1]['arch'],
                self.population[index2]['arch'])
            if self.add_new_arch(offsprings, idx, arch):
                idx += 1; num_crossover -= 1
        while num_random > 0:
            arch = self.sample_search()
            if self.add_new_arch(offsprings, idx, arch):
                idx += 1; num_random -= 1

    def update_pareto_fronts(self, key='metric', reverse=True):
        candidates = list(deepcopy(self.pareto_fronts).values()) + list(deepcopy(self.population).values())
        pareto_fronts = []
        visited = []
        for cand in candidates:
            arch_code = cand['arch_code']
            if arch_code not in visited:
                visited.append(arch_code)
                pareto_fronts.append(cand)
        pareto_fronts = sorted(pareto_fronts, key=lambda x:x[key], reverse=reverse)
        pareto_fronts = pareto_fronts[:self.num_population]
        self.pareto_fronts = {}
        for i in range(self.num_population):
            self.pareto_fronts[i] = pareto_fronts[i]

    def update_individual(self, idx, metric):
        '''更新单个individual的性能
        '''
        if idx not in self.population:
            self.population[idx] = {}
        self.population[idx]['metric'] = metric
        self.update_history(self.population[idx])

    def fitness_increase_speed(self, population):
        speeds = []
        for key, individual in population.items():
            arch_code = individual['arch_code']
            if arch_code in self.history:
                speed = self.history[arch_code]['speed']
            else:
                speed = individual['metric']
            speeds.append(speed)
        return speeds

    def calculate_speed(self, metric):
        '''the slop of the metric change
            metric: (list) a list of metric, e.g. accuracy=[0.2,0.4,0.5]
        '''
        if len(metric) <= 1:
            speed = metric[0]
        else:
            speed = np.polyfit(np.arange(len(metric)), np.array(metric), 1)[0] # slope of increase curve
            if speed < 0: speed = (abs(speed)+1e-8) * 1e-8
        return speed

    def update_history(self, individual):
        key = individual['arch_code']
        if key not in self.history:
            self.history[key] = {}
            self.history[key]['metric'] = []
        for feature in individual: # feature = {'arch_code', 'arch', 'metric', 'flops', 'size'}
            if feature == 'metric':
                self.history[key][feature].append(individual[feature])
            else:
                self.history[key][feature] = individual[feature]
        self.history[key]['speed'] = self.calculate_speed(self.history[key]['metric'])

    def save_history(self, path, history):
        file_path = os.path.join(path, 'history.json')
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=4, cls=TorchTensorEncoder)

    def save_ckpt(self, path_ckpt='.', filename=None):
        if not os.path.exists(path_ckpt):
            os.makedirs(path_ckpt)
        if filename is None:
            filename = f"epoch{self.crt_epoch}.pth"
        ckpt = {}
        ckpt['epoch'] = self.crt_epoch
        ckpt['history'] = self.history
        ckpt['pareto_fronts'] = self.pareto_fronts
        ckpt['population'] = self.population
        self.ckpt_name = os.path.join(path_ckpt, filename)
        torch.save(ckpt, self.ckpt_name)
        print('Save checkpoint to', self.ckpt_name)

    def load_ckpt(self, file):
        if not os.path.exists(file):
            print(file, 'not existed')
            return False
        ckpt = torch.load(file)
        self.crt_epoch = ckpt['epoch']
        self.history = ckpt['history']
        self.pareto_fronts = ckpt['pareto_fronts']
        self.population = ckpt['population']
        print('Load checkpoint from', file)

    def search(self, search_epochs, eval_func, verbose=False, filling_history=False):
        '''Start search using EA
        Args:
        - search_epochs: int. # max epochs of searching
        - eval_func: evaluation function of arch performance
        - filling_history: bool. calc metrics for all individuals in `history`
        '''

        self.start_evolve = True
        self.init_population(self.init_population_mode) # init 
        self.crt_epoch = 0
        while self.crt_epoch < search_epochs:
            print(f"Evolution epoch={self.crt_epoch}")
            if self.crt_epoch > 0:
                self.evolve()
            for j, arch in enumerate(self.population.values()):
                self.reset_cache_mask(arch['arch'])
                if arch.get('metric') is None:
                    arch['metric'] = eval_func(arch, self.model)
            self.crt_epoch += 1
            if verbose:
                metrics = [pool['metric'] for pool in self.population.values()]
                metrics.sort()
                print('population',metrics)
                metrics = [pool['metric'] for pool in self.pareto_fronts.values()]
                metrics.sort()
                print('pareto_fronts',metrics)
        
        if filling_history:
            for j, arch in enumerate(self.history.values()):
                self.reset_cache_mask(arch['arch'])
                if arch.get('metric') is None:
                    arch['metric'] = eval_func(arch, self.model)
        self.save_ckpt()
        print(f'Evolution is finished')

def plot_pareto_fronts(
        obj1: np.array,
        obj2: np.array,
        pareto_indices: np.array,
        name_obj1: str='obj1',
        name_obj2: str='obj2',
        figsize=(8,5),
        figname=None
    ):

    import matplotlib.pyplot as plt
    obj1_pareto = [x for x in obj1[pareto_indices]]
    obj2_pareto = [x for x in obj2[pareto_indices]]
    fig = plt.figure(num=1, figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.scatter(obj1, obj2)
    ax1.plot(obj1_pareto, obj2_pareto)
    ax1.set_xlabel(name_obj1)
    ax1.set_ylabel(name_obj2)
    plt.show()
    if figname is not None:
        fig.savefig(figname)

if __name__ == '__main__':
    from hyperbox.networks.bnnas.bn_net import BNNet
    net = BNNet()
    ea = EAMutator(net)
    ea.start_evolve = True
    ea.reset()
    for i in range(2):
        for j, arch in enumerate(ea.pools.values()):
            arch['metric'] = random.random()
            ea.reset()
        ea.evolve()
    pass