import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from utils.calc_model_size import flops_size_counter
from utils.utils import TorchTensorEncoder
from mutables import InputChoice, LayerChoice

from .random_mutator import RandomMutator
from .utils import CARS_NSGA, encode_arch

__all__ = [
    'EAMutator',
]


OBJECT_KEY_MAP = {
    0: 'flops',
    1: 'size',
}

TARGET_KEY_MAP = {
    0: 'meters',
    1: 'speed',
}

class EAMutator(RandomMutator):
    def __init__(self, model, cfg=None):
        '''
        Args:
            cfg: Optional[DictConfig, dict]
        '''
        super().__init__(model, cfg)
        self.cfg = cfg
        EACfg = self.cfg.mutator.EAMutator
        self.algorithm = EACfg.algorithm
        self.target_keys = [TARGET_KEY_MAP[k] for k in EACfg.target_keys]
        self.object_keys = [OBJECT_KEY_MAP[k] for k in EACfg.object_keys]
        self.num_population = EACfg.num_population # 初始population数量
        self.warmup_epochs  = EACfg.warmup_epochs
        self.init_population_mode  = EACfg.init_population_mode # 初始化population的方式
        self.prob_crossover = EACfg.prob_crossover
        self.prob_mutation  = EACfg.prob_mutation
        self.offspring_ratio = EACfg.offspring_ratio

        self.start_evolve = False
        self.is_initialized = False
        self.idx = 0 # 当前的arch索引
        self.num_crt_population = self.num_population # 最新的population数量
        self.random_state = np.random.RandomState(0)

        # pools = {
            #  {'arch':arch, 'metric':0.92, 'flops':100MFLOPS, 'size':100MB}, # subnetwork
            # {'arch':arch, 'metric':0.97, 'flops':200MFLOPS, 'size':300MB}  # supernet
        # }
        self.pools = {}
        self.history = {} # todo: 记录所有arch的结果

    def reset(self):
        """
        Reset the mutator by call the `sample_search` to resample (for search). Stores the result in a local
        variable so that `on_forward_layer_choice` and `on_forward_input_choice` can use the decision directly.
        """
        if not self.start_evolve:
            self._cache = self.sample_search()
        else:
            if self.idx > self.num_population-1:
                self.idx = 0
            if len(self.pools) < self.num_population:
                self.pools = self.init_population(self.init_population_mode)
                self.idx = 0
                self.is_initialized = True
            self._cache = self.pools[self.idx]['arch']
            self.idx += 1

    def add_new_arch(self, pools, idx, arch):
        if not self.is_pool_repeated(pools, arch):
            self._cache = arch
            size = self.cfg.input.size
            if self.cfg.dataset.is_3d:
                input_size = (1,1,self.cfg.dataset.slice_num,*size)
            else:
                input_size = (1,3,*size)
            flops_size = flops_size_counter(self.model, input_size)
            flops, size = flops_size['flops'], flops_size['size']
            if idx not in pools:
                pools[idx] = {}
            pools[idx]['arch'] = arch
            pools[idx]['arch_code'] = self.encode_arch(arch)
            pools[idx]['flops'] = flops
            pools[idx]['size'] = size
            return True
        return False

    def init_population(self, mode='random'):
        '''初始化population
        '''
        POOLS = {} if len(self.pools) < self.num_population else self.pools
        if mode == 'random':
            idx = 0
            while len(POOLS) < self.num_population+1:
                arch = self.sample_search()
                if self.add_new_arch(POOLS, idx, arch):
                    idx += 1
        elif mode == 'warmup':
            sorted_pools = sorted(self.history.items(), key=lambda x: x[1]['speed'], reverse=True)
            # sorted_pools = sorted(self.history.items(), key=lambda x: np.mean(x[1]['meters']), reverse=True)
            for idx, (key, value) in enumerate(sorted_pools):
                if idx >= self.num_population:
                    break
                value_cp = value.copy()
                value_cp.pop('meters', None)
                POOLS[idx] = value_cp
            # num_remaining = self.num_population - len(POOLS)
            while idx < self.num_population:
                arch = self.sample_search()
                if self.add_new_arch(POOLS, idx, arch):
                    idx += 1
        return POOLS

    def gen_offsprings(self):
        offsprings = {}
        idx = 0
        n_offspring = int(self.offspring_ratio * len(self.pools))
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < self.prob_mutation:
                index = np.random.randint(0, self.num_population)
                arch = self.mutation(self.pools[index]['arch'])
            elif rand < 0.5:
                index1 = np.random.randint(0, self.num_population)
                index2 = np.random.randint(0, self.num_population)
                while index1 == index2:
                    index2 = np.random.randint(0, self.num_population)
                arch = self.crossover(
                    self.pools[index1]['arch'],
                    self.pools[index2]['arch'])
            else:
                arch = self.sample_search()
            if self.add_new_arch(offsprings, idx, arch):
                idx += 1
        return offsprings

    def expand_population(self):
        '''扩充population
        '''
        offsprings = self.gen_offsprings()
        individuals = [offsprings[idx] for idx in offsprings]
        individuals.extend([self.pools[idx] for idx in range(self.num_population)])
        self.pools = {}
        for idx, individual in enumerate(individuals):
            self.pools[idx] = individual
        return len(self.pools)

    def fitness(self, population):
        targets = []
        if 'meters' in self.target_keys:
            meters = np.array([individual['meters'] for key, individual in population.items()])
            targets.append(meters)
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

    def fitness_increase_speed(self, population):
        speeds = []
        for key, individual in population.items():
            arch_code = individual['arch_code']
            if arch_code in self.history:
                speed = self.history[arch_code]['speed']
            else:
                speed = individual['meters']
            speeds.append(speed)
        return speeds

    def selection(self):
        try:
            probs = np.array([self.fitness(self.pools[idx]) for idx in range(self.num_population)])
            nan_indices = np.isnan(probs)
            probs[nan_indices] = probs[~nan_indices].mean()
            # indices = np.random.choice(np.arange(self.num_population), self.num_population, replace=True, p=probs/sum(probs))
            indices = np.random.choice(np.arange(self.num_population), self.num_selection, replace=False, p=probs/sum(probs))
        except Exception as e:
            print(probs, e)
            # indices = np.random.choice(np.arange(self.num_population), self.num_population, replace=True)
            indices = np.random.choice(np.arange(self.num_population), self.num_selection, replace=False)
        return indices

    def crossover(self, arch1, arch2):
        cross_arch = arch1.copy()
        for key in arch1:
            if self.random_state.rand() < self.prob_crossover:
                cross_arch[key] = arch2[key]
        return cross_arch

    def mutation(self, arch):
        if self.random_state.rand() > self.prob_mutation:
            return arch
        new_arch = arch.copy()
        for key in arch:
            if self.random_state.rand() < self.prob_mutation:
                length = len(arch[key])
                index = torch.randint(high=length, size=(1,))
                while index == arch[key].float().argmax():
                    index = torch.randint(high=length, size=(1,))
                new_arch[key] = F.one_hot(index, num_classes=length).view(-1).bool()
        return new_arch

    def encode_arch(self, arch):
        return encode_arch(arch)

    def is_pool_repeated(self, pools, new_arch):
        '''判断新arch是否已经在pools
        '''
        if len(pools)<=1:
            return False
        for idx in pools:
            if idx == -1:
                continue
            arch = pools[idx]['arch']
            if self.is_repeat(arch, new_arch):
                return True
        return False

    def is_repeat(self, arch, new_arch):
        '''判断两个arch是否相同
        '''
        for key in arch:
            if sum(torch.abs(arch[key].float()-new_arch[key].float())) != 0:
                return False
        return True

    def calculate_speed(self, meters):
        '''the slop of the meters change
            meters: (list) a list of meters, e.g. accuracy=[0.2,0.4,0.5]
        '''
        if len(meters) <= 1:
            speed = meters[0]
        else:
            speed = np.polyfit(np.arange(len(meters)), np.array(meters), 1)[0] # slope of increase curve
            if speed < 0: speed = (abs(speed)+1e-8) * 1e-8
        return speed

    def update_history(self, individual):
        key = individual['arch_code']
        if key not in self.history:
            self.history[key] = {}
            self.history[key]['meters'] = []
        for feature in individual: # feature = {arch_code, arch, meters, flops, size}
            if feature == 'meters':
                self.history[key][feature].append(individual[feature])
            else:
                self.history[key][feature] = individual[feature]
        self.history[key]['speed'] = self.calculate_speed(self.history[key]['meters'])

    def update_individual(self, idx, meters):
        '''更新单个individual的性能
        '''
        if idx not in self.pools:
            self.pools[idx] = {}
        self.pools[idx]['meters'] = meters['save_metric'].avg
        self.update_history(self.pools[idx])

    def save_history(self, history):
        file_path = os.path.join(self.cfg.logger.path, 'history.json')
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=4, cls=TorchTensorEncoder)

    def evolve(self):
        '''进化
        '''
        try:
            targets = self.fitness(self.pools)
            objectives = self.objectives(self.pools, self.object_keys)
            if self.algorithm == 'cars':
                    indices = CARS_NSGA(targets, objectives, self.num_population)
        except Exception as e:
            print(e)
        pools = {}
        for i, idx in enumerate(indices):
            pools[i] = self.pools[idx]
        self.pools = pools
        self.num_crt_population = self.expand_population()
