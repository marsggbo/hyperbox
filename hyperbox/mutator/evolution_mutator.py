import os
from copy import deepcopy
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from hyperbox.mutables.spaces import InputSpace, OperationSpace, ValueSpace
from hyperbox.mutator.default_mutator import Mutator
from hyperbox.mutator.random_mutator import RandomMutator
from hyperbox.mutator.utils import CARS_NSGA, NonDominatedSorting, nsga2_select
from hyperbox.utils.calc_model_size import flops_size_counter
from hyperbox.utils.logger import get_logger


log = get_logger(__name__, level='INFO')

__all__ = [
    'EvolutionMutator',
]


def get_int_num(target_num: Optional[Union[int, float]], total_num: int=None) -> int:
    if isinstance(target_num, int):
        return target_num
    elif isinstance(target_num, float):
        assert total_num is not None, '`total_num` must be given when `target_num` is float'
        return int(total_num * target_num)

def is_constraint_satisfied(constraint, obj):
    '''check if the constraint is satisfied by the obj
    Args:
        constraint: a list of constraints
    '''
    if isinstance(constraint, (list, tuple)):
        assert len(constraint) == 2, '`constraint` must be a list of length 2, indicating left and right ranges'
        return constraint[0] <= obj <= constraint[1]
    else:
        return constraint >= obj


class EvolutionMutator(RandomMutator):
    def __init__(
        self,
        model,
        eval_func: callable, # evaluation function of arch performance
        eval_kwargs: dict,   # kwargs of eval_func
        eval_metrics_order: dict, # order of eval metrics, e.g. {'accuracy': 'max', 'f1-score': 'max'}
        warmup_epochs: int=0,
        evolution_epochs: int=100,
        population_num: int=50,
        selection_alg: str='best',
        selection_num: int=0.8,
        crossover_num: Optional[Union[int, float]]=0.5,
        crossover_prob: float=0.3,
        mutation_num: Optional[Union[int, float]]=0.5,
        mutation_prob: float=0.3,
        flops_limit: Optional[Union[list, float]]=None, # MFLOPs, None means no limit
        size_limit: Optional[Union[list, float]]=None, # MB, None means no limit
        log_dir: str='evolution_logs',
        topk: Optional[Union[int, float]]=10,
        resume_from_checkpoint: Optional[str]=None,
        to_save_checkpoint: bool=True,
        to_plot_pareto: bool=True,
        figname: str='evolution_pareto.pdf',
        *args, **kwargs
    ):
        '''Init Args:
            model: model (Supernet) to be searched

            ####Evaluation function####
            eval_func: evaluation function of arch performance
            eval_kwargs: kwargs of eval_func, must contain arguments of `model` and `mutator`
            eval_metrics_order: order of eval metrics, e.g. {'accuracy': 'max', 'f1-score': 'max'}

            ####Evolution parameters####
            warmup_epochs: number of warmup epochs
            evolution_epochs: number of evolution epochs
            selection_alg: 'best' or 'nsga2'
                - best: select the best candidate
                - nsga2: select the best candidate according to NSGA-II
            selection_num: select num for each epoch
            population_num: population num for each epoch
            crossover_num: crossover num for each epoch
            crossover_prob: crossover probability
            mutation_num: mutation num for each epoch
            mutation_prob: mutation probability
            flops_limit: flops limit for each epoch
            size_limit: size limit for each epoch

            ####Logging parameters####
            log_dir: log directory
            topk: top k candidates
            resume_from_checkpoint: resume from checkpoint
            to_save_checkpoint: save checkpoint or not
            to_plot_pareto: plot pareto or not
            figname: pareto figure name
        '''
        super(EvolutionMutator, self).__init__(model)
        if selection_alg=='best' and len(eval_metrics_order)>1:
            raise ValueError('`selection_alg` must be `nsga2` when there are more than one eval metrics')
            
        self.eval_func = eval_func
        self.eval_kwargs = eval_kwargs
        self.eval_metrics_order = eval_metrics_order
        self.warmup_epochs = warmup_epochs
        self.evolution_epochs = evolution_epochs
        self.selection_alg = selection_alg 
        self.selection_num = get_int_num(selection_num, population_num)
        self.population_num = population_num
        self.crossover_num = get_int_num(crossover_num, population_num)
        self.crossover_prob = crossover_prob
        self.mutation_num = get_int_num(mutation_num, population_num)
        self.mutation_prob = mutation_prob
        self.random_num = population_num - (self.crossover_num + self.mutation_num)
        self.flops_limit = flops_limit
        self.size_limit = size_limit
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')
        self.topk = get_int_num(topk, population_num)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.to_save_checkpoint = to_save_checkpoint
        self.to_plot_pareto = to_plot_pareto
        self.figname = figname

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.selection_num: [], self.topk: []}
        self.epoch = 0
        self.candidates = []

    def save_checkpoint(self, epoch=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        if epoch is not None:
            ckpt_name = f"{self.log_dir}/checkpoint_{epoch}.pth.tar"
        else:
            ckpt_name = f"{self.log_dir}/{self.checkpoint_name}"
        torch.save(info, ckpt_name)
        log.info(f'save checkpoint to {ckpt_name}')

    def load_checkpoint(self):
        if not os.path.exists(self.resume_from_checkpoint):
            return False
        info = torch.load(self.resume_from_checkpoint)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        log.info(f'load checkpoint from {self.resume_from_checkpoint}')
        return True

    def search(self):
        log.info(f'''
        Evolution Mutator start searching for {self.evolution_epochs} epochs with parameters:
            - population_num: {self.population_num}
            - selection_num: {self.selection_num}
            - corssover_num: {self.crossover_num}
            - mutation_num: {self.mutation_num}
            - random_num: {self.random_num}
            - selection_alg: {self.selection_alg}
            - crossover_prob: {self.crossover_prob}
            - mutation_prob: {self.mutation_prob}
        ''')
        if self.resume_from_checkpoint:
            self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.evolution_epochs:
            log.info('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            # selection step
            self.update_top_k(self.candidates, k=self.selection_num)
            # update top k individuals globally
            self.update_top_k(self.candidates, k=self.topk)

            log.info(f'evolution epoch {self.epoch}/{self.evolution_epochs}: top {self.topk} results:')
            for i, cand in enumerate(self.keep_top_k[self.topk]):
                encoding = cand["encoding"]
                performance = cand['performance']
                size = cand['size']
                log.info(f'No.{i + 1} {encoding} performance={performance} size={size} MB')

            mutation = self.get_mutation(
                self.selection_num, self.mutation_num, self.mutation_prob)
            crossover = self.get_crossover(self.selection_num, self.crossover_num, self.crossover_prob)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            if self.to_save_checkpoint:
                self.save_checkpoint(self.epoch)
        if self.to_plot_pareto:
            name_objx_list = []
            if self.flops_limit is not None:
                name_objx_list.append('flops')
            if self.size_limit is not None:
                name_objx_list.append('size')
            for name_objx in name_objx_list:
                objx = np.array([cand[name_objx] for cand in self.vis_dict.values() if 'is_filtered' not in cand])
                indices = np.argsort(objx)
                objx = objx[indices]

                objys = []
                for name_objy in self.eval_metrics_order:
                    objy = np.array([cand['performance'][name_objy] for cand in self.vis_dict.values() if 'is_filtered' not in cand])
                    objy = objy[indices]

                    # method1: pareto
                    # pareto_keys = [cand['encoding'] for cand in self.keep_top_k[self.topk]]
                    # pareto_indices = [i for i, encoding in enumerate(self.vis_dict.keys()) if encoding in pareto_keys]
                    # pareto_indices = np.array(pareto_indices)
                    # resort_by_size_indices = objx[pareto_indices].argsort()
                    # pareto_indices = pareto_indices[resort_by_size_indices]

                    # method2: pareto
                    pareto_lists = NonDominatedSorting( np.vstack( (1/objy, objx) ) )
                    pareto_indices = pareto_lists[0]
                    log.info(f"Information of pareto models:")
                    for i, idx in enumerate(pareto_indices):
                        log.info(f"{i} {name_objx}:{objx[idx]} {name_objy}:{objy[idx]}")

                    self.plot_pareto_fronts(
                        objx, objy, pareto_indices, name_objx, name_objy, figname=f'pareto_{name_objx}_{name_objy}.pdf'
                    )

    def get_random(self, num):
        log.info('random select ........')
        while len(self.candidates) < num:
            arch = self.sample_search() # type: dict
            if not self.is_legal(arch):
                continue
            self.candidates.append(self.get_cand_by_arch(arch))
            log.debug('random {}/{}'.format(len(self.candidates), num))
        log.debug('random_num = {}'.format(len(self.candidates)))

    def is_legal(self, arch: dict):
        '''
        Args:
            arch: dict, architecture, e.g.,
            >>> arch = {
                    'op1': [0, 0, 1, 0, 1],
                    'op2': [1, 0, 0]
                }
        '''
        self.sample_by_mask(arch)
        encoding = self.arch2encoding(arch)
        if encoding not in self.vis_dict:
            self.vis_dict[encoding] = {'encoding': encoding, 'arch': arch}
        info = self.vis_dict[encoding]
        if 'visited' in info or 'is_filtered' in info:
            return False

        if 'flops' not in info or 'size' not in info:
            flops_size = flops_size_counter(self.model, input_size=(2,3,64,64), convert=True)
            flops, size = flops_size['flops'], flops_size['size']
            info['flops'] = flops # MFLOPs
            info['size'] = size # MB

        if self.flops_limit is not None and is_constraint_satisfied(info['flops'], self.flops_limit):
            log.debug('flops limit exceed')
            info['is_filtered'] = True
            log.debug(f"{encoding} flops = {info['flops']} MFLOPs size={info['size']} MB")
            return False
        elif self.size_limit is not None and is_constraint_satisfied(info['size'], self.size_limit):
            log.debug('size limit exceed')
            info['is_filtered'] = True
            log.debug(f"{encoding} flops = {info['flops']} MFLOPs size={info['size']} MB")
            return False

        info['visited'] = True
        eval_kwargs = deepcopy(self.eval_kwargs)
        eval_kwargs.update({'network': self.model, 'mutator': self})
        info['performance'] = self.eval_func(**eval_kwargs)
        log.debug(f"{encoding} flops = {info['flops']} MFLOPs size={info['size']} MB perf={info['performance']}")
        return True

    def arch2encoding(self, arch: dict):
        '''
        arch: {
            'op1': [0, 0, 1, 0, 1],
            'op2': [1, 0, 0]
        }
        return "(op1:[0, 0, 1, 0, 1])-(op2:[1, 0, 0])"
        '''
        encoding = ''
        for key, value in arch.items():
            value = value.cpu().detach().int().numpy().tolist()
            encoding += '({}:{})-'.format(key, ''.join(map(str, value)))
        return encoding

    def get_cand_by_arch(self, arch: dict):
        '''return cand by arch
        Args:
            arch: {
                'op1': [0, 0, 1, 0, 1],
                'op2': [1, 0, 0]
            }
        return
            {
                'arch': arch, 'encoding': '(op1:00101)-(op2:100)', 'visited': True,
                'flops': ..., 'size': ..., 'performance': ...}
        '''
        encoding = self.arch2encoding(arch)
        if encoding in self.vis_dict:
            return self.vis_dict[encoding]
        elif self.is_legal(arch):
            return self.vis_dict[encoding]
        else:
            return None

    def update_top_k(self, candidates, k):
        assert k in self.keep_top_k
        log.info(f'update top-{k} models......')
        tmp = self.keep_top_k[k]
        tmp += candidates
        if self.selection_alg == 'best':
            cand_sample = tmp[0]
            performance = cand_sample['performance']
            metric_name, metric_order = next(iter(self.eval_metrics_order.items())) # e.g., accuracy, max
            key = lambda x: x['performance'][metric_name]
            reverse = True if metric_order == 'max' else False
            tmp.sort(key=key, reverse=reverse)
            self.keep_top_k[k] = tmp[:k]
        elif self.selection_alg == 'nsga2':
            targets = []
            constraints = []
            for cand in tmp:
                cand_targets = []
                for metric_name, metric_order in self.eval_metrics_order.items():
                    value = cand['performance'][metric_name]
                    if metric_order == 'max':
                        value = -1 * value # nsga2 is a minimization algorithm, the smaller the better
                    cand_targets.append(cand['performance'][metric_name])
                targets.append(cand_targets)
            targets = np.array(targets) # (num, num_metrics)
            if self.flops_limit is not None:
                constraints.append(np.array([cand['flops'] for cand in tmp]).reshape(-1, 1))
            if self.size_limit is not None:
                constraints.append(np.array([cand['size'] for cand in tmp]).reshape(-1, 1))
            constraints = np.concatenate(constraints, 1) # (num, num_objectives)
            fitnesses = np.hstack(tuple([targets, constraints]))
            indices = nsga2_select(fitnesses, k)
            self.keep_top_k[k] = [tmp[idx] for idx in indices]

    def get_mutation(self, k, mutation_num, mutation_prob):
        assert k in self.keep_top_k
        log.info('mutation ......')
        res = []
        max_iters = mutation_num * 10

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            index = np.random.randint(0, k)
            cand = self.keep_top_k[k][index]
            mutated_arch = self.mutation(cand['arch'], mutation_prob)
            if not self.is_legal(mutated_arch):
                continue
            encoding = self.arch2encoding(mutated_arch)
            res.append(self.vis_dict[encoding])
            log.debug('mutation {}/{}'.format(len(res), mutation_num))

        log.debug('mutation_num = {}'.format(len(res)))
        return res

    def mutation(self, arch, mutation_prob):
        '''Mutate a single network architecture'''
        new_arch = deepcopy(arch)
        for key in arch:
            length = len(arch[key])
            if length > 1 and np.random.rand() < mutation_prob:
                delete_index = arch[key].float().argmax()
                select_range = list(range(delete_index)) + list(range(delete_index+1, length))
                index = torch.tensor(np.random.choice(select_range))
                new_arch[key] = F.one_hot(index, num_classes=length).view(-1).bool()
        return new_arch

    def get_crossover(self, k, crossover_num, crossover_prob):
        assert k in self.keep_top_k
        log.info('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            index1, index2 = np.random.choice(k, 2, replace=False)
            cand1 = self.keep_top_k[k][index1]
            cand2 = self.keep_top_k[k][index2]
            corssover_arch = self.crossover(cand1['arch'], cand2['arch'], crossover_prob)
            if not self.is_legal(corssover_arch):
                continue
            encoding = self.arch2encoding(corssover_arch)
            res.append(self.vis_dict[encoding])
            log.debug('crossover {}/{}'.format(len(res), crossover_num))

        log.debug('crossover_num = {}'.format(len(res)))
        return res

    def crossover(self, arch1, arch2, crossover_prob):
        '''Crossover step
        Generate a new arch by randomly crossover part of architectures of two parent individuals
        '''
        cross_arch = deepcopy(arch1)
        for key in arch1:
            if np.random.rand() < crossover_prob:
                cross_arch[key] = deepcopy(arch2[key])
        return cross_arch

    @classmethod
    def plot_real_proxy_metrics(
        cls,
        real_metrics: list,
        proxy_metrics: list,
        name_objx: str='predicted performance',
        name_objy: str='real performance',
        figsize=(8,5),
        figname=None
    ):
        '''plot real and proxy metrics
        Args:
            real_metrics: list of real metrics
            proxy_metrics: list of proxy metrics
            name_objx: name of x axis
            name_objy: name of y axis
            figsize: figure size
            figname: figure name
        '''
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        plt.clf()  # 清空当前figure下所有axes中的内容
        fig, ax = plt.subplots(figsize=figsize)
        data = pd.DataFrame({name_objx: proxy_metrics, name_objy: real_metrics})
        sns.regplot(x=name_objx,y=name_objy,data=data)
        if figname is None:
            figname = 'real_proxy_metrics.pdf'
        if not os.path.isdir(figname):
            figname = os.path.join(self.log_dir, figname)
        os.makedirs(self.log_dir, exist_ok=True)
        fig.savefig(figname)
        plt.close(fig)

    def plot_pareto_fronts(
        self,
        objx: np.array,
        objy: np.array,
        pareto_indices: np.array,
        name_objx: str='objx',
        name_objy: str='objy',
        figsize=(8,5),
        figname=None
    ):
        import matplotlib.pyplot as plt
        if not isinstance(objx, np.ndarray):
            objx = np.array(objx)
        if not isinstance(objy, np.ndarray):
            objy = np.array(objy)
        objx_pareto = [x for x in objx[pareto_indices]]
        objy_pareto = [x for x in objy[pareto_indices]]
        fig = plt.figure(figsize=figsize)
        plt.clf()  # 清空当前figure下所有axes中的内容
        ax1 = fig.add_subplot(111)
        ax1.scatter(objx, objy)
        ax1.plot(objx_pareto, objy_pareto)
        ax1.set_xlabel(name_objx)
        ax1.set_ylabel(name_objy)
        plt.show()
        if figname is None:
            figname = self.figname
        if not os.path.isdir(figname):
            figname = os.path.join(self.log_dir, figname)
        os.makedirs(self.log_dir, exist_ok=True)
        fig.savefig(figname)
        plt.close(fig)


if __name__ == '__main__':
    from hyperbox.networks.nasbench201.nasbench201 import NASBench201Network
    from hyperbox.networks.nasbench_mbnet.network import NASBenchMBNet


    def eval_func(mutator, network, da=32,gs=5432,gsrh=764):
        return {
            'acc': np.random.rand(), 'loss': np.random.rand()
            }

    selection_alg_list = [
        'nsga2', 
        # 'best'
    ]
    eval_metrics_order_list = [
        # {'acc': 'max'}, 
        {'acc': 'max', 'loss': 'min'}]

    for alg in selection_alg_list:
        for order in eval_metrics_order_list:
            print(f'alg = {alg}, order = {order}')
            net = NASBench201Network(num_classes=10).cuda()
            # net = NASBenchMBNet(num_classes=10).cuda()
        
            em = EvolutionMutator(
                net,
                eval_func=eval_func,
                eval_kwargs={'da':352,'gs':32,'gsrh':764},
                eval_metrics_order=order,
                warmup_epochs=0,
                evolution_epochs=1,
                population_num=10,
                selection_alg=alg,
                selection_num=0.8,
                crossover_num=0.2,
                crossover_prob=0.3,
                mutation_num=0.2,
                mutation_prob=0.3,
                flops_limit=330, # MFLOPs
                size_limit=50, # MB
                log_dir='evolution_logs',
                topk=10,
                to_save_checkpoint=True,
                to_plot_pareto=True,
                figname='evolution_pareto.pdf'
            )


            em.search()
        