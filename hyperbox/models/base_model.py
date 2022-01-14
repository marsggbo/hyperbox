from typing import Any, List, Optional, Union, Tuple

import os
import json
import random
import hydra
import json 
import torch
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import TorchTensorEncoder, hparams_wrapper
from hyperbox.utils.calc_model_size import flops_size_counter
from hyperbox.networks.utils import set_running_statistics

logger = get_logger(__name__)


def instantiate(*args, **kwargs):
    return hydra.utils.instantiate(*args, **kwargs)


@hparams_wrapper
class BaseModel(LightningModule):
    """NAS Model Template
        Example of LightningModule for MNIST classification.

        A LightningModule organizes your PyTorch code into 5 sections:
            - Computations (init).
            - Train loop (training_step)
            - Validation loop (validation_step)
            - Test loop (test_step)
            - Optimizers (configure_optimizers)

        Read the docs:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        **kwargs
    ):
        r'''NAS model template
        Args:
            network [DictConfig, dict, torch.nn.Module]: 
            mutator [DictConfig, dict, BaseMutator]: 
            optimizer [DictConfig, dict, torch.optim.Optimizer]: 
            loss Optional[DictConfig, dict, Callable]: loss function or DictConfig of loss function
            metric: metric function, such as Accuracy, Precision, etc.
        '''
        super(BaseModel, self).__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute, such as
        # self.hparams.network_cfg
        # self.hparams.mutator_cfg
        # self.save_hyperparameters() # same as @hparams_wrapper
        self.build_network(self.hparams.network_cfg)
        self.build_mutator(self.hparams.mutator_cfg)
        self.build_loss(self.hparams.loss_cfg)
        self.build_metric(self.hparams.metric_cfg)
        self.reset_seed_flag = 2
        self.history = {} # save architectures performance

    def build_network(self, cfg):
        # build network
        assert cfg is not isinstance(cfg, (DictConfig, dict)),\
            'Please specify the config for network'
        cfg = DictConfig(cfg)
        self.network = instantiate(cfg)
        logger.info(f'Building {cfg._target_} ...')

    def build_mutator(self, cfg, model=None):
        if model is None:
            model = self.network
        # build mutator
        if isinstance(cfg, (DictConfig, dict)):
            cfg = DictConfig(cfg)
            self.mutator = instantiate(cfg, model=model)
            logger.info(f'Building {cfg._target_} ...')
        elif cfg is None:
            from hyperbox.mutator.random_mutator import RandomMutator
            logger.info('Mutator cfg is not specified, so use RandomMutator as the default.')
            self.mutator = RandomMutator(model=model)

    def build_loss(self, cfg):
        # build loss function
        if isinstance(cfg, (DictConfig,dict)):
            cfg = DictConfig(cfg)
            self.criterion = hydra.utils.instantiate(cfg)
            logger.info(f'Building {cfg._target_} ...')
        elif cfg is None:
            logger.info('Loss cfg is not specified, so use CrossEntropyLoss as the default.')
            self.criterion = torch.nn.CrossEntropyLoss()

    def build_metric(self, cfg):
        # build metric function
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        if isinstance(cfg, (DictConfig,dict)):
            cfg = DictConfig(cfg)
            self.metric = instantiate(cfg)
            logger.info(f'Building {cfg._target_} ...')
        elif cfg is None:
            logger.info('Metric cfg is not specified, so use Accuracy as the default.')
            self.metric = Accuracy()

        self.train_metric = self.metric
        self.val_metric = self.metric
        self.test_metric = self.metric

    @property
    def is_network_search(self):
        '''
        if a network is initialized by a non-None mask, then it is a network to search
        '''
        return getattr(self.network, 'mask', None) is None

    # def on_train_epoch_start(self):
    # def training_step(self, batch: Any, batch_idx: int):
    # def training_epoch_end(self, outputs: List[Any]):
    # def on_validation_epoch_start(self):
    #     if self.is_network_search:
    #         if not self.mutator._cache:
    #             self.mutator.reset()
    #         self.reset_running_statistics(subset_size=64, subset_batch_size=32)
    # def validation_step(self, batch: Any, batch_idx: int):
    # def validation_epoch_end(self, outputs: List[Any]):
    # def on_test_epoch_start(self):
    #     if self.is_network_search:
    #         self.reset_running_statistics(subset_size=64, subset_batch_size=32)
    # def test_step(self, batch: Any, batch_idx: int):
    # def test_epoch_end(self, outputs: List[Any]):

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_cfg = DictConfig(self.hparams.optimizer_cfg)
        optim = instantiate(optimizer_cfg, params=self.network.parameters())

        if self.hparams.scheduler_cfg is not None:
            scheduler_cfg = DictConfig(self.hparams.scheduler_cfg)
            scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optim, _recursive_=False)
            return [optim], [scheduler]
        return optim

    @property
    def rank(self):
        return self.global_rank

    @property
    def arch(self):
        if hasattr(self.network, 'arch'):
            return self.network.arch
        return self.network.__class__.__name__

    def arch_size(self, datasize: Tuple=None, convert=True, verbose=False):
        if hasattr(self.network, 'arch_size'):
            return self.network.arch_size(datasize, convert, verbose)
        size = None
        for candidate_size in [
            datasize,
            self.example_input_array,
            self.train_dataloader().dataset[0][0].shape]:
            if candidate_size is not None:
                size = candidate_size
                break
        assert size is not None, \
            "Please specify valid data size, e.g., size=self.arch_size(datasize=(1,3,32,32))"
        result = flops_size_counter(self.network, size, convert, verbose)
        mflops, mb_size = list(result.values())
        return mflops, mb_size   

    def export(self, file: str, save_history: bool=False, metric: dict=None, is_parse: bool=True):
        """Call ``mutator.export()`` and dump the architecture to ``file``.
        Args:
            file : str
                A file path. Expected to be a JSON.
            save_history: bool
                save history performance of the current arch
        """
        if self.rank!=0:
             return
        if is_parse:
            mutator_export = self.mutator.export() # export parsed arch
        else:
            mutator_export = self.mutator._cache # export current arch
        filename = os.path.basename(file)
        cwd = os.getcwd()
        filepath = os.path.join(cwd, 'mask_json')
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        file = os.path.join(filepath, filename)
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=4, sort_keys=True, cls=TorchTensorEncoder)
        if save_history:
            history_file = os.path.join(cwd, 'history.json')
            self.history[filename] = {
                'filepath': file,
                'metric': metric if metric is not None else {}
            }
            with open(history_file, 'w') as f:
                json.dump(self.history, f)

    def reset_seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(f"[rank {self.rank}] reset seed to {seed}")

    def sample_search(self, is_sync: bool = False, is_net_parallel: bool = False):
        '''
        Args:
            is_sync: bool
                - True: sync mode
                - False: async mode
            is_net_parallel: bool
                - True: sample different models
                - False: sample the same model

                      |  same arch  |  different arch |
            sync mode |    broadcast from rank 0      |
           async mode |  same seed  |  different seed |
        '''
        if not is_sync:
            # async mode
            # sample same archs by default
            if is_net_parallel and self.trainer.world_size>1 and self.reset_seed_flag > 0:
                # sample different archs by set different seed. once is enough
                self.reset_seed(self.rank+666)
                if self.training: self.reset_seed_flag -= 1
            self.mutator.reset()
        else:
            # sync mode
            if self.trainer.world_size <= 1:
                self.mutator.reset()
            else:
                # broadcast mask from rank 0
                mask = None
                if not is_net_parallel:
                    # broadcast the same arch
                    self.mutator.reset()
                    mask = self.mutator._cache
                    mask = self.trainer.accelerator.broadcast(mask, src=0)
                else:
                    # broadcast different archs
                    mask_dict = dict()
                    if self.rank==0:
                        for idx in range(self.trainer.world_size):
                            self.mutator.reset()
                            mask = self.mutator._cache
                            mask_dict[idx] = mask
                    mask_dict = self.trainer.accelerator.broadcast(mask_dict, src=0)
                    mask = mask_dict[self.rank]

                for m in self.mutator.mutables:
                    m.mask.data = mask[m.key].data.to(self.device)
                    if m.key in self.mutator._cache:
                        self.mutator._cache[m.key].data = mask[m.key].data.to(self.device)
                    else:
                        self.mutator._cache[m.key] = mask[m.key].to(self.device)

    @property
    def datamodule(self):
        return self.trainer.datamodule

    def reset_running_statistics(self, net=None, subset_size=160, subset_batch_size=32, dataloader=None):
        if net is None:
            net = self.network
        if dataloader is None:
            try:
                if getattr(self.datamodule, 'is_customized', False):
                    dataset = self.datamodule.train_dataloader()['train'].dataset
                else:
                    dataset = self.datamodule.train_dataloader().dataset
            except:
                dataset = self.datamodule.test_dataloader().dataset
            size = min(subset_size, len(dataset))
            indices = np.random.choice(np.arange(len(dataset)), size=size, replace=False)
            sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            subset_batch_size = min(subset_batch_size, len(dataset))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=subset_batch_size,
                num_workers=4, sampler=sub_sampler)
        try:
            set_running_statistics(net, dataloader, self.mutator)
        except Exception as e:
            logger.info(e)
            raise Exception('''
            Failed to set running statistics, this is probably because 
            1) the ``affine`` attribute of the BatchNorm is not set to True
            2) you use `dp` training strategy 
            and the data of `BatchNorm` is not copied correctly in `DataParallel` mode due to PyTorch issue.
            [1] https://github.com/pytorch/pytorch/issues/1051
            [2] https://github.com/pytorch/pytorch/issues/36035
            ''')
