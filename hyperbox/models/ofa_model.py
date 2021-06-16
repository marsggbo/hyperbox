from typing import Any, List, Optional, Union

import time

import copy
import random
import logging
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from hyperbox.losses.kd_loss import KDLoss
from hyperbox.utils.logger import get_logger
logger = get_logger(__name__, logging.INFO, rank_zero=True)

from .base_model import BaseModel


class SampleSearch(Callback):

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.sample_search()


class OFAModel(BaseModel):
    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        is_net_parallel: bool = True,
        num_valid_archs: int = 5,
        kd_subnets_method: str = '',
        aux_weight: float = 0.4,
        **kwargs
    ):
        '''
        Args:
            is_net_parallel (bool):
                - True: each process train different networks
                - False: same as the normal distributed parallel, all processes train the same network
            kd_subnets_method (str): the method of knowledge distillation (kd) on subnets.
                - '': disable kd subnets
                - 'mutual': subnets kd each other mutually
                - 'ensemble': using subnets' ensemble as the teacher to kd all subnets
        '''
        super().__init__(network_cfg, mutator_cfg, optimizer_cfg,
                         loss_cfg, metric_cfg, **kwargs)
        self.kd_subnets_method = kd_subnets_method
        self.aux_weight = aux_weight
        self.num_valid_archs = num_valid_archs
        self.is_net_parallel = is_net_parallel
        self.arch_perf_history = {}

    def sample_search(self):
        with torch.no_grad():
            if self.trainer.world_size <= 1:
                self.mutator.reset()
            else:
                logger.debug(f"process {self.rank} model is on device: {self.device}")
                mask_dict = dict()
                is_reset = False
                for idx in range(self.trainer.world_size):
                    if self.is_net_parallel or not is_reset:
                        self.mutator.reset()
                        is_reset = True
                    mask = self.mutator._cache
                    mask_dict[idx] = mask
                mask_dict = self.trainer.accelerator.broadcast(mask_dict, src=0)
                mask = mask_dict[self.rank]
                logger.debug(f"[net_parallel={self.is_net_parallel}]{self.rank}: {mask['ValueChoice18']}")
                for m in self.mutator.mutables:
                    m.mask.data = mask[m.key].data.to(self.device)

    def training_step(self, batch: Any, batch_idx: int):
        self.network.train()
        self.mutator.eval()
        start = time.time()
        self.sample_search()
        duration = time.time() - start
        logger.info(f"[rank {self.trainer.global_rank}] batch idx={batch_idx} sample search {duration} seconds")

        logger.debug(f"rank{self.rank} model.fc={self.network.fc}")
        mflops, size = self.arch_size((1,3,32,32), convert=True)
        logger.info(f"current model info: {mflops:.4f} MFLOPs, {size:.4f} MB.")
        inputs, targets = batch
        start = time.time()
        output = self.network(inputs)
        duration = time.time() - start
        logger.info(f"[rank {self.trainer.global_rank}] batch idx={batch_idx} forward {duration} seconds")
        if isinstance(output, tuple):
            output, aux_output = output
            aux_loss = self.loss(aux_output, targets)
        else:
            aux_loss = 0.

        loss = self.criterion(output, targets)
        loss = loss + self.aux_weight * aux_loss
        if self.kd_subnets_method and self.trainer.world_size > 1:
            outputs_list = self.all_gather(output)
            if isinstance(output, list):
                # horovod
                outputs_list = torch.cat(outputs_list, 0).mean()
            loss_kd_subnets = self.calc_loss_kd_subnets(output, outputs_list, self.kd_subnets_method)
            loss = loss + 0.6 * loss_kd_subnets

        # log train metrics
        preds = torch.argmax(output, dim=1)
        acc = self.train_metric(preds, targets)
        logger.debug(f"rank{self.rank} loss={loss} acc={acc}")
        sync_dist = not self.is_net_parallel # sync the metrics if all processes train the same sub network
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=sync_dist, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, sync_dist=sync_dist, prog_bar=False)
        if batch_idx % 10 ==0:
            logger.info(f"Train epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def training_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        logger.info(f"Train epoch{self.current_epoch} final result: loss={loss}, acc={acc}")

    def calc_loss_kd_subnets(self, output, outputs_list, kd_method='ensemble'):
        loss = 0
        if kd_method == 'ensemble':
            ensemble_output = torch.stack(
                [x.detach().to(self.device) for x in outputs_list], 0).mean(0).to(self.device)
            alpha = 1
            temperature = 4
            loss = KDLoss(output, ensemble_output, alpha, temperature)
        elif kd_method == 'mutual':
            pass
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss_avg = 0.
        acc_avg = 0.
        inputs, targets = batch
        output = self.network(inputs)
        loss = self.criterion(output, targets)

        # log val metrics
        preds = torch.argmax(output, dim=1)
        acc = self.val_metric(preds, targets)
        if acc not in self.arch_perf_history:
            self.arch_perf_history[self.arch] = [acc]
        else:
            self.arch_perf_history[self.arch].append(acc)
        sync_dist = not self.is_net_parallel # sync the metrics if all processes train the same sub network
        self.log(f"val/loss", loss, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log(f"val/acc", acc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=True)
        # if batch_idx % 10 == 0:
        #     logger.info(f"Val epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        logger.info(f"Val epoch{self.current_epoch} final result: loss={loss}, acc={acc}")
    
    # def validation_epoch_end(self, outputs: List[Any]):
    def test_step(self, batch: Any, batch_idx: int):
        loss_avg = 0.
        acc_avg = 0.
        inputs, targets = batch
        for i in range(self.num_valid_archs):
            self.mutator.reset()
            output = self.network(inputs)
            loss = self.criterion(output, targets)

            # log test metrics
            preds = torch.argmax(output, dim=1)
            acc = self.test_metric(preds, targets)
            loss_avg += loss
            acc_avg += acc
            # self.log(f"test/{self.arch}_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            # self.log(f"test/{self.arch}_acc", acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
            # logger.info(f"Valid: arch={self.arch} loss={loss}, acc={acc}")
        loss_avg = self.all_gather(loss_avg).mean()
        acc_avg = self.all_gather(acc_avg).mean()
        self.log("test/loss", loss_avg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc_avg, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx % 10 == 0:
            logger.info(f"Test batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        logger.info(f"Test epoch{self.current_epoch} final result: loss={loss}, acc={acc}")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_cfg = DictConfig(self.hparams.optimizer_cfg)
        optim = hydra.utils.instantiate(optimizer_cfg, params=self.network.parameters())
        return optim

    def configure_callbacks(self):
        sample_search_callback = SampleSearch()
        return [
            sample_search_callback
        ]

    @property
    def arch(self):
        return self.network.arch