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
from hyperbox.utils.average_meter import AverageMeter, AverageMeterGroup
logger = get_logger(__name__, logging.INFO, rank_zero=True)

from .base_model import BaseModel


class SampleSearch(Callback):

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        start = time.time()
        with torch.no_grad():
            pl_module.sample_search()
        duration = time.time() - start
        pl_module.time_records.update({'sample_search': duration})
        logger.debug(f"[rank {pl_module.rank}] batch idx={batch_idx} sample search {duration} seconds")

    # def on_validation_epoch_start(self, trainer, pl_module):
    #     pl_module.sample_search()


class OFAModel(BaseModel):
    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        is_net_parallel: bool = True,
        is_sync: bool = False,
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
        self.automatic_optimization = False
        self.kd_subnets_method = kd_subnets_method
        self.aux_weight = aux_weight
        self.num_valid_archs = num_valid_archs
        self.is_net_parallel = is_net_parallel
        self.is_sync = is_sync
        self.arch_perf_history = {}
        self.reset_seed_flag = 1 # initial value should >= 1
        if not hasattr(self, 'time_records'):
            self.time_records = AverageMeterGroup()

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def training_step(self, batch: Any, batch_idx: int):
        self.optimizer = self.optimizers()
        self.optimizer.zero_grad()

        start_whole = time.time()
        self.network.train()
        self.mutator.eval()

        inputs, targets = batch
        start = time.time()
        output = self.network(inputs)
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

        # torch.cuda.synchronize()
        duration = time.time() - start
        self.time_records.update({'forward': duration})

        start = time.time()
        self.manual_backward(loss)
        # torch.cuda.synchronize()
        duration = time.time() - start
        self.time_records.update({'backward': duration})

        # optimizer.step
        start = time.time()
        self.optimizer.step()
        # torch.cuda.synchronize()
        duration = time.time() - start
        self.time_records.update({'step': duration})

        # log train metrics
        start = time.time()
        # preds = torch.argmax(output, dim=1)
        preds = torch.softmax(output, -1)
        acc = self.train_metric(preds, targets)
        duration = time.time() - start
        self.time_records.update({'acc': duration})
        # logger.debug(f"rank{self.rank} loss={loss} acc={acc}")
        start = time.time()
        # sync_dist = not self.is_net_parallel # sync the metrics if all processes train the same sub network
        sync_dist = False
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=sync_dist, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, sync_dist=sync_dist, prog_bar=False)
        duration = time.time() - start
        self.time_records.update({'log': duration})
        if batch_idx % 50 ==0:
            logger.info(f"[rank {self.rank}] Train epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        duration_whole = time.time() - start_whole
        self.time_records.update({'whole_forward': duration_whole})
        # logger.debug(f"[rank {self.rank}] batch idx={batch_idx} whole forward {duration_whole} seconds")
        logger.debug(f"[rank {self.rank}] batch idx={batch_idx} time records={self.time_records}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def training_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        mflops, size = self.arch_size((1,3,32,32), convert=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        # logger.info(f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        logger.info(f"[rank {self.rank}] Train epoch{self.current_epoch} final result: loss={loss}, acc={acc}")

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
        # preds = torch.argmax(output, dim=1)
        preds = torch.softmax(output, -1)
        acc = self.val_metric(preds, targets)
        # if self.arch not in self.arch_perf_history:
        #     self.arch_perf_history[self.arch] = [acc]
        # else:
        #     self.arch_perf_history[self.arch].append(acc)
        # sync_dist = not self.is_net_parallel # sync the metrics if all processes train the same sub network
        # self.log(f"val/loss", loss, on_step=False, on_epoch=True, sync_dist=sync_dist, prog_bar=True)
        # self.log(f"val/acc", acc, on_step=False, on_epoch=True, sync_dist=sync_dist, prog_bar=True)
        # if batch_idx % 10 == 0:
        #     logger.info(f"Val epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        mflops, size = self.arch_size((1,3,32,32), convert=True)
        sync_dist = not self.is_net_parallel # sync the metrics if all processes train the same sub network
        self.log(f"val/loss", loss, on_step=False, on_epoch=True, sync_dist=sync_dist, prog_bar=True)
        self.log(f"val/acc", acc, on_step=False, on_epoch=True, sync_dist=sync_dist, prog_bar=True)
        # logger.info(f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        # logger.info(f"[rank {self.rank}] Val epoch{self.current_epoch} final result: loss={loss}, acc={acc}")

    def test_step(self, batch: Any, batch_idx: int):
        loss_avg = 0.
        acc_avg = 0.
        inputs, targets = batch
        for i in range(self.num_valid_archs):
            self.mutator.reset()
            output = self.network(inputs)
            loss = self.criterion(output, targets)

            # log test metrics
            # preds = torch.argmax(output, dim=1)
            preds = torch.softmax(output, -1)
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