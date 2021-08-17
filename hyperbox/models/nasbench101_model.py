'''
Based on Random Model
'''

from .base_model import BaseModel
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


class NASBench101Model(BaseModel):
    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        is_sync: bool = True,
        is_net_parallel: bool = True,
        **kwargs

    ):
        super().__init__(network_cfg, None, optimizer_cfg,
                         loss_cfg, metric_cfg, scheduler_cfg, **kwargs)
        self.is_sync = is_sync
        self.is_net_parallel = is_net_parallel

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def forward(self, x):
        return self.network(x)

    def parse_batch(self, batch):
        batch = [batch[0]['data'], batch[0]['label'].long().view(-1)]
        return batch

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        # training process
        self.network.train()
        self.mutator.eval()  # ??
        if batch_idx % 5 == 0:
            self.sample_search()
        loss, preds, targets = self.step(batch)

        # metrics
        acc = self.train_metric(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        return super().validation_epoch_end(outputs)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        return super().test_epoch_end(outputs)

    def configure_callbacks(self):
        pass
