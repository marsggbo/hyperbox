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


class NASBench101(BaseModel):
    def __init__(

    ):
        super().__init__()

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def parse_batch(self, batch):
        batch = [batch[0]['data'], batch[0]['label'].long().view(-1)]
        return batch

    def training_step(self, batch, batch_idx):
        # training process

        pass

    def training_epoch_end(self, outputs):
        # end of training epoch
        pass

    def validation_step(self, batch, batch_idx):

        pass

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        return super().validation_epoch_end(outputs)

    def test_step(self, *args, **kwargs):
        return super().test_step(*args, **kwargs)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        return super().test_epoch_end(outputs)

    def configure_callbacks(self):
        pass
