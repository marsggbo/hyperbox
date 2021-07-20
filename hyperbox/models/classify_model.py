from typing import Any, List, Optional, Union

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from hyperbox.utils.logger import get_logger
from .base_model import BaseModel

logger = get_logger(__name__)


class ClassifyModel(BaseModel):
    """Random NAS Model Template
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
        r'''Random NAS model
        Args:
            network [DictConfig, dict, torch.nn.Module]: 
            mutator [DictConfig, dict, BaseMutator]: 
            optimizer [DictConfig, dict, torch.optim.Optimizer]: 
            loss Optional[DictConfig, dict, Callable]: loss function or DictConfig of loss function
            metric: metric function, such as Accuracy, Precision, etc.
        '''
        super().__init__(network_cfg, None, optimizer_cfg,
                         loss_cfg, metric_cfg, scheduler_cfg, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.network(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        self.network.train()
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_metric(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def training_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        logger.info(f"[rank {self.rank}] Val epoch{self.current_epoch} final result: loss={loss}, acc={acc}")
   
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        acc = np.mean([output['acc'].item() for output in outputs])
        loss = np.mean([output['loss'].item() for output in outputs])
        logger.info(f"Test epoch{self.current_epoch} final result: loss={loss}, acc={acc}")

    def on_fit_start(self):
        mflops, size = self.arch_size((1,3,32,32), convert=True)
        logger.info(f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")

    def on_fit_end(self):
        mflops, size = self.arch_size((1,3,32,32), convert=True)
        logger.info(f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        