from typing import Any, List, Optional, Union

import hydra
import torch
import numpy as np
import medmnist
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from kornia.augmentation import RandomMixUp

from hyperbox.utils.logger import get_logger
from hyperbox.models.base_model import BaseModel
from hyperbox_app.medmnist.utils import getAUC
from hyperbox_app.medmnist.losses import MixupLoss, MutualLoss

logger = get_logger(__name__)


class FinetuneModel(BaseModel):
    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        use_mixup: bool = False,
        input_size= (2,3,28,28,28),
        **kwargs
    ):
        r'''Finetune model
        Args:
            network [DictConfig, dict, torch.nn.Module]: 
            mutator [DictConfig, dict, BaseMutator]: 
            optimizer [DictConfig, dict, torch.optim.Optimizer]: 
            loss Optional[DictConfig, dict, Callable]: loss function or DictConfig of loss function
            metric: metric function, such as Accuracy, Precision, etc.
        '''
        if kwargs.get('datamodule_cfg', None) is not None:
            self.datamodule_cfg = kwargs.get('datamodule_cfg')
            info = medmnist.INFO[self.datamodule_cfg.data_flag]
            self.c_in = info['n_channels']
            self.num_classes = len(info['label'])
            if network_cfg.get('c_in'):
                network_cfg['c_in'] = self.c_in
            if network_cfg.get('num_classes'):
                network_cfg['num_classes'] = self.num_classes
        super().__init__(network_cfg, None, optimizer_cfg,
                         loss_cfg, metric_cfg, scheduler_cfg, **kwargs)
        self.use_mixup = use_mixup
        self.random_mixup = RandomMixUp()
        self.input_size = input_size

    def on_fit_start(self):
        self.dataset_info = self.trainer.datamodule.data_train.info
        self.task = self.dataset_info['task']
        if self.task == 'multi-label, binary-class':
            self.criterion = nn.BCEWithLogitsLoss()
        if self.use_mixup:
            self.criterion = MixupLoss(self.criterion)
        if getattr(self.trainer.datamodule.data_train, 'as_rgb', None):
            n_channel = 3
        else:
            n_channel = self.dataset_info['n_channels']
        # if '3d' in self.datamodule_cfg.data_flag:
        #     self.input_size = (2,n_channel,28,28,28)
        # else:
        #     self.input_size = (2,n_channel,28,28)
        mflops, size = self.arch_size(self.input_size, convert=True)
        logger.info(f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")

    def forward(self, x: torch.Tensor):
        return self.network(x)
        # return self.network(x, self.to_aug)

    def step(self, batch: Any):
        x, y = batch
        if self.use_mixup and self.criterion.training:
            if len(x.shape) == 5 and x.shape[1]==1:
                x = x.squeeze(1)
                is_squeeze = True
            x, y = self.random_mixup(x, y)
            if is_squeeze:
                x = x.unsqueeze(1)
        logits = self.forward(x)
        if len(logits.shape) == 3:
            loss = 0.
            for logit in logits:
                loss += self.criterion(logit, y)
            logits = logits.mean(0)
        else:
            loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def on_train_epoch_start(self):
        self.y_true_trn = torch.tensor([]).to(self.device)

    def training_step(self, batch: Any, batch_idx: int):
        self.to_aug = False
        self.network.train()
        trn_X, trn_y = batch
        self.y_true_trn = torch.cat((self.y_true_trn, trn_y), 0)
        loss, preds, targets = self.step(batch)
        if getattr(self.network, 'num_branches', None):
            loss_mutual = 0.
            num_branches = self.network.num_branches
            num_features = len(self.network.branches[0].features)
            # ensemble_features = []
            for idx in range(num_features):
                en_feat = torch.stack([self.network.branches[i].features[idx] for i in range(num_branches)]).mean(0)
                sub_loss_mutual = 0.
                for i in range(num_branches):
                    sub_loss = MutualLoss('kl')(self.network.branches[i].features[idx], en_feat)
                    sub_loss_mutual += sub_loss
                sub_loss_mutual /= num_branches
                loss_mutual += sub_loss_mutual
                # ensemble_features.append(en_feat)
            loss_mutual /= num_features
            if self.current_epoch < 6:
                loss = loss + 0.5 * loss_mutual
            else:
                loss = loss + 1000 * loss_mutual

        # log train metrics
        acc = self.train_metric(preds, trn_y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        if batch_idx % 50 == 0:
            logger.info(
                f"Train epoch{self.current_epoch} batch{batch_idx}: loss={loss} (mutual={loss_mutual} ce{loss-loss_mutual}), acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def training_epoch_end(self, outputs: List[Any]):
        self.y_true_trn = self.y_true_trn.detach().cpu().numpy()
        print('Train class 0', sum(self.y_true_trn==0), 'class 1', sum(self.y_true_trn==1), 'all', len(self.y_true_trn))
        acc_epoch = self.trainer.callback_metrics['train/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['train/loss_epoch'].item()
        logger.info(f'Train epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

    def on_validation_epoch_start(self):
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        if self.use_mixup:
            self.criterion.training = False

    def validation_step(self, batch: Any, batch_idx: int):
        self.network.train()
        self.to_aug = False
        with torch.no_grad():
            x, targets = batch
            preds = self.forward(x)
            if len(preds.shape) == 3:
                loss = 0.
                for logit in preds:
                    loss += self.criterion(logit, targets)
                preds = preds.mean(0)
            else:
                loss = self.criterion(preds, targets)
            # loss, preds, targets = self.step(batch)
        self.y_true = torch.cat((self.y_true, targets), 0)
        self.y_score = torch.cat((self.y_score, preds.softmax(dim=-1)), 0)
        try:
            auc = getAUC(self.y_true, self.y_score, self.task)
            self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        except:
            auc = 0

        # log val metrics
        preds = torch.argmax(preds, dim=1)
        acc = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        self.y_true = self.y_true.detach().cpu().numpy()
        self.y_score = self.y_score.detach().cpu().numpy()
        print('class 0', sum(self.y_true==0), 'class 1', sum(self.y_true==1), 'all', len(self.y_true))
        try:
            auc = getAUC(self.y_true, self.y_score, self.task)
            self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=False)
        except:
            auc = 0
        acc_epoch = self.trainer.callback_metrics['val/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['val/loss_epoch'].item()
        logger.info(f'Val epoch{self.trainer.current_epoch} auc={auc:.4f} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

    def on_test_epoch_start(self):
        self.y_true = torch.tensor([]).to(self.device)
        self.y_score = torch.tensor([]).to(self.device)
        if self.use_mixup:
            self.criterion.training = False
        self.dataset_info = self.trainer.datamodule.data_test.info
        self.task = self.dataset_info['task']

    def test_step(self, batch: Any, batch_idx: int):
        self.network.train()
        self.to_aug = False
        with torch.no_grad():
            x, targets = batch
            preds = self.forward(x)
            if len(preds.shape) == 3:
                loss = 0.
                for logit in preds:
                    loss += self.criterion(logit, targets)
                preds = preds.mean(0)
            else:
                loss = self.criterion(preds, targets)
            # loss, preds, targets = self.step(batch)
        self.y_true = torch.cat((self.y_true, targets), 0)
        self.y_score = torch.cat((self.y_score, preds.softmax(dim=-1)), 0)

        # log test metrics
        preds = torch.argmax(preds, dim=1)
        acc = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.y_true = self.y_true.detach().cpu().numpy()
        self.y_score = self.y_score.detach().cpu().numpy()
        auc = getAUC(self.y_true, self.y_score, self.task)
        acc = self.trainer.callback_metrics['test/acc'].item()
        loss = self.trainer.callback_metrics['test/loss'].item()
        logger.info(f'Test epoch{self.trainer.current_epoch} auc={auc:.4f} acc={acc:.4f} loss={loss:.4f}')

    def on_fit_end(self):
        mflops, size = self.arch_size(self.input_size, convert=True)
        logger.info(f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        