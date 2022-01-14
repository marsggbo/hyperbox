from typing import Any, List, Optional, Union

import copy
import random
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from hyperbox.utils.logger import get_logger
from hyperbox.models.base_model import BaseModel

logger = get_logger(__name__, rank_zero=True)


class DARTSModel(BaseModel):
    def __init__(
        self,
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        arc_lr: float = 3e-4,
        unrolled: bool = False,
        aux_weight: float = 0.4,
        is_sync: bool = True,
        is_net_parallel: bool = False,
        **kwargs
    ):
        super().__init__(network_cfg, mutator_cfg, optimizer_cfg,
                         loss_cfg, metric_cfg, scheduler_cfg, **kwargs)
        self.arc_lr = arc_lr
        self.unrolled = unrolled
        self.aux_weight = aux_weight
        self.automatic_optimization = False
        self.is_net_parallel = is_net_parallel
        self.is_sync = is_sync
        # self.reset_seed_flag = 2 # initial value should >= 1

    def sample_search(self):
        super().sample_search(self.is_sync, self.is_net_parallel)

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        # debug info
        # self.trainer.accelerator.barrier()
        # print(f"[rank {self.rank}] seed={np.random.get_state()[1][0]}")
        # print(f"[rank {self.rank}] epoch-{self.current_epoch} batch-{batch_idx} arch={self.network.arch}")
        (trn_X, trn_y) = batch['train']
        (val_X, val_y) = batch['val']
        self.weight_optim, self.ctrl_optim = self.optimizers()

        # phase 1. architecture step
        self.ctrl_optim.zero_grad()
        if self.unrolled:
            self._unrolled_backward(trn_X, trn_y, val_X, val_y)
        else:
            self._backward(val_X, val_y)
        self.ctrl_optim.step()

        # phase 2: child network step
        self.weight_optim.zero_grad()
        with torch.no_grad():
            self.sample_search()
        preds, loss = self._logits_and_loss(trn_X, trn_y)
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.network.parameters(), 5.)  # gradient clipping
        self.weight_optim.step()

        # log train metrics
        preds = torch.argmax(preds, dim=1)
        acc = self.train_metric(preds, trn_y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        if batch_idx % 50 == 0:
            logger.info(
                f"Train epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": trn_y, 'acc': acc}

    def _logits_and_loss(self, X, y):
        output = self.network(X)
        if isinstance(output, tuple):
            output, aux_output = output
            aux_loss = self.criterion(aux_output, y)
        else:
            aux_loss = 0.
        loss = self.criterion(output, y)
        loss = loss + self.aux_weight * aux_loss
        # self._write_graph_status()
        return output, loss

    def training_epoch_end(self, outputs: List[Any]):
        acc_epoch = self.trainer.callback_metrics['train/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['train/loss_epoch'].item()
        logger.info(f'Train epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

    def validation_step(self, batch: Any, batch_idx: int):
        (X, targets) = batch
        preds, loss = self._logits_and_loss(X, targets)

        # log val metrics
        acc = self.val_metric(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        if batch_idx % 10 == 0:
        # if True:
            logger.info(f"Val epoch{self.current_epoch} batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def validation_epoch_end(self, outputs: List[Any]):
        acc_epoch = self.trainer.callback_metrics['val/acc_epoch'].item()
        loss_epoch = self.trainer.callback_metrics['val/loss_epoch'].item()
        logger.info(f'Val epoch{self.trainer.current_epoch} acc={acc_epoch:.4f} loss={loss_epoch:.4f}')

        mflops, size = self.arch_size((2, 3, 32, 32), convert=True)
        logger.info(
            f"[rank {self.rank}] current model({self.arch}): {mflops:.4f} MFLOPs, {size:.4f} MB.")
        logger.info(f"self.mutator._cache: {len(self.mutator._cache)} choices")
        for key, value in self.mutator._cache.items():
            logger.info(f"{key}: {value.detach()}")

        if self.current_epoch % 10 == 0:
            self.export("mask_epoch_%d.json" % self.current_epoch,
            True, {'val_acc': acc_epoch, 'val_loss': loss_epoch})

    def on_test_epoch_start(self):
        self.reset_running_statistics(subset_size=128, subset_batch_size=32)

    def test_step(self, batch: Any, batch_idx: int):
        (X, targets) = batch
        preds, loss = self._logits_and_loss(X, targets)

        # log test metrics
        acc = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        if batch_idx % 10 == 0:
            logger.info(f"Test batch{batch_idx}: loss={loss}, acc={acc}")
        return {"loss": loss, "preds": preds, "targets": targets, 'acc': acc}

    def test_epoch_end(self, outputs: List[Any]):
        acc = self.trainer.callback_metrics['test/acc'].item()
        loss = self.trainer.callback_metrics['test/loss'].item()
        logger.info(f'Test epoch{self.trainer.current_epoch} acc={acc:.4f} loss={loss:.4f}')

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_cfg = DictConfig(self.hparams.optimizer_cfg)
        weight_optim = hydra.utils.instantiate(optimizer_cfg, params=self.network.parameters())
        ctrl_optim = torch.optim.Adam(
            self.mutator.parameters(), self.arc_lr, betas=(0.5, 0.999), weight_decay=1.0E-3)
        return weight_optim, ctrl_optim

    def _backward(self, val_X, val_y):
        """
        Simple backward with gradient descent
        """
        self.sample_search()
        _, loss = self._logits_and_loss(val_X, val_y)
        self.manual_backward(loss)

    def _unrolled_backward(self, trn_X, trn_y, val_X, val_y):
        """
        Compute unrolled loss and backward its gradients
        """
        backup_params = copy.deepcopy(tuple(self.network.parameters()))

        # do virtual step on training data
        lr = self.optimizer.param_groups[0]["lr"]
        momentum = self.optimizer.param_groups[0]["momentum"]
        weight_decay = self.optimizer.param_groups[0]["weight_decay"]
        self._compute_virtual_model(trn_X, trn_y, lr, momentum, weight_decay)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        self.sample_search()
        _, loss = self._logits_and_loss(val_X, val_y)
        w_model, w_ctrl = tuple(self.network.parameters()), tuple(self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_ctrl)
        d_model, d_ctrl = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, trn_X, trn_y)
        with torch.no_grad():
            for param, d, h in zip(w_ctrl, d_ctrl, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, X, y, lr, momentum, weight_decay):
        """
        Compute unrolled weights w`
        """
        # don't need zero_grad, using autograd to calculate gradients
        self.sample_search()
        _, loss = self._logits_and_loss(X, y)
        gradients = torch.autograd.grad(loss, self.network.parameters())
        with torch.no_grad():
            for w, g in zip(self.network.parameters(), gradients):
                m = self.optimizer.state[w].get("momentum_buffer", 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.network.parameters(), backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, trn_X, trn_y):
        """
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            self.logger.warning(
                "In computing hessian, norm is smaller than 1E-8, cause eps to be %.6f.", norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.network.parameters(), dw):
                    p += e * d

            self.sample_search()
            _, loss = self._logits_and_loss(trn_X, trn_y)
            dalphas.append(torch.autograd.grad(loss, self.mutator.parameters()))

        dalpha_pos, dalpha_neg = dalphas  # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
