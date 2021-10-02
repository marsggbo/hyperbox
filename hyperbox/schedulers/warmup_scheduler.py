'''
source code: https://github.com/ildoonet/pytorch-gradual-warmup-lr
'''
import hydra
from omegaconf import DictConfig

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier: float=0.9, warmup_epoch: int=10, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.warmup_epoch = warmup_epoch
        if isinstance(after_scheduler, (dict, DictConfig)):
            after_scheduler_cfg = after_scheduler
            after_scheduler = hydra.utils.instantiate(after_scheduler_cfg, optimizer=optimizer)
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch+1) / self.warmup_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.warmup_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.warmup_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.warmup_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == '__main__':
    import torch
    from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR, CosineAnnealingLR
    from torch.optim.sgd import SGD

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    total_epoch = 30
    warmup_epoch = 10
    scheduler_steplr = StepLR(optim, step_size=2, gamma=0.9)
    scheduler_steplr = CosineAnnealingLR(optim, T_max=total_epoch-warmup_epoch, eta_min=1e-6)
    # scheduler_steplr = None
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, warmup_epoch=warmup_epoch, after_scheduler=scheduler_steplr)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, total_epoch):
        scheduler_warmup.step(epoch)
        print(epoch, optim.param_groups[0]['lr'])

        optim.step()    # backward pass (update network)