from typing import Any, List, Optional, Union

import hydra
import torch
from torchmetrics import Accuracy

from hyperbox.lites.base_lite import HyperboxLite
from hyperbox.utils.logger import get_logger
from hyperbox.utils.utils import DotDict


class TrainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.args.log_interval = 10
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.dataloader_iter = None
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)

    def advance(self, epoch) -> None:
        batch_idx, (data, target) = next(self.dataloader_iter)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        self.lite.backward(loss)
        self.optimizer.step()

        if (batch_idx == 0) or ((batch_idx + 1) % self.args.log_interval == 0):
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(self.dataloader),
                    len(self.dataloader.dataset),
                    100.0 * batch_idx / len(self.dataloader),
                    loss.item(),
                )
            )

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        self.scheduler.step()
        self.dataloader_iter = None


class TestLoop(Loop):
    def __init__(self, lite, args, model, dataloader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.args.log_interval = 10
        self.model = model
        self.dataloader = dataloader
        self.dataloader_iter = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy().to(lite.device)
        self.test_loss = 0

    @property
    def done(self) -> bool:
        return False

    def reset(self):
        self.dataloader_iter = enumerate(self.dataloader)
        self.test_loss = 0
        self.accuracy.reset()

    def advance(self) -> None:
        _, (data, target) = next(self.dataloader_iter)
        output = self.model(data)
        self.test_loss += self.criterion(output, target)
        self.accuracy(output, target)

        if self.args.dry_run:
            raise StopIteration

    def on_run_end(self):
        test_loss = self.lite.all_gather(self.test_loss).sum() / len(self.dataloader.dataset)

        if self.lite.is_global_zero:
            logger.info(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({self.accuracy.compute():.0f}%)\n")


class MainLoop(Loop):
    def __init__(self, lite, args, model, optimizer, scheduler, train_loader, test_loader):
        super().__init__()
        self.lite = lite
        self.args = args
        self.epoch = 0
        self.train_loop = TrainLoop(self.lite, self.args, model, optimizer, scheduler, train_loader)
        self.test_loop = TestLoop(self.lite, self.args, model, test_loader)

    @property
    def done(self) -> bool:
        return self.epoch >= self.args.epochs

    def reset(self):
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self.train_loop.run(self.epoch)
        self.test_loop.run()

        if self.args.dry_run:
            raise StopIteration

        self.epoch += 1


class ClassiftLite(HyperboxLite):
    def run(self, hparams: Optional[Union[DotDict, dict]]) -> None:
        if hparams is not None and isintsance(hparams, dict):
            hparams = DotDict(hparams)

        #1. setup_dataloaders
        if self.datamodule is None:
            # manual dataloaders
            train_loader, test_loader = self.setup_dataloaders(self.train_loader, self.test_loader)
        else:
            train_loader, test_loader = self.train_loader, self.test_loader

        #2. setup_network
        model = self.network
        optimizer = self.optimizers()

        model, optimizer = self.setup(model, optimizer)
        scheduler_cfg = DotDict(self.hparams.scheduler_cfg)
        scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer, _recursive_=False)

        MainLoop(self, hparams, model, optimizer, scheduler, train_loader, test_loader).run()

        if hparams.save_model and self.is_global_zero:
            self.save(model.state_dict(), "mnist_cnn.pt")
