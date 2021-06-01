from typing import Any, List, Optional, Union

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from utils.logger import get_logger
logger = get_logger(__name__)

def instantiate(*args, **kwargs):
    return hydra.utils.instantiate(*args, **kwargs)

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
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute, such as
        # self.hparams.network_cfg
        # self.hparams.mutator_cfg
        self.save_hyperparameters()
        self.build_network(self.hparams.network_cfg)
        self.build_mutator(self.hparams.mutator_cfg)
        self.build_loss(self.hparams.loss_cfg)
        self.build_metric(self.hparams.metric_cfg)

    def build_network(self, cfg):
        # build network
        assert cfg is not None, 'Please specify the config for network'
        if isinstance(cfg, (DictConfig, dict)):
            cfg = DictConfig(cfg)
            self.network = instantiate(cfg)
            logger.info(f'Building {cfg._target_} ...')

    def build_mutator(self, cfg):
        # build mutator
        if isinstance(cfg, (DictConfig, dict)):
            cfg = DictConfig(cfg)
            self.mutator = instantiate(cfg, model=self.network)
            logger.info(f'Building {cfg._target_} ...')
        elif cfg is None:
            from mutator.random_mutator import RandomMutator
            logger.info('Mutator cfg is not specified, so use RandomMutator as the default.')
            self.mutator = RandomMutator(self.network)
        self.mutator.reset()

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

    # def training_step(self, batch: Any, batch_idx: int):
    # def training_epoch_end(self, outputs: List[Any]):
    # def validation_step(self, batch: Any, batch_idx: int):
    # def validation_epoch_end(self, outputs: List[Any]):
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
        return optim
