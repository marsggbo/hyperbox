import os
from typing import Any, Dict, List, Optional, Union

import hydra
import torch
from torch.nn import Module
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.lite import LightningLite

from hyperbox.models.base_model import BaseModel
from hyperbox.utils.utils import DotDict, hparams_wrapper


@hparams_wrapper
class HyperboxLite(LightningLite):
    def __init__(
        self,
        # LightningModule
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        # DataModule
        datamodule_cfg: Optional[Union[DictConfig, dict]] = None,
        # LightningLite
        lite_cfg: Optional[Union[DictConfig, dict]] = None,
        # logger
        logger_cfg: Optional[Union[DictConfig, dict]] = None,
        # Run config
        run_cfg: Optional[Union[DictConfig, dict]] = None,
        **kwargs
    ) -> None:
        # Init LightningLite
        self.lite_cfg = self.parse_lite_cfg(lite_cfg)
        cfg = dict(self.lite_cfg)
        LightningLite.__init__(self, **cfg)

        path = os.getcwd()
        log_file = os.path.join(path, 'log.txt')
        logger.add(log_file)

        self.pl_model = BaseModel(network_cfg, mutator_cfg, optimizer_cfg,
            loss_cfg, metric_cfg, scheduler_cfg)

        # Init DataModule
        self._datamodule = self.build_datamodule(datamodule_cfg)
        logger.info('Building datamodule ...')

        # Init logger
        if logger_cfg is not None:
            self._logger = hydra.utils.instantiate(logger_cfg)

        self.run_cfg = run_cfg
        self.kwargs = DotDict(kwargs)

    def parse_lite_cfg(self, lite_cfg=None) -> None:
        '''
        Args:
            accelerator: Optional[Union[str, "Accelerator"]] = None, # gpu, cpu, ipu, tpu
            strategy: Optional[Union[str, "Strategy"]] = None, # "dp", "ddp", "ddp_spawn", "tpu_spawn", "deepspeed", "ddp_sharded", or "ddp_sharded_spawn"
            devices: Optional[Union[List[int], str, int]] = None,
            num_nodes: int = 1,
            precision: Union[int, str] = 32,
            plugins: Optional[Union["PLUGIN_INPUT", List["PLUGIN_INPUT"]]] = None,
            gpus: Optional[Union[List[int], str, int]] = None,
            tpu_cores: Optional[Union[List[int], str, int]] = None,
        '''
        default_lite_cfg = {
            'accelerator': None,
            'strategy': None,
            'devices': None,
            'num_nodes': 1,
            'precision': 32,
            'plugins': None,
            'gpus': None,
            'tpu_cores': None,
        }
        default_lite_cfg = DotDict(default_lite_cfg)
        for key, value in lite_cfg.items():
            default_lite_cfg[key] = value
        return default_lite_cfg

    def build_datamodule(self, datamodule_cfg: DictConfig) -> Any:
        self.datamodule_cfg = datamodule_cfg
        if datamodule_cfg is not None:
            datamodule = hydra.utils.instantiate(datamodule_cfg)
        else:
            datamodule = None
        datamodule.setup()
        return datamodule

    @property
    def datamodule(self):
        return self._datamodule

    @datamodule.setter
    def datamodule(self, value):
        self._datamodule = value

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def dataloaders(self) -> Dict[str, Any]:
        return {
            'train': self.train_dataloader,
            'val': self.val_dataloader,
            'test': self.test_dataloader
        }

    @property
    def train_dataloader(self) -> Any:
        if getattr(self, '_train_dataloader', None) is not None:
            return self._train_dataloader
        elif self.datamodule is not None and hasattr(self.datamodule, 'train_dataloader'):
            return self.datamodule.train_dataloader()
        else:
            return None

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self._train_dataloader = loader

    @property
    def val_dataloader(self) -> Any:
        if getattr(self, '_val_dataloader', None) is not None:
            return self._val_dataloader
        elif self.datamodule is not None and hasattr(self.datamodule, 'val_dataloader'):
            return self.datamodule.val_dataloader()
        else:
            return None

    @val_dataloader.setter
    def val_dataloader(self, loader):
        self.val_dataloader = loader

    @property
    def test_dataloader(self) -> Any:
        if getattr(self, '_test_dataloader', None) is not None:
            return self._test_dataloader
        elif self.datamodule is not None and hasattr(self.datamodule, 'test_dataloader'):
            return self.datamodule.test_dataloader()
        else:
            return None

    @test_dataloader.setter
    def test_dataloader(self, loader):
        self.test_dataloader = loader
