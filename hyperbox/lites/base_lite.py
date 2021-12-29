from typing import List, Optional, Union, Any

import torch
from omegaconf import DictConfig
from pytorch_lightning.lite import LightningLite
from pytorch_lightning.loops import Loop

from hyperbox.models.base_model import BaseModel
from hyperbox.utils.logger import get_logger


class HyperboxLite(BaseModel, LightningLite):
    def __init__(
        self,
        # LightningLite
        accelerator: Optional[Union[str, "Accelerator"]] = None,
        strategy: Optional[Union[str, "Strategy"]] = None,
        devices: Optional[Union[List[int], str, int]] = None,
        num_nodes: int = 1,
        precision: Union[int, str] = 32,
        plugins: Optional[Union["PLUGIN_INPUT", List["PLUGIN_INPUT"]]] = None,
        gpus: Optional[Union[List[int], str, int]] = None,
        tpu_cores: Optional[Union[List[int], str, int]] = None,

        # LightningModule
        network_cfg: Optional[Union[DictConfig, dict]] = None,
        mutator_cfg: Optional[Union[DictConfig, dict]] = None,
        optimizer_cfg: Optional[Union[DictConfig, dict]] = None,
        loss_cfg: Optional[Union[DictConfig, dict]] = None,
        metric_cfg: Optional[Union[DictConfig, dict]] = None,
        scheduler_cfg: Optional[Union[DictConfig, dict]] = None,
        datamodule_cfg: Optional[Union[DictConfig, dict]] = None,
        run_cfg: Optional[Union[DictConfig, dict]] = None,
        **kwargs
    ) -> None:
        BaseModel.__init__(self, network_cfg, mutator_cfg, optimizer_cfg,
            loss_cfg, metric_cfg, scheduler_cfg, **kwargs)
        self.other_args = kwargs
        self.run_cfg = run_cfg
        LightningLite.__init__(self, accelerator, strategy, devices, num_nodes,
            precision, plugins, gpus, tpu_cores)
        logger.info(f'Building HyperboxLite:')
        self.build_datamodule(datamodule_cfg)

    def build_datamodule(self, datamodule_cfg: DictConfig) -> Any:
        self.datamodule_cfg = datamodule_cfg
        if datamodule_cfg is not None:
            self.datamodule: LightningDataModule = hydra.utils.instantiate(datamodule_cfg)
        else:
            self.datamodule = None

    def setup_dataloaders(self, datamodule) -> Any:
        if datamodule is not None:
            logger.info('1. setup_dataloaders')
            dataloaders = {key: d for key, d in self.dataloaders.items() if d is not None}
            keys = list(dataloaders.keys())
            loaders = self.setup_dataloaders(*list(dataloaders.values()))
            if len(loaders) == 3:
                self._train_dataloader, self._val_dataloader, self._test_dataloader = loaders
            elif len(loaders) == 2:
                self._train_dataloader, self._test_dataloader = loaders
        else:
            logger.info('1. setup_dataloaders: no dataloaders found')
            self._train_dataloader, self._val_dataloader, self._test_dataloader = None, None, None

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
            return self.datamodule.train_dataloader
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
            return self.datamodule.val_dataloader
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
            return self.datamodule.test_dataloader
        else:
            return None

    @test_dataloader.setter
    def test_dataloader(self, loader):
        self.test_dataloader = loader

    def run(self, *args, **kwargs) -> Any:
        ...
        ''' 
        #1. setup_dataloaders
        if self.datamodule is None:
            # manual dataloaders
            train_loader, test_loader = self.setup_dataloaders(self.train_loader, self.test_loader)
        else:
            train_loader, test_loader = self.train_loader, self.test_loader

        #2. setup_network
        model = self.network
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        model, optimizer = self.setup(model, optimizer)
        scheduler = StepLR(optimizer, step_size=1, gamma=hparams.gamma)

        MainLoop(self, model, optimizer, scheduler, train_loader, test_loader, hparams).run()

        if hparams.save_model and self.is_global_zero:
            self.save(model.state_dict(), "mnist_cnn.pt")
        '''


if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("/home/xihe/xinhe/hyperbox/logs/runs/bnnas/bnnas_c10_all_adam0.001_sync_hete/2021-10-06_06-31-00/.hydra/config.yaml")
    cfg_model = cfg.model
    lite = HyperboxLite(**cfg_model)
