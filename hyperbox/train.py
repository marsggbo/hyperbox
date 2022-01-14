from typing import List, Optional

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import open_dict
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from hyperbox.utils import utils, logger

log = logger.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    with open_dict(config.model):
        config.model.datamodule_cfg = config.datamodule
    model: LightningModule = hydra.utils.instantiate(config.model, _recursive_=False)
    if config.get('pretrained_weight'):
        # loading pretrained weight to network and mutator
        import torch
        from hydra.utils import to_absolute_path
        ckpt_path = to_absolute_path(config.get("pretrained_weight"))
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'epoch' in ckpt:
            ckpt = ckpt['state_dict']
        try:
            # load state_dict of network, mutator, and etc,.
            # model.load_state_dict(ckpt)
            model = model.load_from_checkpoint(ckpt_path, **config.model)
            del ckpt
            log.info(f"Loading pretrained weight from {ckpt_path}, including network, mutator")
        except Exception as e:
            try:
                # only load network weight
                model.network.load_state_dict(ckpt)
                log.info(f"Loading pretrained network weight from {ckpt_path}")
            except Exception as e:
                try:
                    # load subnet weight from a supernet weight
                    from hyperbox.networks.utils import extract_net_from_ckpt
                    weight_supernet = extract_net_from_ckpt(ckpt_path)
                    model.network.load_from_supernet(weight_supernet)
                    log.info(f"Loading subnet weight from supernet weight: {ckpt_path}")
                except Exception as e:
                    raise Exception(f'failed to load pretrained weight from {ckpt_path}.\n{e}')

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    if config.get("only_test"):
        if config.get('pretrained_weight') is None:
            log.info('No petrained weight provided')
        result = trainer.test(model=model, datamodule=datamodule)
        print(result)
    else:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            ckpt_path = config.trainer.get('ckpt_path') or "best"
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        log.info(f"Best score:\n{optimized_metric}")
        return trainer.callback_metrics[optimized_metric]
