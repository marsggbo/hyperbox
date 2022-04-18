from typing import List, Optional
import os

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
    log.info(f'pid {os.getpid()}')
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
        model = utils.load_pretrained_weights(config, model, config.get('pretrained_weight'))

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

    result = None
    if config.get("only_test"):
        # Only test the model
        if config.get('pretrained_weight') is None:
            log.info('No petrained weight provided')
        result = trainer.test(model=model, datamodule=datamodule)
        print(result)
    elif config.get('engine') is not None and len(config.get('engine')) > 0:
        # customized engine
        engine = hydra.utils.instantiate(config.engine, 
            trainer=trainer, model=model, datamodule=datamodule, cfg=config, _recursive_=False)
        log.info(f"Running customized engine: {engine.__class__}")
        result = engine.run()
    else:
        # Train the model
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
        result = trainer.callback_metrics

        # Evaluate model on test set, using the best model achieved during training
        if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
            log.info("Starting testing!")
            ckpt_path = config.trainer.get('ckpt_path') or "best"
            try:
                trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            except:
                trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
            result.update(trainer.callback_metrics)

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
    log.info(f"result: {result}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and result is not None:
        log.info(f"Best score:\n{optimized_metric}={result[optimized_metric]}")
        return result[optimized_metric]
