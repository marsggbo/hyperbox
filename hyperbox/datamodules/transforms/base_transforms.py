
from omegaconf import DictConfig

from hyperbox.utils.utils import hparams_wrapper
from hyperbox.utils.logger import get_logger
_logger = get_logger(__name__)


@hparams_wrapper
class BaseTransforms(object):
    def __init__(self, *args, **kwargs):
        for key, value in self.hparams.items():
            if isinstance(value, dict):
                value = DictConfig(value)
            setattr(self, key, value)

    def get_transform(self, is_train:bool = True):
        if not is_train: 
            _logger.info('Generating validation transform ...')
            transform = self.valid_transform
            _logger.info(f'Valid transform={transform}')
        else:
            _logger.info('Generating training transform ...')
            transform = self.train_transform
            _logger.info(f'Train transform={transform}')
        return transform

    @property
    def transform_valid(self):
        raise NotImplementedError

    @property
    def transform_train(self):
        raise NotImplementedError