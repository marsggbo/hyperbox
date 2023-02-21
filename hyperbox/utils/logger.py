import os
import sys
import logging
from loguru import logger


def custom_format(record):
    path = os.path.abspath(record["file"].path)
    record["extra"]["abspath"] = path
    fmt = "<cyan>[{time:YYYY-MM-DD HH:mm:ss}]</cyan> <green>[{level}]</green> <red>[{extra[abspath]}:{line} ({name})]</red> {message}\n{exception}"
    return fmt


LOGGER_DICT = {}

def get_logger(name=None, level=logging.INFO, is_rank_zero=True, log2file=False):
    if is_rank_zero or name is None:
        name = 'exp'
    if name in LOGGER_DICT:
        return LOGGER_DICT[name]
    fmt = custom_format
    logger.remove()
    kwargs = {'sink': sys.stderr, 'format': fmt, 'level': level, 'colorize': True, 'backtrace': True}
    handlers = [kwargs]
    logger.opt(exception=True)
    if log2file:
        snd_kwargs = {k: v for k, v in kwargs.items() if k != 'sink'}
        snd_kwargs['sink'] = os.path.join(os.getcwd(), name + '.log')
        handlers.append(snd_kwargs)
    config = {"handlers": handlers}
    logger.configure(**config)
    LOGGER_DICT[name] = logger
    return logger


if __name__ == '__main__':
    log1 = get_logger(None, level=logging.DEBUG, log2file=False)
    log1.debug('test') # showing
    log2 = get_logger('test2', level=logging.INFO, is_rank_zero=False, log2file=False)
    log2.debug('test2') # not showing
    log1.debug('test') # not showing
    print(log1 is log2) # True