import os
import sys
from loguru import logger


def custom_format(record):
    path = os.path.abspath(record["file"].path)
    record["extra"]["abspath"] = path
    fmt = "<cyan>[{time:YYYY-MM-DD HH:mm:ss}]</cyan> <green>[{level}]</green> <red>[{extra[abspath]}:{line} ({name})]</red> {message}\n{exception}"
    return fmt

def get_logger(name=None, level='info', is_rank_zero=True):
    if is_rank_zero or name is None:
        name = 'exp'
    fmt = custom_format
    logger.remove()
    config = {
        "handlers": [
            {"sink": sys.stderr, "format": fmt},
        ],
    }
    logger.configure(**config)
    logger.opt(exception=True)
    logger.add(
        os.path.join(os.getcwd(), name + '.log'),
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True
    )
    return logger

if __name__ == '__main__':
    log1 = get_logger(None)
    log1.info('test')
    log2 = get_logger('test2', is_rank_zero=False)
    log2.info('test2')
