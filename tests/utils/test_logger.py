from hyperbox.utils.logger import get_logger

if __name__ == '__main__':
    logger = get_logger(__name__)
    logger.info('This is info')
    logger.debug('This is debug')
    logger.warning('This is warning')
    logger.error('This is error')