import sys
import logging
sys.dont_write_bytecode = True

formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


def create_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# if __name__ == '__main__':
#     logger = create_logger('Test')
#     logger.info('This is just for test')
#     add_filehandler(logger, 'test.txt')
#     logger.info('This is after add handler')