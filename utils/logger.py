# encoding: utf-8
"""
@author:  Jiajun Fu
@contact: Jaakk0F@foxmail.com
"""

import logging
import os
import sys

import colorlog
import tqdm


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    # console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(
        colorlog.ColoredFormatter(
            # '%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s',
            '%(message)s',
            datefmt='%Y-%d-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'brown',
                'INFO': 'cyan',
                'SUCCESS:': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'
            },
        ))
    logger.addHandler(ch)

    # file handler
    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        print(save_dir, "log.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            colorlog.ColoredFormatter(
                # '%(log_color)s%(name)s %(asctime)s %(levelname)s | %(message)s',
                '%(message)s',
                datefmt='%Y-%d-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'brown',
                    'INFO': 'cyan',
                    'SUCCESS:': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white'
                },
            ))
        logger.addHandler(fh)

    # tqdm hander
    # for now we don't use this
    # th = TqdmLoggingHandler(logging.DEBUG)
    # th.setFormatter(formatter)
    # logger.addHandler(th)
    return logger


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)
