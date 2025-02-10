import os
import sys
import time
import logging


def get_logger(log_dir, name, log_filename=None, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = r'{}.log'.format(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory:', log_dir)

    return logger
