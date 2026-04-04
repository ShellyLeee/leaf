import logging
import os


def get_logger(output_dir: str, name: str = 'leaf_mlp'):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(output_dir, 'train.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
