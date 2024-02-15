import logging
import os

def set_logger(folder):
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(folder, 'training.log'), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def train_session():
    """
        - Environnement
        - Architecture
        - Bootstrap (load)
        - Training HP
        - Log settings
    """