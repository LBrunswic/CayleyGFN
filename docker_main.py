import argparse
parser = argparse.ArgumentParser(
                    prog='CayleyGFN',
                    description='Launch a unit experiment',
)
parser.add_argument(
    '--hp_file',
    type=str,
    default='Hyperparameters/experimental_settings/base_test.hp',
    help='Provide the experimental setting file, If not provided, defaults to base_test.hp'
)

parser.add_argument(
    '--pool_size',
    type=int,
    default=32,
    help='Set the experiment pool size, the effect of results is small ans is inly intended to improve efficiency',
)
parser.add_argument(
    '--memory',
    type=int,
    default=0,
    help='Non positive value activate memory growth. Positive values set the actual GPU memory limit in MB'
)

parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help='Not implemented yet'
)
parser.add_argument(
    '--save',
    type=str,
    default='test.csv',
    help='Not implemented yet'
)

args = parser.parse_args().__dict__

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    if args['memory'] >0:
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args['memory'])])
    elif args['memory'] == 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    elif args['memory'] == -1:
        pass

import pickle
import hashlib

import os, sys
from datetime import datetime
from time import time
from train import train_test_model
from logger_config import get_logger, log_dict

with open(args['hp_file'],'br') as f:
    experiments_hparams = pickle.load(f)
    hash = hashlib.sha256(bytes(str(experiments_hparams), 'utf8')).hexdigest()+ ('_%s'%args['pool_size'])
    experiments_hparams.update(
        {'pool_size':args['pool_size']}
    )

experiments_hparams['logdir'] = 'LOGS'
logger = get_logger(
    name=hash,
    filename=os.path.join(experiments_hparams['logdir'], hash + '.log'),
    filemode='w'
)
T = time()
result, returns, flow = train_test_model(experiments_hparams, logger)
data_save = os.path.join(
    'RESULTS',
    args['save']
)
result.to_csv(data_save)
dt = (time()-T)/3600
logger.info(f"{experiments_hparams['N_SAMPLE']} experiments done in { (time()-T)/60} minutes. So {experiments_hparams['N_SAMPLE']/dt} experiments per hours.")