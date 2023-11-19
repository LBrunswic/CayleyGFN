
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import os, sys
from datetime import datetime
from train import train_test_model
from logger_config import get_logger, log_dict





experiments_hparams = {
    'EXPERIMENT_ID':'1',

    'profile':True,
    'N_SAMPLE': 20,

    'seed': 9220,

    'graph_size': 15,
    'graph_generators': 'trans_cycle_a',
    'inverse': True,
    'initial_pos': 'SymmetricUniform',
    'rew_fn': 'TwistedManhattan',
    'rew_param': {'width': 1, 'scale': -100, 'exp': False, 'mini': 0.0},
    'reg_proj': 'OrthReg',
    'reg_fn_logmin': 5.0,
    'grad_batch_size': 1,
    'batch_size': 1024,
    'length_cutoff': 30,
    'initial_flow': 0.001,
    'learning_rate': 0.002,
    'epochs': 100,
    'step_per_epoch': 5,
    'B_beta': -1000.0,
    'path_redraw': 0,
    'neighborhood': 0,
    'flowestimator_opt': {
        'options': {'kernel_depth': 1, 'width': 32, 'final_activation': 'linear'},
        'kernel_options': {'kernel_initializer': 'Orthogonal', 'activation': 'tanh', 'implementation': 1}},
    'embedding': ('cos', 'sin', 'natural'),
    'optimizer': 'Adam',
    'loss_base': 'Apower',
    'loss_alpha': 2.0,
    'rew_factor': 1.0,
    'heuristic_fn': 'R_zero',
    'heuristic_param': {},
    'heuristic_factor': 0.0,
    'reward_rescale': 'Trivial',
    'reg_fn_gen': 'norm2',
    'loss_cutoff': 'none',
    'lr_schedule': 'none',
    'reg_fn_alpha_schedule': 'none',
    'normalization_fn': 0,
    'normalization_nu_fn': 0,
    'group_dtype': 'float32',
    'reg_fn_alpha': (5, 10),
    'pool_size': 10,
}


experiments_hparams['logdir'] = 'LOGS'
logger = get_logger(
    name=experiments_hparams['EXPERIMENT_ID'],
    filename=os.path.join(experiments_hparams['logdir'], experiments_hparams['EXPERIMENT_ID'] + '.log'),
    filemode='w'
)
result, returns, flow = train_test_model(experiments_hparams, logger)
data_save = os.path.join(
    'RESULTS',
    experiments_hparams['EXPERIMENT_ID'].csv
)
result.to_csv(data_save)