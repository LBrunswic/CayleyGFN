import numpy as np
import itertools
import pickle
import os
import hashlib
rng = np.random.default_rng(seed=12345)

FOLDER = 'HP'

FIXED_HP = {
    'N_SAMPLE': 32,
    'graph_size': 15,
    'graph_generators': 'transpositions',
    'inverse': True,
    'initial_pos': 'SymmetricUniform',
    'rew_fn': 'TwistedManhattan',
    'rew_param': {'width': 2, 'scale': -100, 'exp': False, 'mini': 0.0},
    'reg_fn_logmin': 5.0,
    'grad_batch_size': 1,
    'initial_flow': 0.001,
    'learning_rate': 0.001,
    'epochs': 40,
    'B_beta': -1000.0,
    'path_redraw': 0,
    'neighborhood': 0,
    'flowestimator_opt': {
        'options': {'kernel_depth': 2, 'width': 32, 'final_activation': 'linear'},
        'kernel_options': {'kernel_initializer': 'Orthogonal', 'activation': 'tanh', 'implementation': 1}},
    'optimizer': 'AdamW',
    'loss_base': 'Apower',
    'rew_factor': 1.0,
    'heuristic_fn': 'R_zero',
    'heuristic_param': {},
    'heuristic_factor': 0.0,
    'reward_rescale': 'Trivial',
    'loss_cutoff': 'none',
    'lr_schedule': 'none',
    'reg_fn_alpha_schedule': 'none',
    'group_dtype': 'float32',
    'reg_fn_alpha': (-4, 4),
}

VARIANLE_HP = {
    'reg_fn_gen': ['LogPathLen'],
    'reg_proj': ['AddReg','OrthReg'],
    'seed': list(rng.integers(1000, 10000, 128))[0:4],
    'normalization_fn': [1,3],
    'normalization_nu_fn': [5],
    'batch_size': [2048],
    'step_per_epoch': [50],
    'length_cutoff': [30],
    'embedding': [('cos', 'sin', 'natural')],
    'loss_alpha': [2.0],
}

os.makedirs(FOLDER,exist_ok=True)
variable_hp_names = list(VARIANLE_HP.keys())
for case in itertools.product(*[VARIANLE_HP[variable_hp_name] for variable_hp_name in variable_hp_names]):
    hp_set = dict(zip(variable_hp_names,case))
    hp_set.update(FIXED_HP)
    hash = hashlib.sha256(bytes(str(hp_set),'utf8')).hexdigest()
    with open(os.path.join(FOLDER,hash+'.hp'), 'wb') as f:
        pickle.dump(hp_set,f)