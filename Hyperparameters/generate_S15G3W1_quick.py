import numpy as np
import itertools
import pickle
import os
import hashlib
rng = np.random.default_rng(seed=12345)

FOLDER = 'experimental_settings_S15G3W1_quick'

FIXED_HP = {
    'N_SAMPLE': 64,
    'graph_size': 15,
    'graph_generators': 'transpositions',
    'inverse': True,
    'initial_pos': 'SymmetricUniform',
    'rew_fn': 'TwistedManhattan',
    'rew_param': {'width': 6, 'scale': -100, 'exp': True, 'mini': 0.},
    'reg_fn_logmin': 5.0,
    'grad_batch_size': 1,
    'initial_flow': 0.001,
    'learning_rate': 0.002,
    'epochs': 100,
    'B_beta': -1000.0,
    'path_redraw': 0,
    'neighborhood': 0,
    'flowestimator_opt': {
        'options': {'kernel_depth': 2, 'width': 32, 'final_activation': 'linear'},
        'kernel_options': {'kernel_initializer': 'Orthogonal', 'activation': 'tanh', 'implementation': 1}},
    'optimizer': 'Adam',
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
    'reg_fn_alpha': (-10, 10),
}

VARIANLE_HP = {
    'reg_fn_gen': ['norm2','LogPathLen'],
    'reg_proj': ['OrthReg', 'AddReg'],
    'seed': list(rng.integers(1000, 10000, 128))[40:56],
    'normalization_fn': [0,1],
    'normalization_nu_fn': [0,3,5],
    'batch_size': [512],
    'step_per_epoch': [10],
    'length_cutoff': [30],
    'embedding': [('cos', 'sin', 'natural')],
    'loss_alpha': [2.0],
}

print(os.listdir())

os.makedirs(FOLDER,exist_ok=True)
variable_hp_names = list(VARIANLE_HP.keys())
for case in itertools.product(*[VARIANLE_HP[variable_hp_name] for variable_hp_name in variable_hp_names]):
    hp_set = dict(zip(variable_hp_names,case))
    hp_set.update(FIXED_HP)
    hash = hashlib.sha256(bytes(str(hp_set),'utf8')).hexdigest()
    with open(os.path.join(FOLDER,hash+'.hp'), 'wb') as f:
        pickle.dump( hp_set,f)


# hp_set = dict(zip(variable_hp_names, case))
# hp_set.update(FIXED_HP)
# hp_set['N_SAMPLE'] = 32
# hp_set['epochs'] = 3
# hash = hashlib.sha256(bytes(str(hp_set),'utf8')).hexdigest()
# with open(os.path.join(FOLDER,'base_test.hp'), 'wb') as f:
#     pickle.dump(hp_set,f)

print('Estimated time:', len(list(itertools.product(*[VARIANLE_HP[variable_hp_name] for variable_hp_name in variable_hp_names])))*FIXED_HP['N_SAMPLE']/2000/24)