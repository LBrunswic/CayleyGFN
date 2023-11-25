import numpy as np
import itertools
import pickle
import os
import hashlib
rng = np.random.default_rng(seed=12345)

CPU_FRACTION = 0.2
FIXED_HP = {
    'N_SAMPLE': 32*24,
    'graph_size': 15,
    'graph_generators': 'trans_cycle_a',
    'inverse': True,
    'initial_pos': 'SymmetricUniform',
    'rew_fn': 'TwistedManhattan',
    'rew_param': {'width': 1, 'scale': -100, 'exp': False, 'mini': 0.0},
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
    'reg_fn_alpha': (-20, 20),
}

VARIANLE_HP = {
    'reg_fn_gen': ['norm2'],
    'reg_proj': ['AddReg'],
    'seed': list(rng.integers(1000, 10000, 4)),
    'normalization_fn': [0],
    'normalization_nu_fn': [0],
    'batch_size': [256,512,1024],
    'step_per_epoch': [5,10,20],
    'length_cutoff': [15,30,40],
    'embedding': [('cos', 'sin', 'natural')],
    'loss_alpha': [2.0,1.5,3.,0.5,5.],
}

print(os.listdir())
FOLDER_BASE = 'experimental_settings_low'
os.makedirs(FOLDER_BASE,exist_ok=True)
os.makedirs(FOLDER_BASE+'_cpu',exist_ok=True)
variable_hp_names = list(VARIANLE_HP.keys())

cases = list(itertools.product(*[VARIANLE_HP[variable_hp_name] for variable_hp_name in variable_hp_names]))
N_CPU = int(CPU_FRACTION*len(cases))
for i,case in enumerate(cases):
    hp_set = dict(zip(variable_hp_names,case))
    hp_set.update(FIXED_HP)
    hash = hashlib.sha256(bytes(str(hp_set),'utf8')).hexdigest()
    if i<N_CPU:
        FOLDER = FOLDER_BASE+'_cpu'
    else:
        FOLDER = FOLDER_BASE
    with open(os.path.join(FOLDER,hash+'.hp'), 'wb') as f:
        pickle.dump( hp_set,f)

print('Estimated time:', len(list(itertools.product(*[VARIANLE_HP[variable_hp_name] for variable_hp_name in variable_hp_names])))*FIXED_HP['N_SAMPLE']/700/24)