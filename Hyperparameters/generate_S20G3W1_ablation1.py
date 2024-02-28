import numpy as np
import itertools
import pickle
import os
import hashlib
rng = np.random.default_rng(seed=12345)

FOLDER = 'HP'
BASE_HP = {
    #ARCHITECTURE
    'graph_size': [20],
    'graph_generators': ['transpositions'],
    'inverse': [True],
    'group_dtype': ['float32'],
    'initial_pos': ['SymmetricUniform'],
    'flowestimator_opt': [{
        'options': {'kernel_depth': 2, 'width': 32, 'final_activation': 'linear'},
        'kernel_options': {'kernel_initializer': 'Orthogonal', 'activation': 'tanh', 'implementation': 1}
    }],
    'reward_rescale': ['Trivial'],
    'path_redraw': [0],
    'neighborhood': [0],
    'length_cutoff': [30],
    'embedding': [('cos', 'sin', 'natural')],

    #REWARD
    'rew_fn': ['TwistedManhattan'],
    'rew_param': [{'width': 1, 'scale': -100, 'exp': False, 'mini': 0.0}],
    'heuristic_fn': ['R_zero'],
    'heuristic_param': [{}],
    'heuristic_factor': [0.0],
    'rew_factor': [1.0],

    #LOSS
    'loss_cutoff': ['none'],
    'loss_alpha': [2.0],

    #REGULARIZATION
    'reg_fn_gen': ['norm2','LogPathLen','LogEPathLen'],
    'B_beta': [None],
    'reg_proj': ['AddReg','OrthReg'],
    'reg_fn_logmin': [5.0],
    'reg_fn_alpha_schedule': ['none'],
    'alpha': [0.],

    #TRAINING
    'batch_size': [512],
    'grad_batch_size': [1],
    'optimizer': ['AdamW'],
    'initial_flow': [0.001],
    'learning_rate': [0.001],
    'lr_schedule': ['none'],
    'epochs': [800],
    'step_per_epoch': [10],
    'path_strategy': ['baseline','best_reward'],

    #NORMALIZATION
    'normalization_fn': [2],
    'normalization_nu_fn': [0],

    #HP_TUNING
    'tuning_method': ['fn_alpha_tune_grid'],

    #BENCHMARK
    'bench_seed': [1234],
    'bench_batch_size': [4096],
}


TUNER_HP = {
    'alpha_range': (-1.4, 1.4),
    'N_SAMPLE': 5,
}

SEED = list(rng.integers(1000, 10000, 128))[0:16]


VARIALE_HP = {


}
for bbs in BASE_HP['bench_batch_size']:
    for bs in BASE_HP['batch_size']:
        assert(bbs%bs==0)
os.makedirs(FOLDER,exist_ok=True)
variable_hp_names = list(BASE_HP.keys())
for case in itertools.product(*[BASE_HP[variable_hp_name] for variable_hp_name in variable_hp_names]):
    hp_set = dict(zip(variable_hp_names,case))
    hp_set.update(TUNER_HP)
    hash = hashlib.sha256(bytes(str(hp_set), 'utf8')).hexdigest()
    for seed in SEED:
        hp_set.update({'seed':seed})
        with open(os.path.join(FOLDER,hash+f'_{seed}.hp'), 'wb') as f:
            pickle.dump(hp_set,f)