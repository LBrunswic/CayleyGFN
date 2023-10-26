
PROBLEM_HYPERPARAMETERS = {
    'graph_size': 'freeint',
    'graph_generators': ['trans_cycle_a', 'cycles_a', 'transpositions'],
    'inverse': 'freebool',
    'initial_pos': ['SymmetricUniform'],
    'rew_fn':['R_zero','TwistedManhattan'],
    'rew_param':'freedict',
    'rew_factor':'freefloat',
}

MODEL_HYPERPARAMETERS = {
    'reg_fn_alpha':'freefloat',
    'reg_proj':['OrthReg','AddReg'],
    'reg_fn_logmin':'freefloat',
    'grad_batch_size':'freeint',
    'batch_size':'freeint',
    'length_cutoff':'freeint',
    'initial_flow':'freefloat',
    'learning_rate':'freefloat',
    'epochs':'freeint',
    'step_per_epoch':'freeint',
    'B_beta':'freefloat',
    'path_redraw':'freeint',
    'neighborhood':'freeint',
    'flowestimator_opt':'freedict',
    'embedding': (['cos','sin','natural','hot'], lambda x: '_'.join(x)),
    'optimizer':['Adam'],
    'loss_base':['Apower'],
    'loss_alpha':'freefloat',
    'heuristic_fn':['R_zero','TwistedManhattan'],
    'heuristic_param':'freedict',
    'heuristic_factor':'freefloat',
    'reward_rescale':['Trivial','ERew'],
    'reg_fn_gen':['norm2','LogPathLen','PathAccuracy'],
    'loss_cutoff':['none'],
    'lr_schedule':['none'],
    'reg_fn_alpha_schedule':['none'],
    'normalization_fn': [0, 1, 2, 3, 4, 5],
    'normalization_nu_fn': [0, 1, 2, 3],
    'SEED_REPEAT': 'freeint',
    'seed': 'freeint',
    'N_SAMPLE': 'freeint',
    'group_dtype':['float32','int32','int16']
}

TUNABLE_HYPERPARAMETERS = ['reg_fn_alpha','B_beta']


def naming(FIXED_HYPERPARAMETERS,TUNING_HYPERPARAMETERS):
    return 'S'+str(FIXED_HYPERPARAMETERS['graph_size'][0])+'_'+'_'.join(FIXED_HYPERPARAMETERS['graph_generators'])
SEEDS = [8978, 9220, 4213, 8814, 7814, 6908, 7000, 9398, 8151, 5196, 1852, 5852, 8847, 2692, 8377, 3862, 1417, 8229, 5107, 6636, 6032, 5099, 2789, 5692]
SEEDS += [10283,19943,11111,291110,209871,26379,986401,754671,536187,983767,99732795,
    7508, 9845, 6726, 9339, 8885, 4892, 1347, 5243, 7591, 3661,
           6626, 1007, 3951, 1215, 5597, 6864, 4676, 3922, 2490, 3927, 8842,
           3203, 8428, 3366, 5770, 3226, 6206, 9234, 5521, 4414, 5157, 7775,
           2762, 9248, 7612, 2678, 4066, 6574, 8923, 4360, 9276, 2752, 8101,
           3122, 9882, 8790, 1894, 3426, 8760, 8499, 1964, 8675, 1109, 7910,
           8877, 4682, 1623, 1086, 3062, 1716, 5139, 6708, 6799,4445]