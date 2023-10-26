
FlowEstimator_options = [{
        'options': {
            'kernel_depth' : 2,
            'width' : 32,
            'final_activation' : 'linear',
        },
        'kernel_options': {

            'kernel_initializer' : 'Orthogonal',
            'activation': 'tanh',
            'implementation': 1
        }
    },

    {
            'options': {
                'kernel_depth' : 2,
                'width' : 32,
                'final_activation' : 'linear',
            },
            'kernel_options': {

                'kernel_initializer' : 'Orthogonal',
                'activation': 'relu',
                'implementation': 1
            }
    },
]

reward_param = {
    'width':1,
    'scale':-100,
    'exp':False,
    'mini':0.
}

FIXED_HYPERPARAMETERS = {
    'graph_size': [15],
    'graph_generators': ['trans_cycle_a', 'cycles_a', 'transpositions'][:1],
    'inverse': [True],
    'initial_pos': ['SymmetricUniform'],
    'rew_fn':['TwistedManhattan'],
    'rew_param':[reward_param],
    'reg_proj':['OrthReg','AddReg'],
    'reg_fn_logmin':[5.],
    'grad_batch_size':[1],
    'batch_size':[1024],
    'length_cutoff':[30],
    'initial_flow':[1e-3],
    'learning_rate':[5*1e-3],
    'epochs':[10],
    'step_per_epoch':[5],
    'B_beta':[-1000.],
    'path_redraw':[0],
    'neighborhood':[0],
    'flowestimator_opt':FlowEstimator_options[:1],
    'embedding': [('cos','sin','natural')],
    'optimizer':['Adam'],
    'loss_base':['Apower'],
    'loss_alpha':[2.],
    'rew_factor':[0.1,1.,10][1:2],
    'heuristic_fn':['R_zero'],
    'heuristic_param':[{}],
    'heuristic_factor':[0.],
    'reward_rescale':['Trivial'],
    'reg_fn_gen':['norm2','LogPathLen'],
    'loss_cutoff':['none'],
    'lr_schedule':['none'],
    'reg_fn_alpha_schedule':['none'],
    'normalization_fn': [0],
    'normalization_nu_fn': [0],
    'group_dtype': ['float32']
}


TUNING_HYPERPARAMETERS = {
    'reg_fn_alpha': [(-20,40)]
}

HARDWARE_PARAMETERS = {
    "POOL_SIZE": 1,
    "GPU_WORKER": 2,
    'CPU_WORKER': 0,
    'GPU_MEMORY': 16300

}

DENSITY_PARAMETERS = {
    'SEED_REPEAT': [8],
    'N_SAMPLE': [600]
}



