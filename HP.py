[
    {
        'batch_memory': [1],
        'epochs': [201],
        'memory_limit': [11],
        'inverse': [0],
        'generators': [
            # 'transpositions',
            'trans_cycle_a'
            # 'large_cycles'
        ],
        'length_cutoff_factor': [3],
        'scheduler': [0],
        'seed': [0,7508, 9845, 6726, 9339, 8885, 4892, 1347, 5243, 7591, 3661,
           6626, 1007, 3951, 1215, 5597, 6864, 4676, 3922, 2490, 3927, 8842,
           3203, 8428, 3366, 5770, 3226, 6206, 9234, 5521, 4414, 5157, 7775,
           2762, 9248, 7612, 2678, 4066, 6574, 8923, 4360, 9276, 2752, 8101,
           3122, 9882, 8790, 1894, 3426, 8760, 8499, 1964, 8675, 1109, 7910,
           8877, 4682, 1623, 1086, 3062, 1716, 5139, 6708, 6799,4445][1:9],
        'loss': [
            'AlogsquareB,2.,2,0.0001',
            'AlogsquareB,2.,2,0.001',
        ],
        'size': [10],
        'LR': [0.001],
        'MLP_depth': [3],
        'MLP_width': [32],
        'reward': [
            # 'TwistedManhattan,SIZE,1e-5,1.',
            # 'R_first_one',
            'R_first_k,1',
            # 'R_first_k,2',
            # 'R_first_k,3',
            # 'R_first_k,4',
            # 'R_first_k,5',
            # 'RubicksCube'
        ],
        'heuristic': [
            # 'TwistedManhattan,SIZE,1e-5,1e-2',
            'TwistedManhattan,1,1e-5,10.',
            # 'R_first_one',
            # 'R_first_k,1',
            # 'R_first_k,2',
            # 'R_first_k,3',
            # 'R_first_k,4',
            # 'R_first_k,5',
            # 'RubicksCube'
        ],
        'step_per_epoch': [10],
        'batchsize': [128],
        'exploration': [1e-4],
        # 'bootstrap': ['graphS3_%s' % i for i in range(8)]
        # 'load' : [i for i in range(8)]
        'initflow':[1e-3],
        'encoding':[10],
    },
]
