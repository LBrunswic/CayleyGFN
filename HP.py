[
    {
        'batch_memory': [1],
        'epochs': [5000],
        'memory_limit': [11],
        'inverse': [0],
        'generators': ['trans_cycle_a'],
        'length_cutoff_factor': [2],
        'scheduler': [4],
        'seed': [1234, 4593, 3651, 6874, 1457, 8523, 1367, 8577, 9854, 1657, 6972, 9687, 7592, 7712, 1545, 3855, 8474,
                 1275, 9684, 1269, 1077, 8992, 6280, 4821, 8279, 2740, 5662, 1394, 2321, 5893, 7383, 432, 9319, 839,
                 8432, 5178, 4278, 8794, 8189, 4995, 3929, 5133, 4107, 3343, 7621, 7181, 5057, 6770, 185, 6154, 1821],
        'loss': [
            'powB,2,1.1,0.1',
            # 'powB,2,1,0.2',
            # 'powB,2,1,0.3',
            # 'powB,2,1,1',
            # 'powB,2,1,5',
            # 'powB,2,0.5,0.4',
            # 'powB,2,0.5,0.3',
        ],
        'size': [10],
        'LR': [0.002],
        'MLP_depth': [3],
        'reward': [
            # 'Manhattan,1',
            # 'Manhattan,2',
            # 'Manhattan,3',
            # 'Manhattan,4',
            # 'Manhattan,5',
            # 'Manhattan,6',
            # 'Manhattan,7',
            # 'Manhattan,8',
            'R_first_one'
        ],
        'step_per_epoch': [20],
        'batchsize': [64],
        'exploration': [5 * 1e-1],

        # 'bootstrap': ['graphS54_1215']
    },
]
