import sys,os
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from itertools import permutations
import numpy as np
from Graphs.CayleyGraph import Symmetric
from GFlowBase.kernel import dense_gen
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import R_one, R_first_one,H_first_one,Reward,R_zero,R_rubick,H_rubick,Manhattan

def edgeflow(FOLDER):
    # FOLDER = sys.argv[1]
    # FOLDER = 'tests/graphS3_5'
    print(FOLDER)
    with open(os.path.join(FOLDER,'HP_dict'),'r') as f:
        HP = eval(''.join(f.readlines()))
    print(HP['reward'])
    default = {
        'MLP_width':128
    }
    for key in default:
        if key not in HP:
            HP[key] = default[key]
    print(HP)
    G = Symmetric(
        HP['size'],
        Gen=HP['generators'],
        inverse=HP['inverse'],
        k=1,
    )

    FlowEstimator = [
        dense_gen,
        ]

    FlowEstimator_options = {
            'options': {
                'kernel_depth' : HP['MLP_depth'],
                'width' : HP['MLP_width'],
                'final_activation' : 'linear',
            },
            'kernel_options': {
                'activation': tf.keras.layers.LeakyReLU(),
                'kernel_initializer' : tf.keras.initializers.HeNormal(),
            }
        }
    Rewards = {
        'R_one': R_one,
        'R_first_one':R_first_one,
        'Manhattan': Manhattan,
    }

    reward_fn_dic = {
        'Manhattan': lambda size,width:Manhattan(size,width=width),
        'R_first_one': R_first_one,
    }

    key = HP['reward'].split(',')[0]
    param = [eval(x) for x in HP['reward'].split(',')[1:]]

    flow = GFlowCayleyLinear(
        graph=G,
        reward=Reward(
            reward_fn=Rewards[key](HP['size'],*param)
        ),
        batch_size=HP['batchsize'],
        batch_memory=HP['batch_memory'],
        FlowEstimatorGen=(FlowEstimator[0], FlowEstimator_options),
        length_cutoff_factor=HP['length_cutoff_factor'],
    )
    train_batch_size = HP['batchsize']
    X = tf.zeros((train_batch_size,(1+G.nactions),flow.embedding_dim))
    flow(X)
    flow.load_weights(os.path.join(FOLDER,'model.ckpt'))
    states = np.array(list(permutations(list(range(HP['size'])))),dtype='float32')
    return flow.FlowEstimator(states).numpy(),flow,states
