import sys,os
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from itertools import permutations
import numpy as np
from Graphs.CayleyGraph import Symmetric
from GFlowBase.kernel import dense_gen
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from Groups.symmetric_groups import random_perm
from metrics import FollowInitLoss, plot,ReplayBuffer

def edgeflow(FOLDER):
    # FOLDER = sys.argv[1]
    # FOLDER = 'tests/graphS3_5'
    print(FOLDER)
    with open(os.path.join(FOLDER,'HP_dict'),'r') as f:
        HP = eval(''.join(f.readlines()))
    print(HP['reward'])
    default = {
        'MLP_width':128,
        'heuristic_param':1e-4,
        'heuristic_scale':1.
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
                'encoding':HP['encoding']
            },
            'kernel_options': {
                'activation': tf.keras.layers.LeakyReLU(),
                'kernel_initializer' : tf.keras.initializers.HeNormal(),
            }
        }



    reward_fn_dic = {
        'Manhattan': lambda size,width:Manhattan(size,width=width),
        'R_first_one': R_first_one,
        'RubicksCube': R_first_one,
        'R_first_k': R_first_k,
    }
    key = HP['reward'].split(',')[0]
    param = [eval(x) for x in HP['reward'].split(',')[1:]]
    if HP['heuristic']:
        if HP['heuristic']<=HP['size']:
            heuristic_fn = Manhattan(HP['size'],width=HP['heuristic'])
        else:
            heuristic_fn = TwistedManhattan(
                HP['size'],
                width=HP['heuristic']-HP['size'],
                scale=HP['heuristic_param'],
                factor=HP['heuristic_scale']
            )


    else:
        heuristic_fn = R_zero(HP['size'])


    flow = GFlowCayleyLinear(
        graph=G,
        reward=Reward(
            reward_fn=reward_fn_dic[key](HP['size'],*param),
            heuristic_fn=heuristic_fn,
        ),
        batch_size=HP['batchsize'],
        batch_memory=HP['batch_memory'],
        FlowEstimatorGen=(FlowEstimator[0], FlowEstimator_options),
        length_cutoff_factor=HP['length_cutoff_factor'],

    )
    train_batch_size = HP['batchsize']
    X = tf.zeros((train_batch_size,(1+G.nactions),flow.embedding_dim))
    flow(X)
    flow.load_weights(os.path.join(FOLDER,'model1000.ckpt'))
    # flow.load_weights(os.path.join(FOLDER,'model.ckpt'))

    flow2 = GFlowCayleyLinear(
        graph=G,
        reward=Reward(
            reward_fn=reward_fn_dic[key](HP['size'],*param),
            heuristic_fn=heuristic_fn,
        ),
        batch_size=HP['batchsize']*4,
        batch_memory=HP['batch_memory'],
        FlowEstimatorGen=(FlowEstimator[0], FlowEstimator_options),
        length_cutoff_factor=5
    )
    flow2(X)
    for i,weight in enumerate(flow2.FlowEstimator.weights):
        weight.assign(flow.FlowEstimator.weights[i])
    # flow2.initflow=flow.initflow
    # states = np.array(list(permutations(list(range(HP['size'])))),dtype='float32')
    Replay2 = ReplayBuffer(
        model=flow2,
        # epoch_length=20,
        # path_draw=True
    )
    Replay2.model = flow2
    Replay1 = ReplayBuffer(
        model=flow,
        # epoch_length=20,
        # path_draw=True
    )
    Replay1.model = flow
    return flow,flow2,Replay1,Replay2


def estimate_reward(reward,size,N=10000):
    return np.mean(reward(random_perm(N,size)))
