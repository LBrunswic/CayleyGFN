import itertools
import sys
import os
import tensorflow as tf

tf.config.experimental.enable_op_determinism()
from tensorboard.plugins.hparams import api as hp
GPU = 0
if GPU>=0:
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print(gpus)
        tf.config.set_visible_devices(gpus[GPU], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[GPU], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print('done!',logical_gpus)
    except Exception as e:
        tf.config.set_visible_devices([], 'GPU')
        print(e)
        raise
else:
    tf.config.set_visible_devices([], 'GPU')

import platform
import numpy as np
import time
time.sleep(1)
from datetime import datetime
from Graphs.CayleyGraph import Symmetric, path_representation
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from GFlowBase.regularization import *
from Groups.symmetric_groups import SymmetricUniform,Modal,rubick_generators,inversion,iteration_random
from metrics import ReplayBuffer,ExpectedLen,ExpectedReward,FlowSize,RandomCheck
from datetime import datetime
import logging
import signal
import pickle

log_dir = 'logs/hparam_tuning/'
series_name = '1'
#_____________PROBLEM DEFINITION___________________
SIZE = 10
GENERATORS = 'transpositions'
reward_fn = TwistedManhattan(SIZE,width=1,scale=-100,factor=1)
heuristic_fn = R_zero()
INITIAL_POSITION = SymmetricUniform(SIZE)
INVERSE=True

#___________MODEL HP________________
FlowEstimator_options = {
    'options': {
        'kernel_depth' : 4,
        'width' : 64,
        'final_activation' : 'linear',
    },
    'kernel_options': {
        # 'activation': tf.keras.layers.LeakyReLU(),
        'activation': 'tanh',
        # 'kernel_regularizer' : tf.keras.regularizers.L1L2(l1=0.01, l2=0.1)
    }
}

#___________TRAINING HP______________
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512]))
LENGTH_CUTOFF_FACTOR = hp.HParam('length_cutoff_factor', hp.Discrete([3]))
INIT_FLOW = hp.HParam('initial_flow', hp.Discrete([1e-3]))
LR = hp.HParam('learning_rate', hp.Discrete([1e-3]))
EPOCHS = hp.HParam('number_epoch', hp.Discrete([10]))
STEP_PER_EPOCH = hp.HParam('step_per_epoch', hp.Discrete([10]))
# B_BETA = hp.HParam('B_beta', hp.Discrete([0.001,0.01,0.1,1.]))
B_BETA = hp.HParam('B_beta', hp.Discrete([0.]))
NORMALIZATION_FN = hp.HParam('normalization_fn', hp.Discrete([1,2,3,4,5]))
REG_FN_alpha = hp.HParam('reg_fn_alpha', hp.Discrete([float(x) for x in np.linspace(start=-5,stop=-3,num=100)]))

REG_FN_logpmin = hp.HParam('reg_fn_logmin', hp.Discrete([20]))
EMBEDDING =[('hot', {'choice': None})]
#     hp.HParam('embedding', hp.Discrete([
#     [('hot', {'choice': None})],
#     [('cos',{'choice':None}), ('sin',{'choice':None})],
#
# ]))
a = hp.RealInterval(0.1, 0.2)

PATH_REDRAW = hp.HParam('path_redraw', hp.Discrete([0]))
NEIGHBORHOOD = hp.HParam('neighborhood', hp.Discrete([0]))

HP = [BATCH_SIZE, LENGTH_CUTOFF_FACTOR, INIT_FLOW,LR,EPOCHS,STEP_PER_EPOCH,B_BETA,NORMALIZATION_FN,PATH_REDRAW,NEIGHBORHOOD,REG_FN_alpha,REG_FN_logpmin]


normalization_fns =[
    None,
    lambda flownu,finit: 1e-3 + tf.reduce_sum(flownu[..., :4], axis=-1) / 2,
    lambda flownu,finit: 1e-3 + flownu[...,1],
    lambda flownu,finit: 1e-3 + finit,
    lambda flownu,finit: 1e-3 + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1)) ,
    lambda flownu,finit: 1e-3 + (finit+tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1))),
]






def train_test_model(hparams,log_dir=None,seed=1234):
    if log_dir is None:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.keras.utils.set_random_seed(seed)
    G = Symmetric(
        SIZE,
        generators=GENERATORS,
        representation=[('natural', {})],
        inverse=INVERSE,
        random_gen=INITIAL_POSITION,
        embedding=EMBEDDING,
        dtype='float32'
    )
    loss_fn = MeanABError(
        A=Alogsquare(),
        # B=Bpower(beta=hparams[B_BETA]),
        normalization_fn=normalization_fns[hparams[NORMALIZATION_FN]],
    )
    reg_fn = reg_fn_gen(
        alpha=hparams[REG_FN_alpha],
        logpmin=hparams[REG_FN_logpmin],
    )
    flow = GFlowCayleyLinear(
        graph=G,
        reward=Reward(
            reward_fn=reward_fn,
            heuristic_fn=heuristic_fn,
        ),
        batch_size=hparams[BATCH_SIZE],
        FlowEstimatorGen=(dense_gen, FlowEstimator_options),
        length_cutoff_factor=hparams[LENGTH_CUTOFF_FACTOR],
        initflow=hparams[INIT_FLOW],
        neighborhood=hparams[NEIGHBORHOOD],
        improve_cycle=hparams[PATH_REDRAW],
        reg_post=reg_post,
        reg_fn=reg_fn
        # reg_fn=reg_withcutoff_fn
    )
    flow(0)
    flow.FlowEstimator.kernel.summary()
    flow.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[LR]),
        loss=loss_fn,
    )

    Replay = ReplayBuffer()
    Replay.reward = reward_fn

    flow.metric_list.append(ExpectedReward())
    flow.metric_list.append(ExpectedLen())
    flow.metric_list.append(FlowSize())
    flow.metric_list.append(RandomCheck())
    flow.metric_list.append(tf.keras.metrics.Mean(name='initflow'))

    flow.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[LR]),
        loss=loss_fn,
    )

    flow.fit(
        np.zeros(hparams[STEP_PER_EPOCH]),
        initial_epoch=0,
        epochs=hparams[EPOCHS],
        verbose=1,
        batch_size=1,
        callbacks=[
            Replay,
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            )
        ]
    )
    return flow.evaluate()

METRIC = 'ExpectedReward'

with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
    hparams=HP,
    metrics=[hp.Metric(METRIC, display_name=METRIC)],
  )

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    loss_val,expected_reward, expected_len,flow_size,_,initflow = train_test_model(hparams,log_dir=run_dir)
    tf.summary.scalar(METRIC, expected_reward, step=1)



reg_post = proj_reg
SEEDS = [0,7508, 9845, 6726, 9339, 8885, 4892, 1347, 5243, 7591, 3661,
           6626, 1007, 3951, 1215, 5597, 6864, 4676, 3922, 2490, 3927, 8842,
           3203, 8428, 3366, 5770, 3226, 6206, 9234, 5521, 4414, 5157, 7775,
           2762, 9248, 7612, 2678, 4066, 6574, 8923, 4360, 9276, 2752, 8101,
           3122, 9882, 8790, 1894, 3426, 8760, 8499, 1964, 8675, 1109, 7910,
           8877, 4682, 1623, 1086, 3062, 1716, 5139, 6708, 6799,4445]

session_num = 0

# hp_names = [hp.name for hp in HP]
hp_sets = itertools.product(*[hp.domain.values for hp in HP])

for hp_set in hp_sets:
    hparams = {
        HP[i] : hp_val
        for i, hp_val in enumerate(hp_set)
    }
    print(hparams)
    run_name = "run" + series_name +"-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run(log_dir + run_name, hparams)
    session_num += 1
