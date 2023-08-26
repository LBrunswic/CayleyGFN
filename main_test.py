import sys
import os
import tensorflow as tf
from utils.parser import generate_argparser
HP = generate_argparser().parse_args()
GPU = HP.GPU
print(GPU)
MEMORY_LIMIT = 2**generate_argparser().parse_args().MEMORY_LIMIT
# GPU=0
if GPU>=0:
  try:
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    tf.config.set_visible_devices(gpus[GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU], True)
    # tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
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
from time import time
from Graphs.CayleyGraph import Symmetric
from GFlowBase.kernel import dense_gen
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from metrics import ReplayBuffer
from datetime import datetime
import logging
import signal
import time
import pickle


tf.random.set_seed(12474)


loss_fn_dic = {
    'powB': lambda alphaA,alphaB,betaB:MeanABError(A=Apower(alphaA),B=Bpower(alphaB,betaB)),
    'AlogsquareB': lambda alphaA,alphaB,betaB:MeanABError(A=Alogsquare(alphaA),B=Bpower(alphaB,betaB)),
    'AlogsquareBdelta': lambda alphaA,alphaB,betaB,deltaB:MeanABError(A=Alogsquare(alphaA),B=Bpower(alpha=alphaB,beta=betaB,delta=deltaB)),
    'pow': lambda alphaA:MeanABError(A=Apower(alphaA)),
    'Bengio':lambda :divergence(),
}


SIZE = HP.SIZE
LOSS = HP.LOSS
BATCH_SIZE = HP.BATCH_SIZE
length_cutoff_factor = HP.length_cutoff_factor
initflow = HP.initflow
LR= HP.LR
EPOCHS = HP.EPOCHS
STEP_PER_EPOCH = HP.STEP_PER_EPOCH
GENERATORS = HP.GENERATORS

key = LOSS.split(',')[0]
param = [eval(x) for x in LOSS.split(',')[1:]]
loss_fn = loss_fn_dic[key](*param)
reward_fn = R_first_k(SIZE,1)
# heuristic_fn = R_zero()
heuristic_fn = TwistedManhattan(SIZE,width=1,scale=1e-4,factor=1.)

G = Symmetric(
    SIZE,
    generators=GENERATORS,
    representation = [('natural',{})],
    inverse = False,
    random_gen = 'uniform',
    embedding = [('hot',{}),('cos',{}),('sin',{})],
    dtype='float32'
)


FlowEstimator_options = {
        'options': {
            'kernel_depth' : 5,
            'width' : 64,
            'final_activation' : 'linear',
        },
        'kernel_options': {
            'activation': tf.keras.layers.LeakyReLU(),
            'kernel_regularizer' : tf.keras.regularizers.L1L2(l1=0.01, l2=0.1)
        }
    }

flow = GFlowCayleyLinear(
    graph=G,
    reward=Reward(
        reward_fn=reward_fn,
        heuristic_fn=heuristic_fn,
    ),
    batch_size=BATCH_SIZE,
    FlowEstimatorGen=(dense_gen, FlowEstimator_options),
    length_cutoff_factor=length_cutoff_factor,
    initflow=initflow
)

flow(np.zeros(0))
flow.FlowEstimator.kernel.summary()


flow.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn
)



Replay = ReplayBuffer()
R = flow.reward(G.random_gen(100000))
print(np.mean(R),np.sum(R**2)/np.sum(R))


flow.fit(
    np.zeros(STEP_PER_EPOCH),
    initial_epoch=0,
    epochs=EPOCHS,
    verbose=1,
    batch_size=1,
    callbacks=[
        Replay,
    ]
)
