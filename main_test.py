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
from Graphs.CayleyGraph import Symmetric, path_representation
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from Groups.symmetric_groups import SymmetricUniform,SymmetricModal,rubick_generators,inversion,iteration_random
from metrics import ReplayBuffer
from datetime import datetime
import logging
import signal
import time
import pickle

# tf.random.set_seed(12474)


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
NEIGHBORHOOD = HP.NEIGHBORHOOD

key = LOSS.split(',')[0]
param = [eval(x) for x in LOSS.split(',')[1:]]
loss_fn = loss_fn_dic[key](*param)
# reward_fn = R_first_k(1,2)
# reward_fn = R_first_k(1,SIZE)
reward_fn = TwistedManhattan(SIZE,width=2,scale=-10,factor=1)
heuristic_fn = R_zero()
# heuristic_fn = TwistedManhattan(SIZE,width=SIZE,scale=1e-3,factor=1e-3)


# def rubick_modes(depth):
#     rubicks_generators = [inversion(x) for x in rubick_generators(48)[0]] + rubick_generators(48)[0]
#     rubicks_generators = np.array(list(set([tuple(x) for x in rubicks_generators])),dtype='int32')
#     def iter(base,g,N):
#         old = [tuple(x) for x in base]
#         old_set = set(old)
#         new = [tuple(x) for x in base[...,g].reshape(-1,N) if tuple(x) not in old_set]
#         return np.array(old+new)
#     modes = np.arange(48,dtype='int32').reshape(1,-1)
#     sizes = [1]
#     for i in range(depth):
#         modes = iter(modes,rubicks_generators,48)
#         sizes.append(len(modes))
#     return modes,sizes
# modes,sizes = rubick_modes(1)

G = Symmetric(
    SIZE,
    generators=GENERATORS,
    representation = [('natural',{})],
    inverse = True,
    # random_gen = SymmetricModal(SIZE,modes),
    random_gen = SymmetricUniform(SIZE),
    embedding = [
        ('hot',{'choice':None}),
        # ('hot',{'choice':[0]}),
        # ('cos',{})
    ],
    dtype='float32'
)
print(GENERATORS)

FlowEstimator_options = {
        'options': {
        'kernel_depth' : 3,
            'width' : 64,
            'final_activation' : 'linear',
            # 'head' : direct_head
        },
        'kernel_options': {
            # 'activation': tf.keras.layers.LeakyReLU(),
            'activation': 'tanh',
            # 'kernel_regularizer' : tf.keras.regularizers.L1L2(l1=0.01, l2=0.1)
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
    initflow=initflow,
    neighborhood=NEIGHBORHOOD
)

flow(np.zeros(0))
flow.FlowEstimator.kernel.summary()
# R = flow.reward(G.sample(100000))
# print(np.mean(R),np.sum(R**2)/np.sum(R))

flow.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn
)


Replay = ReplayBuffer()
Replay.reward = reward_fn

class LearningCuriculum(tf.keras.callbacks.Callback):
    def on_train_begin(self,logs=None):
        period1 = 1
        period2 = 10001
        period3 = 2009
        period4 = 1000
        period5 = 2000
        print(modes[0])
        print(flow.graph.random_gen.sample(10))
        flow.graph.random_gen.set_cutoff_logits(sizes[0])
    def on_epoch_end(self, epoch, logs=None):
        if epoch == period1:
            self.model.graph.random_gen.set_cutoff_logits(sizes[1])
            print(flow.graph.random_gen.sample(10))
        if epoch == period2:
            self.model.graph.random_gen.set_cutoff_logits(sizes[2])
            print(flow.graph.random_gen.sample(10))
        if epoch == period3:
            self.model.graph.random_gen.set_cutoff_logits(sizes[3])
            print(flow.graph.random_gen.sample(10))
        if epoch == period4:
            self.model.graph.random_gen.set_cutoff_logits(sizes[4])
            print(flow.graph.random_gen.sample(10))
        if epoch == period5:
            self.model.graph.random_gen.set_cutoff_logits(sizes[5])
            print(flow.graph.random_gen.sample(10))

flow.fit(
    np.zeros(STEP_PER_EPOCH),
    initial_epoch=0,
    epochs=EPOCHS,
    verbose=1,
    batch_size=1,
    callbacks=[
        Replay,
        # LearningCuriculum()
    ]
)
