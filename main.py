import sys
import os
import tensorflow as tf
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
from time import time
from datetime import datetime
from Graphs.CayleyGraph import Symmetric, path_representation
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from Groups.symmetric_groups import SymmetricUniform,Modal,rubick_generators,inversion,iteration_random
from metrics import ReplayBuffer,ExpectedLen,ExpectedReward,FlowSize
from datetime import datetime
import logging
import signal
import time
import pickle

tf.random.set_seed(12474)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")


SIZE = 10
BATCH_SIZE = 2048
length_cutoff_factor = 3
initflow = 1e-3
LR= 1e-2
EPOCHS = 500
STEP_PER_EPOCH = 10
GENERATORS = 'transpositions'
improve_cycle = 0
NEIGHBORHOOD = 0
reward_fn = TwistedManhattan(SIZE,width=1,scale=-100,factor=1)
heuristic_fn = R_zero()

loss_fn = MeanABError(
    A=Alogsquare(alpha=2.),
    normalization= lambda flownu:1e-3+flownu[...,1]
)




G = Symmetric(
    SIZE,
    generators=GENERATORS,
    representation = [('natural',{})],
    inverse = True,
    # random_gen = Modal(SIZE,[np.arange(SIZE)]),
    random_gen = SymmetricUniform(SIZE),
    embedding = [
        ('hot',{'choice':None}),
    ],
    dtype='float32'
)
print(GENERATORS)

FlowEstimator_options = {
    'options': {
        'kernel_depth' : 1,
        'width' : 64,
        'final_activation' : 'linear',
    },
    'kernel_options': {
        # 'activation': tf.keras.layers.LeakyReLU(),
        'activation': 'tanh',
        # 'kernel_regularizer' : tf.keras.regularizers.L1L2(l1=0.01, l2=0.1)
    }
}
@tf.function
def straight_reg(self,loss_gradients,reg_gradients):
    n_train =  self.n_train
    res = [loss_gradients[i]+reg_gradients[i]  for i in range(n_train)]
    return res


@tf.function
def proj_reg(self,loss_gradients,reg_gradients):
    n_train =  self.n_train
    res = [
        loss_gradients[i] + reg_gradients[i] -  projection_orth(reg_gradients[i],loss_gradients[i])
        for i in range(n_train)
    ]
    return res


@tf.function
def projection_orth(u,v):
    v_norm = tf.math.l2_normalize(v)
    return tf.reduce_sum(u*v_norm)*v_norm

@tf.function
def reg_fn(Flownu):
    logdensity_trainable = Flownu[..., 5]
    Ebigtau = tf.reduce_mean(logdensity_trainable[:,-1])
    return 1e-1*tf.nn.relu(20+Ebigtau)

@tf.function
def reg_withcutoff_fn(Flownu):
    logdensity_trainable = Flownu[..., 5]
    Expected_length = tf.reduce_mean(tf.reduce_sum(Flownu[..., 4],axis=1))
    cutoff = tf.cast(tf.reduce_min([3*Expected_length,Flownu[..., 4].shape[1]])-1,dtype='int32')
    Ebigtau = tf.reduce_mean(logdensity_trainable[:,cutoff])
    return tf.nn.relu(40+Ebigtau)

reg_post = proj_reg

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
    neighborhood=NEIGHBORHOOD,
    improve_cycle=improve_cycle,
    reg_post=reg_post,
    reg_fn=reg_fn
    # reg_fn=reg_withcutoff_fn
)
flow(0)
flow.FlowEstimator.kernel.summary()
flow.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn,
)

Replay = ReplayBuffer()
Replay.reward = reward_fn
print([x.name for x in flow.metrics])
# print([x.name for x in flow.compiled_metrics])



flow.metric_list.append(ExpectedReward())
flow.metric_list.append(ExpectedLen())
flow.metric_list.append(FlowSize())
flow.metric_list.append(tf.keras.metrics.Mean(name='initflow'))

flow.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn,
)
print(flow.metrics)
print(flow.compiled_metrics)


flow.fit(
    np.zeros(STEP_PER_EPOCH),
    initial_epoch=0,
    epochs=EPOCHS,
    verbose=1,
    batch_size=1,
    callbacks=[
        Replay,
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
)
