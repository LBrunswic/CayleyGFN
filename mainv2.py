import sys
import os
import tensorflow as tf
from utils.parser import generate_argparser

GPU = generate_argparser().parse_args().GPU
print(GPU)
MEMORY_LIMIT = 2**generate_argparser().parse_args().MEMORY_LIMIT
# GPU=0
if GPU>=0:
  try:
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    tf.config.set_visible_devices(gpus[GPU], 'GPU')
    tf.config.experimental.set_virtual_device_configuration(gpus[GPU], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
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
from GFlowBase.rewards import R_one, R_first_one,H_first_one,Reward,R_zero,R_rubick,H_rubick,Manhattan
from metrics import FollowInitLoss, plot,ReplayBuffer
from metrics import represent_symmetric_R_first_one
from datetime import datetime
import logging
import signal
import time
import pickle

#TODO
# 2- Reward class with memory
# 3- outsourced reward computation
# 4- organize battery of tests for weekend
folder_add = generate_argparser().parse_args().folder
EPOCHS = generate_argparser().parse_args().EPOCHS
BATCH_SIZE = generate_argparser().parse_args().BATCH_SIZE
SIZE = generate_argparser().parse_args().SIZE
GENERATORS = generate_argparser().parse_args().GENERATORS
MLP_DEPTH = generate_argparser().parse_args().MLP_DEPTH
MLP_WIDTH = generate_argparser().parse_args().MLP_WIDTH
STEP_PER_EPOCH = generate_argparser().parse_args().STEP_PER_EPOCH
INVERSE = generate_argparser().parse_args().INVERSE
LR = generate_argparser().parse_args().LR
LOAD = bool(generate_argparser().parse_args().LOAD)
SCHEDULER = int(generate_argparser().parse_args().SCHEDULER)
LOSS = generate_argparser().parse_args().LOSS
BATCH_MEMORY = generate_argparser().parse_args().BATCH_MEMORY
SEED = generate_argparser().parse_args().SEED
HEURISTIC = generate_argparser().parse_args().HEURISTIC
REWARD = generate_argparser().parse_args().REWARD
length_cutoff_factor = generate_argparser().parse_args().length_cutoff_factor
EXPLORATION = generate_argparser().parse_args().EXPLORATION
BOOTSTRAP = generate_argparser().parse_args().BOOTSTRAP

tf.random.set_seed(SEED)

FOLDER_NAME = 'graphS%s'% SIZE + folder_add
folder = os.path.join('tests/',FOLDER_NAME)
os.makedirs(os.path.join('tests',FOLDER_NAME), exist_ok=True)
save_name = 'initloss'

logger = logging.getLogger('training')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(folder,'training.log'),mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)


logger.info('START')

PID_PATH=os.path.join(folder,'PID')
with open(PID_PATH,'w') as f:
    f.write(str(os.getpid()))

STOP = False
# def receive_signal(signum, stack):
#     global STOP
#     STOP = True
# try:
#     signal.signal(signal.SIGUSR1, receive_signal)
# except:
#     pass


G = Symmetric(
    SIZE,
    Gen=GENERATORS,
    inverse=INVERSE,
    k=1,
    # logger=logger
)

FlowEstimator = [
    dense_gen,
    # linear,
    ]

FlowEstimator_options = {
        'options': {
            'kernel_depth' : MLP_DEPTH,
            'width' : MLP_WIDTH,
            # 'final_activation' : tf.keras.layers.LeakyReLU(),
            # 'final_activation' : 'sigmoid',
            # 'final_activation' : 'swish',
            'final_activation' : 'linear',
        },
        'kernel_options': {
            'activation': tf.keras.layers.LeakyReLU(),
            # 'activation': 'swish',
            # 'activation': 'linear',
            'kernel_initializer' : tf.keras.initializers.HeNormal(),
            # 'kernel_regularizer' : tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
        }
    }

if HEURISTIC:
    heuristic_fn = H_first_one(SIZE)
else:
    heuristic_fn = R_zero(SIZE)
Rewards = {
    'R_one': R_one,
    'R_first_one':R_first_one,
    'Manhattan': Manhattan,
}

reward_fn_dic = {
    'Manhattan': lambda size,width:Manhattan(size,width=width),
    'R_first_one': R_first_one,
}
key = REWARD.split(',')[0]
param = [eval(x) for x in REWARD.split(',')[1:]]

flow = GFlowCayleyLinear(
    graph=G,
    # reward=R_one,
    # reward=R_main,
    reward=Reward(
        reward_fn=Rewards[key](SIZE,*param),
        heuristic_fn=heuristic_fn,
    ),
    batch_size=BATCH_SIZE,
    batch_memory=BATCH_MEMORY,
    FlowEstimatorGen=(FlowEstimator[0], FlowEstimator_options),
    length_cutoff_factor=length_cutoff_factor,
)


train_batch_size = BATCH_SIZE
X = tf.zeros((train_batch_size,(1+G.nactions),flow.embedding_dim))
flow(X)
flow.FlowEstimator.kernel.summary(print_fn=logger.info)
if LOAD:
    flow.load_weights(os.path.join(folder,'model.ckpt'))

if BOOTSTRAP != '':
    flow.load_weights(os.path.join('tests/',BOOTSTRAP,'model.ckpt'))




def scheduler(epoch, lr):
  if epoch < 1000:
    return LR
  else:
    return LR/(1+np.sqrt(epoch/5)-np.sqrt(998/5))

def exp_scheduler(epoch, lr):
  if epoch < 1000:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


def cosine_scheduler(steps):
    def aux(epoch,lr):
        lr_min = 0
        lr_max = LR/np.sqrt(1 + 0.3* (epoch//steps))
        return lr_min+(lr_max-lr_min)*0.5*(1+np.cos(np.pi*(epoch//steps)/steps))
    return tf.keras.callbacks.LearningRateScheduler(aux)

Scheduler = [
    tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr:lr),
    tf.keras.callbacks.LearningRateScheduler(exp_scheduler),
    cosine_scheduler(1000),
    cosine_scheduler(3000),
    cosine_scheduler(5000),
    cosine_scheduler(2000),
    cosine_scheduler(200),
]

loss_fn_dic = {
    'powB': lambda alphaA,alphaB,betaB:MeanABError(A=Apower(alphaA),B=Bpower(alphaB,betaB)),
    'pow': lambda alphaA:MeanABError(A=Apower(alphaA)),
    'Bengio':lambda alpha:divergence(),
}
key = LOSS.split(',')[0]
param = [eval(x) for x in LOSS.split(',')[1:]]
flow.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn_dic[key](*param)
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(folder,'model.ckpt'),
    verbose=1,
    save_weights_only=True,
    save_freq=STEP_PER_EPOCH
    )


Replay = ReplayBuffer(
    model=flow,
    epoch_length=STEP_PER_EPOCH,
    logger=logger,
    folder=folder,
    load=LOAD,
    # path_draw=True
)
logger.info('MEMORY_LIMIT: %s'% MEMORY_LIMIT)

for i in range(BATCH_MEMORY):
    Replay.gen_path(model=flow)

T = datetime.now()
logger.info('Training starts')
logger.handlers[0].flush()
start_epoch = len(Replay.InitVals)
for i in range(start_epoch,EPOCHS):
    if i == start_epoch + 1 :
        T = datetime.now()
    logger.info('EPOCH : %s/%s' % (i,EPOCHS) )
    dataset,mu_initial_reward = Replay.train_data(model=flow,exploration=EXPLORATION)
    logger.info('reward : \n %s' % mu_initial_reward[:5,2])
    flow.fit(
        dataset,
        mu_initial_reward,
        initial_epoch=i*STEP_PER_EPOCH,
        epochs=i*STEP_PER_EPOCH+STEP_PER_EPOCH,
        verbose=0,
        batch_size=dataset.shape[0],
        shuffle=False,
        callbacks=[
            Replay,
            cp_callback,
            Scheduler[SCHEDULER]
        ]
    )
    logger.info('Initial Flow: %s' % float(flow.initial_flow))
    T2=datetime.now()
    if i>=start_epoch+1:
        logger.info('ETA : %s' %( T2 + (T2-T)*(EPOCHS-i-start_epoch-1)/(i-start_epoch)))
    logger.handlers[0].flush()
    if STOP:
        break
if STOP:
    logger.info('Stop signal received')
    with open(PID_PATH, 'w') as f:
        f.write(str(-1))
else:
    logger.info('All done! Bye bye!')
logger.handlers[0].flush()


# s = SIZE
# G = Symmetric(s)
# generators = G.actions.numpy().astype('int')
# A=represent_symmetric_R_first_one(s,generators,flow.FlowEstimator)
# A.save('S%s' % SIZE)
