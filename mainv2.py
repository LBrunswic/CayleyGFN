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
from Graphs.CayleyGraph import Symmetric,RubicksCube
from GFlowBase.kernel import dense_gen
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
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
initflow = generate_argparser().parse_args().initflow
ENCODING = generate_argparser().parse_args().ENCODING
HEURISTIC_PARAM = generate_argparser().parse_args().HEURISTIC_PARAM
heuristic_scale = generate_argparser().parse_args().heuristic_scale
if SEED == 0:
    SEED = 1234
    BASELINE = 1
else:
    BASELINE = 0

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

if REWARD != 'RubicksCube':
    G = Symmetric(
        SIZE,
        Gen=GENERATORS,
        inverse=INVERSE,
        k=1,
        # logger=logger
    )
else:
    G = RubicksCube(
        inverse=INVERSE,
    )
logger.info(str(G.initial))
FlowEstimator = [
    dense_gen,
    ]

FlowEstimator_options = {
        'options': {
            'kernel_depth' : MLP_DEPTH,
            'width' : MLP_WIDTH,
            # 'final_activation' : tf.keras.layers.LeakyReLU(),
            'final_activation' : 'linear',
            # 'final_activation' : 'sigmoid',
            # 'final_activation' : 'swish',
            # 'final_activation' : lambda x: 5*tf.math.tanh(x),
            'encoding':ENCODING,
        },
        'kernel_options': {
            'activation': tf.keras.layers.LeakyReLU(),
            # 'activation': 'tanh',
            # 'activation': 'swish',
            # 'activation': 'linear',
            # 'kernel_initializer' : tf.keras.initializers.HeNormal(),
            # 'kernel_initializer' : tf.keras.initializers.Orthogonal(seed=SEED),
            'kernel_regularizer' : tf.keras.regularizers.L1L2(l1=0.01, l2=0.1)
        }
    }

if BASELINE:
    FlowEstimator_options['options']['final_activation'] = lambda x:0*x


reward_fn_dic = {
    'Manhattan': lambda size,width:Manhattan(size,width=width),
    'R_first_one': R_first_one,
    'RubicksCube': R_first_one,
    'R_first_k': R_first_k,
    'TwistedManhattan':lambda size,width,scale,factor:TwistedManhattan(size,width=width,scale=scale,factor=factor),
}

key = REWARD.split(',')[0]
param = [eval(x) for x in REWARD.split(',')[1:]]
if HEURISTIC:
    if HEURISTIC<=SIZE:
        heuristic_fn = Manhattan(SIZE,width=HEURISTIC)
    else:
        heuristic_fn = TwistedManhattan(SIZE,width=HEURISTIC-SIZE,scale=HEURISTIC_PARAM,factor=heuristic_scale)
else:
    heuristic_fn = R_zero(SIZE)

flow = GFlowCayleyLinear(
    graph=G,
    reward=Reward(
        reward_fn=reward_fn_dic[key](SIZE,*param),
        heuristic_fn=heuristic_fn,
    ),
    batch_size=BATCH_SIZE,
    batch_memory=BATCH_MEMORY,
    FlowEstimatorGen=(FlowEstimator[0], FlowEstimator_options),
    length_cutoff_factor=length_cutoff_factor,
    initflow=initflow
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
    if epoch < 200:
        return LR
    else:
        alpha=0.5
        phi = lambda x:np.power(x,alpha)
        MAX_STEP = EPOCHS*STEP_PER_EPOCH
        return LR*phi(MAX_STEP-epoch)/phi(MAX_STEP)

def exp_scheduler(epoch, lr):
  if epoch < 200:
    return lr
  else:
    return lr * 0.99983


def cosine_scheduler(steps):
    def aux(epoch,lr):
        lr_min = 0
        lr_max = LR/np.sqrt(1 + 0.3* (epoch//steps))
        return lr_min+(lr_max-lr_min)*0.5*(1+np.cos(np.pi*(epoch//steps)/steps))
    return tf.keras.callbacks.LearningRateScheduler(aux)

Scheduler = [
    tf.keras.callbacks.LearningRateScheduler(lambda epoch,lr:lr),
    tf.keras.callbacks.LearningRateScheduler(scheduler),
    cosine_scheduler(1000),
    cosine_scheduler(3000),
    cosine_scheduler(5000),
    cosine_scheduler(2000),
    cosine_scheduler(200),
]

loss_fn_dic = {
    'powB': lambda alphaA,alphaB,betaB:MeanABError(A=Apower(alphaA),B=Bpower(alphaB,betaB)),
    'AlogsquareB': lambda alphaA,alphaB,betaB:MeanABError(A=Alogsquare(alphaA),B=Bpower(alphaB,betaB)),
    'AlogsquareBdelta': lambda alphaA,alphaB,betaB,deltaB:MeanABError(A=Alogsquare(alphaA),B=Bpower(alpha=alphaB,beta=betaB,delta=delta)),
    'pow': lambda alphaA:MeanABError(A=Apower(alphaA)),
    'Bengio':lambda :divergence(),
}
key = LOSS.split(',')[0]
param = [eval(x) for x in LOSS.split(',')[1:]]
flow.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=loss_fn_dic[key](*param)
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(folder,'model{epoch:03d}.ckpt'),
    verbose=1,
    save_weights_only=True,
    save_freq=STEP_PER_EPOCH
    )

if SIZE<=10:
    from itertools import permutations
    states = np.array(list(permutations(np.arange(SIZE)))).astype('float32')
    total_R = flow.reward(states)
    with open(os.path.join(folder,'total_R'),'wb') as f:
        np.save(f,total_R)

Replay = ReplayBuffer(
    model=flow,
    logger=logger,
    folder=folder,
    load=LOAD,
)
logger.info('MEMORY_LIMIT: %s'% MEMORY_LIMIT)

for i in range(BATCH_MEMORY):
    flow.gen_path_aux(exploration=EXPLORATION)

T = datetime.now()
logger.info('Training starts')
logger.handlers[0].flush()
start_epoch = len(Replay.InitVals)

print(flow.graph.actions)
skip = 0
for i in range(start_epoch,EPOCHS):
    if i == start_epoch + 1 :
        T = datetime.now()
        skip = BASELINE
    logger.info('EPOCH : %s/%s' % (i,EPOCHS) )
    if not skip:
        flow.fit(
            np.zeros(1),
            np.zeros(1),
            initial_epoch=i*STEP_PER_EPOCH,
            epochs=i*STEP_PER_EPOCH+STEP_PER_EPOCH,
            verbose=0,
            batch_size=1,
            callbacks=[
                Replay,
                cp_callback,
                Scheduler[SCHEDULER]
            ]
        )
    else:
        Replay.model = flow
        Replay.on_train_begin()
        Replay.on_train_end()
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
