import os
import tensorflow as tf
import numpy as np
from Graphs.CayleyGraph import Symmetric
from GFlowBase.kernel import dense_gen
from GFlowBase.rewards import Reward
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import MeanABError
from metrics import ReplayBuffer
from datetime import datetime
import logging
import sys

def set_gpu(gpu=-1, memory_limit=2**12):
    if gpu >= 0:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(gpus[gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[gpu], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print('done!', logical_gpus)
        except Exception as e:
            print(e)


def set_seed(seed):
    tf.random.set_seed(seed)


def set_folder(size, folder_add):
    FOLDER_NAME = 'graphS%s' % size + folder_add
    folder = os.path.join('tests/', FOLDER_NAME)
    os.makedirs(os.path.join('tests', FOLDER_NAME), exist_ok=True)
    return FOLDER_NAME, folder


def set_logger(folder):
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(folder, 'training.log'), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_pid(folder):
    PID_PATH = os.path.join(folder, 'PID')
    with open(PID_PATH, 'w') as f:
        f.write(str(os.getpid()))
    return str(os.getpid())




def exp_scheduler(epoch, lr):
  if epoch < 1000:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


def cosine_scheduler(steps,LR):
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
]

DEFAULT_TRAINING_PARAM = {
    'SIZE': 3,
    'GENERATORS': 'trans_cycle_a',
    'EPOCHS': 100,
    'BATCH_SIZE': 4,
    'folder': set_folder(3, '_default'),
    'STEP_PER_EPOCH': 100,
    'MLP_DEPTH': 4,
    'MLP_WIDTH': 128,
    'INVERSE': 0,
    'MEMORY_LIMIT': 2**12,
    'LR': 1e-2,
    'LOAD': 0,
    'LOSS': MeanABError(),
    'SEED': 1234,
    'BATCH_MEMORY': 4,
    'HEURISTIC_FN': lambda x: np.zeros(x.shape[0], dtype='float32'),
    'REWARD_FN': lambda x: np.ones(x.shape[0], dtype='float32'),
    'LENGTH_CUTOFF_FACTOR': 4,
    'OPTIMIZER': tf.optimizers.Adam,
    'SCHEDULER':
}


def create_model(TRAINING_PARAM=None):
    if TRAINING_PARAM is None:
        TRAINING_PARAM = DEFAULT_TRAINING_PARAM

    G = Symmetric(
        TRAINING_PARAM['SIZE'],
        Gen=TRAINING_PARAM['GENERATORS'],
        inverse=TRAINING_PARAM['INVERSE'],
        k=1,
    )

    FlowEstimator = [
        dense_gen,
    ]

    FlowEstimator_options = {
        'options': {
            'kernel_depth': TRAINING_PARAM['MLP_DEPTH'],
            'width': TRAINING_PARAM['MLP_WIDTH'],
            'final_activation': 'linear',
        },
        'kernel_options': {
            'activation': tf.keras.layers.LeakyReLU(),
            'kernel_initializer': tf.keras.initializers.HeNormal(),
        }
    }
    flow = GFlowCayleyLinear(
        graph=G,
        # reward=R_one,
        # reward=R_main,
        reward=Reward(
            reward_fn=TRAINING_PARAM['REWARD_FN'],
            heuristic_fn=TRAINING_PARAM['HEURISTIC_FN'],
        ),
        batch_size=TRAINING_PARAM['BATCH_SIZE'],
        batch_memory=TRAINING_PARAM['BATCH_MEMORY'],
        FlowEstimatorGen=(FlowEstimator[0], FlowEstimator_options),
        length_cutoff_factor=TRAINING_PARAM['LENGTH_CUTOFF_FACTOR'],
    )
    X = tf.zeros((TRAINING_PARAM['BATCH_SIZE'], (1+G.nactions), flow.embedding_dim))
    flow(X)
    if TRAINING_PARAM['LOAD']:
        flow.load_weights(os.path.join(TRAINING_PARAM['folder'], 'model.ckpt'))
    return flow


def train(
    flow,
    TRAINING_PARAM=None
):
    if TRAINING_PARAM is None:
        TRAINING_PARAM = DEFAULT_TRAINING_PARAM
    flow.compile(
        optimizer=TRAINING_PARAM['OPTIMIZER'](learning_rate=TRAINING_PARAM['LR']),
        loss=TRAINING_PARAM['LOSS']
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(TRAINING_PARAM['folder'], 'model.ckpt'),
        verbose=1,
        save_weights_only=True)
    logger = set_logger(TRAINING_PARAM['folder'])

    Replay = ReplayBuffer(
        model=flow,
        epoch_length=TRAINING_PARAM['STEP_PER_EPOCH'],
        logger=logger,
        folder=TRAINING_PARAM['folder'],
        load=TRAINING_PARAM['LOAD'],
    )
    logger.info('MEMORY_LIMIT: %s' % TRAINING_PARAM['MEMORY_LIMIT'])
    for i in range(TRAINING_PARAM['BATCH_MEMORY']):
        Replay.gen_path(model=flow)

    T = datetime.now()
    logger.info('Training starts')
    logger.handlers[0].flush()
    start_epoch = len(Replay.InitVals)
    for i in range(start_epoch, TRAINING_PARAM['EPOCHS']):
        if i == start_epoch + 1:
            T = datetime.now()
        logger.info('EPOCH : %s/%s' % (i, TRAINING_PARAM['EPOCHS']))
        dataset, mu_initial_reward = Replay.train_data(model=flow)
        logger.info('reward : \n %s' % mu_initial_reward[:5, 2])
        flow.fit(
            dataset,
            mu_initial_reward,
            initial_epoch=i*TRAINING_PARAM['STEP_PER_EPOCH'],
            epochs=i*TRAINING_PARAM['STEP_PER_EPOCH']+TRAINING_PARAM['STEP_PER_EPOCH'],
            verbose=2,
            batch_size=dataset.shape[0],
            shuffle=False,
            callbacks=[
                Replay,
                cp_callback,
                TRAINING_PARAM['SCHEDULER']
            ]
        )
        logger.info('Initial Flow: %s' % float(flow.initial_flow))
        T2 = datetime.now()
        if i >= start_epoch+1:
            logger.info('ETA : %s' % (T2 + (T2-T)*(TRAINING_PARAM['EPOCHS']-i-start_epoch-1)/(i-start_epoch)))
        logger.handlers[0].flush()

    logger.info('All done! Bye bye!')
    logger.handlers[0].flush()


def exec(param=None,EXEC_NUMB=0):
    if param is None:
        TRAINING_PARAM = DEFAULT_TRAINING_PARAM
    else:
        TRAINING_PARAM = {
            'SIZE': 3,
            'GENERATORS': 'trans_cycle_a',
            'EPOCHS': 100,
            'BATCH_SIZE': 4,
            'folder': set_folder(3, '_default'),
            'STEP_PER_EPOCH': 100,
            'MLP_DEPTH': 4,
            'MLP_WIDTH': 128,
            'INVERSE': 0,
            'MEMORY_LIMIT': 2 ** 12,
            'LR': 1e-2,
            'LOAD': 0,
            'LOSS': MeanABError(),
            'SEED': 1234,
            'BATCH_MEMORY': 4,
            'HEURISTIC_FN': lambda x: np.zeros(x.shape[0], dtype='float32'),
            'REWARD_FN': lambda x: np.ones(x.shape[0], dtype='float32'),
            'LENGTH_CUTOFF_FACTOR': 4,
            'OPTIMIZER': tf.optimizers.Adam
        }

    set_gpu(param['GPU'])
    param['folder'] = '_%s' % EXEC_NUMB
    set_folder(param['folder'])
    with open(os.path.join(param['folder'],'HP_dict'), "w") as hp_file:
        hp_file.write(str(param))
    flow = create_model(TRAINING_PARAM=param)
    train(flow,TRAINING_PARAM=param)

