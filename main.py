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
    tf.config.threading.set_intra_op_parallelism_threads(16)
    tf.config.set_visible_devices([], 'GPU')

import platform
import numpy as np
import time
time.sleep(1)
# time.sleep(13000)
from datetime import datetime
from Graphs.CayleyGraph import Symmetric, path_representation
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from GFlowBase.regularization import *
from Groups.symmetric_groups import SymmetricUniform,Modal,rubick_generators,inversion,iteration_random
from metrics import ReplayBuffer,ExpectedLen,ExpectedMaxSeenReward,MaxSeenReward,ExpectedReward,FlowSize,RandomCheck
from datetime import datetime
from tensorflow.python.platform import tf_logging as logging
import logging
import signal
import pickle
import sys

seq_param = 0 # int(sys.argv[1])
seed = 0 #int(sys.argv[2])


series_name = 'test'

POOL_SIZE = 3

#_____________PROBLEM DEFINITION___________________
## __graph__
SIZE = 5
GENERATORS = 'transpositions'
INITIAL_POSITION = SymmetricUniform(SIZE)
INVERSE = True
## __reward__
reward_base_fn = 'TwistedManhattan'
reward_param = {
    'width':1,
    'scale':-100,
    'exp':False,
    'mini':0.
}
## __heuristic__
heuristic_base_fn = 'R_zero'
heuristic_param = {}





# __MODEL__

FlowEstimator_options = {
    'options': {
        'kernel_depth' : 0,
        'width' : 64,
        'final_activation' : 'linear',
    },
    'kernel_options': {

        'kernel_initializer' : 'Orthogonal',
        'activation': 'tanh',
        # 'activation': 'linear',
    }
}

FE_OPT = hp.HParam('flowestimator_opt',hp.Discrete([
    str(FlowEstimator_options)
]))





#___________TRAINING HP______________
GRAD_BATCH_SIZE = hp.HParam('grad_batch_size', hp.Discrete([1]))
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64]))
LENGTH_CUTOFF_FACTOR = hp.HParam('length_cutoff_factor', hp.Discrete([10]))
INIT_FLOW = hp.HParam('initial_flow', hp.Discrete([1e-3]))
LR = hp.HParam('learning_rate', hp.Discrete([1e-3]))
EPOCHS = hp.HParam('number_epoch', hp.Discrete([10]))
STEP_PER_EPOCH = hp.HParam('step_per_epoch', hp.Discrete([20]))
OPTIMIZER = hp.HParam('optimizer',hp.Discrete(['Adam','Nesterov']))

##__loss_base__
LOSS = hp.HParam('loss_base', hp.Discrete(['Apower']))
LOSS_ALPHA = hp.HParam('loss_alpha', hp.Discrete([2.]))
##__loss_normalization__
NORMALIZATION_FN = hp.HParam('normalization_fn', hp.Discrete([0,4]))
NORMALIZATION_NU_FN = hp.HParam('normalization_nu_fn', hp.Discrete([0,2]))
##__loss_regularization__
B_BETA = hp.HParam('B_beta', hp.Discrete([0.]))
REG_FN_alpha = hp.HParam('reg_fn_alpha', hp.RealInterval(-10.,10.))
REG_FN_logpmin = hp.HParam('reg_fn_logmin', hp.Discrete([25]))
REG_PROJ = hp.HParam('reg_proj', hp.Discrete(['OrthReg','AddReg'][1:]))

PATH_REDRAW = hp.HParam('path_redraw', hp.Discrete([0]))
NEIGHBORHOOD = hp.HParam('neighborhood', hp.Discrete([0]))

SEED = hp.HParam('seed',hp.Discrete([7508, 9845, 6726, 9339, 8885, 4892, 1347, 5243, 7591, 3661,
           6626, 1007, 3951, 1215, 5597, 6864, 4676, 3922, 2490, 3927, 8842,
           3203, 8428, 3366, 5770, 3226, 6206, 9234, 5521, 4414, 5157, 7775,
           2762, 9248, 7612, 2678, 4066, 6574, 8923, 4360, 9276, 2752, 8101,
           3122, 9882, 8790, 1894, 3426, 8760, 8499, 1964, 8675, 1109, 7910,
           8877, 4682, 1623, 1086, 3062, 1716, 5139, 6708, 6799,4445][seed:seed+1]))

##_____Problem def HP____
GRAPH_SIZE = hp.HParam('graph_size', hp.Discrete([SIZE]))
GRAPH_GENERATORS = hp.HParam('graph_generators', hp.Discrete([GENERATORS]))
GRAPH_INVERSE = hp.HParam('inverse',hp.Discrete([INVERSE]))
REW_FN =  hp.HParam('rew_fn', hp.Discrete(['TwistedManhattan']))
REW_PARAM = hp.HParam('rew_param', hp.Discrete([str(reward_param)]))
REW_FACTOR = hp.HParam('rew_factor', hp.Discrete([0.1,1.,10.,100.]))


EMBEDDING = hp.HParam('embedding',hp.Discrete(
    [
        '_'.join(x) for x in
        [
            ('hot',), # direct
            ('hot','cos','sin','natural'), #full
            ('cos','sin','natural'), #light
        ][:1]
    ]
))



reg_post_choices = {
    'OrthReg' : proj_reg,
    'AddReg' :  straight_reg,
}

SAMPLE_SIZE = 400




HP = [NORMALIZATION_FN,NORMALIZATION_NU_FN,REG_FN_alpha,REG_PROJ,REG_FN_logpmin,GRAD_BATCH_SIZE,BATCH_SIZE, LENGTH_CUTOFF_FACTOR, INIT_FLOW,LR,EPOCHS,STEP_PER_EPOCH,B_BETA,PATH_REDRAW,NEIGHBORHOOD,
      GRAPH_SIZE,GRAPH_GENERATORS,GRAPH_INVERSE,FE_OPT,SEED,EMBEDDING,OPTIMIZER,LOSS,LOSS_ALPHA,
      FE_OPT,REW_FN,REW_PARAM,REW_FACTOR]

log_dir = 'logs/S%s_%s_%s_%s/' % (SIZE,GENERATORS,series_name,seq_param)

normalization_fns =[
    None,
    lambda flownu,finit: 1e-3 + tf.reduce_sum(flownu[..., :4], axis=-1) / 2,
    lambda flownu,finit: 1e-3 + flownu[...,1],
    lambda flownu,finit: 1e-3 + finit,
    lambda flownu,finit: 1e-3 + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1)) ,
    lambda flownu,finit: 1e-3 + (finit+tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1))),
    lambda flownu,finit: 1e-3 + 10*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1)) ,
    # lambda flownu,finit: 1e-3 + (finit+tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1))),
]

normalization_nu_fns =[
    None,
    lambda flownu,finit: tf.reduce_mean(tf.reduce_sum(tf.math.exp(flownu[..., 4]),axis=1)),
    lambda flownu,finit: tf.reduce_sum(tf.math.exp(flownu[..., 4]),axis=1,keepdims=True)
]

optimizers = {
        'Adam' : lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
        'Nesterov' : lambda lr: tf.keras.optimizers.SGD(learning_rate=lr,nesterov=True),
    }

Metrics = [
        ExpectedReward,
        ExpectedMaxSeenReward,
        MaxSeenReward,
        ExpectedLen,
        FlowSize,
        RandomCheck,
        lambda :tf.keras.metrics.Mean(name='initflow')
    ]


class FlowSizeStop(tf.keras.callbacks.Callback):
    def __init__(self, monitor="FlowSize", min_val=1e-3,max_val=1e4):
        super().__init__()
        self.monitor = monitor
        self.min_val = min_val
        self.max_val = max_val
        self.stopped = False

    def on_train_begin(self, logs=None):
        self.stopped = False
    def on_epoch_end(self, epoch, logs=None):
        if self.get_monitor_value(logs) < self.min_val or self.get_monitor_value(logs) > self.max_val:
            self.model.stop_training = True
            self.stopped = True

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value
def train_test_model(hparams,log_dir=None):
    if log_dir is None:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.keras.utils.set_random_seed(hparams[SEED])
    G = Symmetric(
        SIZE,
        generators=GENERATORS,
        representation=[('natural', {})],
        inverse=INVERSE,
        random_gen=INITIAL_POSITION,
        embedding=[ (x,{}) for x in  hparams[EMBEDDING].split('_')],
        dtype='float32'
    )
    if hparams[B_BETA]>0:
        B = Bpower(beta=tf.math.exp(hparams[B_BETA]))
    else:
        B = lambda x,y:1
    print(eval(hparams[LOSS])(alpha=hparams[LOSS_ALPHA]))
    loss_fn = MeanABError(
        A=eval(hparams[LOSS])(alpha=hparams[LOSS_ALPHA]),
        B=B,
        normalization_fn=normalization_fns[hparams[NORMALIZATION_FN]],
        normalization_nu_fn=normalization_nu_fns[hparams[NORMALIZATION_NU_FN]],
    )
    reg_fn = reg_fn_gen(
        alpha=hparams[REG_FN_alpha],
        logpmin=hparams[REG_FN_logpmin],
    )
    reg_post = reg_post_choices[hparams[REG_PROJ]]
    reward_fn = eval(hparams[REW_FN])(SIZE, factor=hparams[REW_FACTOR], **eval(hparams[REW_PARAM]))
    heuristic_fn = eval(heuristic_base_fn)(SIZE, **heuristic_param)

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
        reg_fn=reg_fn,
        grad_batch_size=hparams[GRAD_BATCH_SIZE],
        # reg_fn=reg_withcutoff_fn
    )
    flow(0)
    # flow.FlowEstimator.kernel.summary()


    Replay = ReplayBuffer()
    Replay.reward = reward_fn


    for m in Metrics:
        # print(m)
        flow.metric_list.append(m())


    flow.compile(
        optimizer=optimizers[hparams[OPTIMIZER]](hparams[LR]),
        loss=loss_fn,
    )
    print(Metrics)
    print(flow.metrics)



    flow.fit(
        np.zeros(hparams[STEP_PER_EPOCH]),
        initial_epoch=0,
        epochs=hparams[EPOCHS],
        verbose=1,
        batch_size=1,
        callbacks=[
            FlowSizeStop(),
            tf.keras.callbacks.TerminateOnNaN(),
            Replay,
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            )
        ]
    )
    # metrics = flow.metrics
    return flow.evaluate(),flow.metrics

print(Metrics)
with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=HP,
        metrics=[hp.Metric(m, display_name=m) for m in ["loss"]+[m().name for m in Metrics] ],
    )

def run(run_dir, hparams,seed=1234):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    vals,metrics = train_test_model(hparams, log_dir=run_dir)
    val_name_list = zip(vals,[m.name for m in metrics])
    for val, name in val_name_list:
        tf.summary.scalar(name, val, step=1)







session_num = 0

# hp_names = [hp.name for hp in HP]
def get_values(hyperparam,samples=SAMPLE_SIZE):
    if isinstance(hyperparam,hp.Discrete):
        return hyperparam.values
    elif isinstance(hyperparam,hp.RealInterval):
        return [hyperparam.sample_uniform() for _ in range(samples)]


hp_sets = itertools.product(*[get_values(hps.domain) for hps in HP])

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
    tf.keras.backend.clear_session()

