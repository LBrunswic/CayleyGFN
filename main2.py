import itertools
import multiprocessing
import os
from tensorboard.plugins.hparams import api as hp
import time



seq_param = 0 # int(sys.argv[1])
seed = 0 #int(sys.argv[2])


series_name = 'test'

POOL_SIZE = 3

#_____________PROBLEM DEFINITION___________________
## __graph__
SIZE = 5
GENERATORS = 'transpositions'
INITIAL_POSITION = 'SymmetricUniform'
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
EPOCHS = hp.HParam('epochs', hp.Discrete([10]))
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
GRAPH_INITIAL_POSITION = hp.HParam('initial_pos',hp.Discrete([INITIAL_POSITION]))
REW_FN =  hp.HParam('rew_fn', hp.Discrete(['TwistedManhattan']))
REW_PARAM = hp.HParam('rew_param', hp.Discrete([str(reward_param)]))
REW_FACTOR = hp.HParam('rew_factor', hp.Discrete([0.1,1.,10.,100.]))
HEU_FN =  hp.HParam('heuristic_fn', hp.Discrete([heuristic_base_fn]))
HEU_PARAM = hp.HParam('heuristic_param', hp.Discrete([str(heuristic_param)]))
HEU_FACTOR = hp.HParam('heuristic_factor', hp.Discrete([0.]))


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




SAMPLE_SIZE = 400


log_dir = 'logs/S%s_%s_%s_%s/' % (SIZE,GENERATORS,series_name,seq_param)



HP = [NORMALIZATION_FN,NORMALIZATION_NU_FN,REG_FN_alpha,REG_PROJ,REG_FN_logpmin,GRAD_BATCH_SIZE,BATCH_SIZE, LENGTH_CUTOFF_FACTOR, INIT_FLOW,LR,EPOCHS,STEP_PER_EPOCH,B_BETA,PATH_REDRAW,NEIGHBORHOOD,
      GRAPH_SIZE,GRAPH_GENERATORS,GRAPH_INVERSE,FE_OPT,SEED,EMBEDDING,OPTIMIZER,LOSS,LOSS_ALPHA,
      FE_OPT,REW_FN,REW_PARAM,REW_FACTOR,GRAPH_INITIAL_POSITION,HEU_FN,HEU_FACTOR,HEU_PARAM]

HP_name = [x.name for x in HP]

def initialize():

    import tensorflow as tf
    GPU = 0
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    tf.config.set_visible_devices(gpus[GPU], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU], True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print('done!', logical_gpus)
    tf.config.experimental.enable_op_determinism()
    import sys


    worker = int(multiprocessing.current_process().name.split('-')[1])
    sys.stdout = open(str(worker) + ".log", "w")

    from metrics import FlowSizeStop, ReplayBuffer, ExpectedLen, ExpectedMaxSeenReward, MaxSeenReward, ExpectedReward, FlowSize, RandomCheck

    Metrics = [
        ExpectedReward,
        ExpectedMaxSeenReward,
        MaxSeenReward,
        ExpectedLen,
        FlowSize,
        RandomCheck,
        lambda: tf.keras.metrics.Mean(name='initflow')
    ]

    if worker == 1:
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
                hparams=HP,
                metrics=[hp.Metric(m, display_name=m) for m in ["loss"] + [m().name for m in Metrics]],
            )


# worker = 1
def run_wrapper(inputs):
    hp_set,run_number,hp_sets_names = inputs
    hp_sets, HPname = hp_sets_names
    hparams = {
        HPname[i]: hp_val
        for i, hp_val in enumerate(hp_set)

    }
    worker = int(multiprocessing.current_process().name.split('-')[1])

    run_name = "run%s_" % worker + series_name + "-%d" % time()
    from train import train_test_model
    import tensorflow as tf
    run_dir = os.path.join(log_dir, run_name)
    vals,metrics = train_test_model(hparams, log_dir=run_dir)
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        val_name_list = zip(vals,[m.name for m in metrics])
        for val, name in val_name_list:
            tf.summary.scalar(name, val, step=1)

    tf.keras.backend.clear_session()
    print(HP)



def get_values(hyperparam,samples=SAMPLE_SIZE):
    if isinstance(hyperparam,hp.Discrete):
        return hyperparam.values
    elif isinstance(hyperparam,hp.RealInterval):
        return [hyperparam.sample_uniform() for _ in range(samples)]

from multiprocessing import Pool


from time import time

if __name__ == '__main__':
    hp_sets = list(itertools.product(*[get_values(hps.domain) for hps in HP]))
    run_number = range(len(hp_sets))
    args = list(zip(hp_sets, run_number, [(hp_sets,HP_name)] * len(hp_sets)))

    # initialize()
    # for x in args:
    #     run_wrapper(x)
    with Pool(processes=POOL_SIZE,initializer=initialize) as pool:
        pool.map_async(run_wrapper,args).get()


