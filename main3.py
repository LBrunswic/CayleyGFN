import itertools
import multiprocessing
import os

import numpy as np
from tensorboard.plugins.hparams import api as hp
import time



seq_param = 0 # int(sys.argv[1])
seed_repeat = 3 #int(sys.argv[2])


series_name = 'test118'

POOL_SIZE = 3

# PARAMETER_SEARCH = ['reg_fn_alpha']


#_____________PROBLEM DEFINITION___________________
## __graph__
SIZE = 15
GENERATORS = 'cycles_a'
INITIAL_POSITION = 'SymmetricUniform'
INVERSE = True
## __reward__
reward_base_fn = 'TwistedManhattan'
reward_param = {
    'width':2,
    'scale':-100,
    'exp':True,
    'mini':0.
}
## __heuristic__
heuristic_base_fn = 'R_zero'
heuristic_param = {}





# __MODEL__

FlowEstimator_options = {
    'options': {
        'kernel_depth' : 2,
        'width' : 64,
        'final_activation' : 'linear',
    },
    'kernel_options': {

        'kernel_initializer' : 'Orthogonal',
        # 'activation': 'relu',
        'activation': 'tanh',
    }
}

FE_OPT = hp.HParam('flowestimator_opt',hp.Discrete([
    str(FlowEstimator_options)
]))


#___________TRAINING HP______________
GRAD_BATCH_SIZE = hp.HParam('grad_batch_size', hp.Discrete([1]))
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1024]))
LENGTH_CUTOFF_FACTOR = hp.HParam('length_cutoff_factor', hp.Discrete([2]))
INIT_FLOW = hp.HParam('initial_flow', hp.Discrete([1e-3]))
LR = hp.HParam('learning_rate', hp.Discrete([1e-2]))
LR_SCHEDULE = hp.HParam('lr_schedule', hp.Discrete(['none','one','two'][2:]))
EPOCHS = hp.HParam('epochs', hp.Discrete([30]))
STEP_PER_EPOCH = hp.HParam('step_per_epoch', hp.Discrete([5]))
OPTIMIZER = hp.HParam('optimizer',hp.Discrete(['Adam','Nesterov'][:1]))
REWARD_RESCALE = hp.HParam('reward_rescale',hp.Discrete(['Trivial','ERew'][:1]))

##__loss_base__
LOSS = hp.HParam('loss_base', hp.Discrete(['Apower']))
LOSS_ALPHA = hp.HParam('loss_alpha', hp.Discrete([2.]))
LOSS_CUT = hp.HParam('loss_cutoff', hp.Discrete(['none','non_zero'][:1]))
##__loss_normalization__
NORMALIZATION_FN = hp.HParam('normalization_fn', hp.Discrete([0]))
NORMALIZATION_NU_FN = hp.HParam('normalization_nu_fn', hp.Discrete([0]))
##__loss_regularization__
# B_BETA = hp.HParam('B_beta', hp.RealInterval(-3.5,-3.))
B_BETA = hp.HParam('B_beta', hp.Discrete([-1000.]))
REG_FN_GEN = hp.HParam('reg_fn_gen', hp.Discrete(['norm2','LogPathLen','PathAccuracy'][:2]))
# REG_FN_alpha = hp.HParam('reg_fn_alpha', hp.Discrete([-1000.]))
REG_FN_alpha = hp.HParam('reg_fn_alpha', hp.RealInterval(-20.,10.))
REG_FN_alpha_SCHEDULE = hp.HParam('reg_fn_alpha_schedule', hp.Discrete(['none','one','two','short'][3:]))
REG_FN_logpmin = hp.HParam('reg_fn_logmin', hp.Discrete([5]))
REG_PROJ = hp.HParam('reg_proj', hp.Discrete(['OrthReg','AddReg']))

PATH_REDRAW = hp.HParam('path_redraw', hp.Discrete([0]))
NEIGHBORHOOD = hp.HParam('neighborhood', hp.Discrete([0]))

SEED = hp.HParam('seed',hp.RealInterval(0.,1000000.))

##_____Problem def HP____
GRAPH_SIZE = hp.HParam('graph_size', hp.Discrete([SIZE]))
GRAPH_GENERATORS = hp.HParam('graph_generators', hp.Discrete([GENERATORS]))
GRAPH_INVERSE = hp.HParam('inverse',hp.Discrete([INVERSE]))
GRAPH_INITIAL_POSITION = hp.HParam('initial_pos',hp.Discrete([INITIAL_POSITION]))
REW_FN =  hp.HParam('rew_fn', hp.Discrete(['TwistedManhattan']))
REW_PARAM = hp.HParam('rew_param', hp.Discrete([str(reward_param)]))
REW_FACTOR = hp.HParam('rew_factor', hp.Discrete([1.,10.,100.][:1]))
HEU_FN =  hp.HParam('heuristic_fn', hp.Discrete([heuristic_base_fn]))
HEU_PARAM = hp.HParam('heuristic_param', hp.Discrete([str(heuristic_param)]))
HEU_FACTOR = hp.HParam('heuristic_factor', hp.Discrete([0.]))


EMBEDDING = hp.HParam('embedding',hp.Discrete(
    [
        '_'.join(x) for x in
        [
            # ('hot',), # direct
            # ('hot','cos','sin','natural'), #full
            ('cos','sin','natural'), #light
            # ('cos','sin',), #very-light
        ]
    ]
))

SAMPLE_SIZE = 100

log_dir = 'logs/S%s_%s_%s_%s/' % (SIZE,GENERATORS,series_name,seq_param)



HP = [NORMALIZATION_FN,NORMALIZATION_NU_FN,REG_FN_alpha,REG_PROJ,REG_FN_logpmin,GRAD_BATCH_SIZE,BATCH_SIZE, LENGTH_CUTOFF_FACTOR, INIT_FLOW,LR,EPOCHS,STEP_PER_EPOCH,B_BETA,PATH_REDRAW,NEIGHBORHOOD,
      GRAPH_SIZE,GRAPH_GENERATORS,GRAPH_INVERSE,SEED,EMBEDDING,OPTIMIZER,LOSS,LOSS_ALPHA,LR_SCHEDULE,REG_FN_alpha_SCHEDULE,
      FE_OPT,REW_FN,REW_PARAM,REW_FACTOR,GRAPH_INITIAL_POSITION,HEU_FN,HEU_FACTOR,HEU_PARAM,REWARD_RESCALE,REG_FN_GEN,LOSS_CUT]

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

    from metrics import FlowSizeStop, ReplayBuffer,PathAccuracy, ExpectedLen, ExpectedMaxSeenReward, MaxSeenReward, ExpectedReward, FlowSize, RandomCheck,PathLeak

    Metrics = [
        ExpectedReward,
        ExpectedMaxSeenReward,
        MaxSeenReward,
        ExpectedLen,
        FlowSize,
        RandomCheck,
        lambda: tf.keras.metrics.Mean(name='initflow'),
        PathLeak,
        PathAccuracy
    ]

    if worker == 1:
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
                hparams=HP,
                metrics=[hp.Metric(m, display_name=m) for m in ["loss"] + [m().name for m in Metrics]],
            )


# worker = 1
def run_wrapper(hparams):
    worker = int(multiprocessing.current_process().name.split('-')[1])

    run_name = "run%s_" % worker + series_name + "-%d" % time()
    from train import train_test_model
    import tensorflow as tf
    run_dir = os.path.join(log_dir, run_name)
    # print(hparams)
    vals = train_test_model(hparams, log_dir=run_dir)
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)
        for name in vals:
            tf.summary.scalar(name, vals[name], step=1)
    tf.keras.backend.clear_session()
    return {name:vals[name].numpy() for name in vals}


def get_values(hyperparam,samples=SAMPLE_SIZE):
    if isinstance(hyperparam,hp.Discrete):
        return hyperparam.values
    elif isinstance(hyperparam,hp.RealInterval):
        return [hyperparam.sample_uniform() for _ in range(samples)]

from multiprocessing import Pool


from time import time

if __name__ == '__main__':

    HP_name = [x.name for x in HP if x not in [REG_FN_alpha,SEED]]
    hp_sets = list(itertools.product(*[get_values(hps.domain) for hps in HP if hps not in [REG_FN_alpha,SEED] ] ))
    hparams_list = [
        {
            HP_name[i]: hp_val
            for i, hp_val in enumerate(hp_set)
        }
        for hp_set in hp_sets
    ]
    seeds = [7508, 9845, 6726, 9339, 8885, 4892, 1347, 5243, 7591, 3661,
           6626, 1007, 3951, 1215, 5597, 6864, 4676, 3922, 2490, 3927, 8842,
           3203, 8428, 3366, 5770, 3226, 6206, 9234, 5521, 4414, 5157, 7775,
           2762, 9248, 7612, 2678, 4066, 6574, 8923, 4360, 9276, 2752, 8101,
           3122, 9882, 8790, 1894, 3426, 8760, 8499, 1964, 8675, 1109, 7910,
           8877, 4682, 1623, 1086, 3062, 1716, 5139, 6708, 6799,4445][:seed_repeat]
    # print(len(seeds))

    with Pool(processes=POOL_SIZE,initializer=initialize) as pool:
        for hparams in hparams_list:
            # search for flow size implosion domain (including instability)
            a,b = REG_FN_alpha.domain.min_value, REG_FN_alpha.domain.max_value
            print(a,b)
            i=0
            c = b
            while b-a > 2.:
                c = (a+b)/2
                print(a,c,b)
                args = [{SEED.name: seed, REG_FN_alpha.name: c, **hparams} for seed in seeds]
                # print(args)
                results = pool.map_async(run_wrapper, args).get()
                # print(results)

                flowsizes = [res['FlowSize'] for res  in results]
                if min(flowsizes) < 1:
                    b = c
                else:
                    a = c
            implosion_domain_min = c
            a,b = REG_FN_alpha.domain.min_value, implosion_domain_min
            n = 10
            i = 0
            former_best = (-1.,-1.)
            while b-a > 0.3:
                i += 1
                delta = np.abs((np.random.rand(2)*(b-a)/(n-1)))
                x = np.linspace(a-delta[0],b+delta[1], n,dtype='float32')
                args = [{SEED.name: seed, REG_FN_alpha.name: alpha, **hparams} for seed,alpha in itertools.product(seeds,x)]
                results = pool.map_async(run_wrapper, args).get()
                indices = {alpha : [i for i,arg in enumerate(args) if arg[REG_FN_alpha.name] == alpha] for alpha in x}
                EMaxSeenRew = [np.mean([results[i]['EMaxSeenRew'] for i in indices[alpha]]) for alpha in x]
                best_index = np.argmax(EMaxSeenRew)
                if EMaxSeenRew[best_index] > former_best[1]:
                    former_best = (x[best_index], EMaxSeenRew[best_index])
                a = (a+former_best[0])/2
                b = (b+former_best[0])/2


print('THATS ALL FOLKS!')