import itertools
import multiprocessing
import os
import re
import numpy as np
from tensorboard.plugins.hparams import api as hp
from time import time
import datetime
LOG = 'test2.log'
try:
    with open(LOG,'r') as f:
        A = ''.join(f.readlines())
        CASE_DONE = max([1]+[int(x) for x in re.findall('(?<=------------HP) (?P<HPcase>[0-9]+) / [0-9]+', A)])-1
except:
    CASE_DONE = 0


with open(LOG,'w') as f:
    f.write('START\n')

seq_param = 0 # int(sys.argv[1])
seed_repeat = 2
# POOL_SIZE =
#int(sys.argv[2])
TEST = True
TEST = False
THRESHOLD_IMPLOSIOM = 10
THRESHOLD_MAX = 0.4
N_SAMPLE = 2

series_name = 'W1_lightE_test'

POOL_SIZE = 1
THREAD_SIZE = 1
MAX_GPU_WORKER = 2


# PARAMETER_SEARCH = ['reg_fn_alpha']


#_____________PROBLEM DEFINITION___________________
## __graph__
SIZE = 15
GENERATORS = [
    # 'trans_cycle_a',
    # 'cycles_a',
    'transpositions'
]
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

FlowEstimator_options = [
    {
    'options': {
        'kernel_depth' : 2,
        'width' : 32,
        'final_activation' : 'linear',
    },
    'kernel_options': {

        'kernel_initializer' : 'Orthogonal',
        # 'activation': 'relu',
        'activation': 'tanh',
    }
},

# {
#     'options': {
#         'kernel_depth' : 3,
#         'width' : 64,
#         'final_activation' : 'linear',
#     },
#     'kernel_options': {
#
#         'kernel_initializer' : 'Orthogonal',
#         'activation': 'tanh',
#     }
# }
]
FE_OPT = hp.HParam('flowestimator_opt',hp.Discrete(
    [str(x) for x in FlowEstimator_options]
))


#___________TRAINING HP______________
GRAD_BATCH_SIZE = hp.HParam('grad_batch_size', hp.Discrete([1]))
BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([1024]))
LENGTH_CUTOFF = hp.HParam('length_cutoff', hp.Discrete([30]))
INIT_FLOW = hp.HParam('initial_flow', hp.Discrete([1e-3]))
LR = hp.HParam('learning_rate', hp.Discrete([5*1e-3]))
LR_SCHEDULE = hp.HParam('lr_schedule', hp.Discrete(['none','one','two'][:1]))
EPOCHS = hp.HParam('epochs', hp.Discrete([10]))
STEP_PER_EPOCH = hp.HParam('step_per_epoch', hp.Discrete([5]))
OPTIMIZER = hp.HParam('optimizer',hp.Discrete(['Adam','Nesterov'][:1]))
REWARD_RESCALE = hp.HParam('reward_rescale',hp.Discrete(['Trivial','ERew'][:1]))

##__loss_base__
LOSS = hp.HParam('loss_base', hp.Discrete(['Apower']))
LOSS_ALPHA = hp.HParam('loss_alpha', hp.Discrete([2.]))
LOSS_CUT = hp.HParam('loss_cutoff', hp.Discrete(['none','non_zero'][:1]))
##__loss_normalization__
NORMALIZATION_FN = hp.HParam('normalization_fn', hp.Discrete([4,5]))
NORMALIZATION_NU_FN = hp.HParam('normalization_nu_fn', hp.Discrete([0,2,3]))
##__loss_regularization__
# B_BETA = hp.HParam('B_beta', hp.RealInterval(-3.5,-3.))
B_BETA = hp.HParam('B_beta', hp.Discrete([-1000.]))
REG_FN_GEN = hp.HParam('reg_fn_gen', hp.Discrete(['norm2','LogPathLen','PathAccuracy'][0:2]))
# REG_FN_alpha = hp.HParam('reg_fn_alpha', hp.Discrete([-1000.]))
REG_FN_alpha = hp.HParam('reg_fn_alpha', hp.RealInterval(-15.,15.))
REG_FN_alpha_SCHEDULE = hp.HParam('reg_fn_alpha_schedule', hp.Discrete(['none','one','two','short'][:1]))
REG_FN_logpmin = hp.HParam('reg_fn_logmin', hp.Discrete([5]))
REG_PROJ = hp.HParam('reg_proj', hp.Discrete(['OrthReg','AddReg']))

PATH_REDRAW = hp.HParam('path_redraw', hp.Discrete([0]))
NEIGHBORHOOD = hp.HParam('neighborhood', hp.Discrete([0]))

SEED = hp.HParam('seed',hp.RealInterval(0.,1000000.))

##_____Problem def HP____
GRAPH_SIZE = hp.HParam('graph_size', hp.Discrete([SIZE]))
GRAPH_GENERATORS = hp.HParam('graph_generators', hp.Discrete(GENERATORS))
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

log_dir = 'logs/S%s_%s_%s_%s/' % (SIZE,'_'.join(GENERATORS),series_name,seq_param)



HP = [NORMALIZATION_FN,NORMALIZATION_NU_FN,REG_FN_alpha,REG_PROJ,REG_FN_logpmin,GRAD_BATCH_SIZE,BATCH_SIZE, LENGTH_CUTOFF, INIT_FLOW,LR,EPOCHS,STEP_PER_EPOCH,B_BETA,PATH_REDRAW,NEIGHBORHOOD,
      GRAPH_SIZE,GRAPH_GENERATORS,GRAPH_INVERSE,SEED,EMBEDDING,OPTIMIZER,LOSS,LOSS_ALPHA,LR_SCHEDULE,REG_FN_alpha_SCHEDULE,
      FE_OPT,REW_FN,REW_PARAM,REW_FACTOR,GRAPH_INITIAL_POSITION,HEU_FN,HEU_FACTOR,HEU_PARAM,REWARD_RESCALE,REG_FN_GEN,LOSS_CUT]

HP_name = [x.name for x in HP]

def initialize():

    import tensorflow as tf
    try:
        worker = int(multiprocessing.current_process().name.split('-')[1])
    except:
        worker = 0
    if worker <= MAX_GPU_WORKER:
        GPU = 0
        gpus = tf.config.list_physical_devices('GPU')
        # print(gpus)
        tf.config.set_visible_devices(gpus[GPU], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[GPU], True)
        # tf.config.threading.set_inter_op_parallelism_threads(16)
        # tf.config.threading.set_intra_op_parallelism_threads(128)
        logical_gpus = tf.config.list_logical_devices('GPU')
            # print('done!', logical_gpus)
    else:
        tf.config.set_visible_devices([], 'GPU')
        # tf.config.threading.set_inter_op_parallelism_threads(2)
        # tf.config.threading.set_inter_op_parallelism_threads(8)

    print(multiprocessing.current_process().name + 'THREAD INTER:',tf.config.threading.get_inter_op_parallelism_threads())
    print(multiprocessing.current_process().name + 'THREAD INTRA:',tf.config.threading.get_intra_op_parallelism_threads())

    # tf.config.experimental.enable_op_determinism()
    import sys

    # sys.stdout = open(str(worker) + ".log", "w")

    from metrics import ExpectedLen, ExpectedMaxSeenReward, MaxSeenReward, ExpectedReward, FlowSize, RandomCheck,PathLeak

    Metrics = [
        ExpectedReward,
        ExpectedMaxSeenReward,
        MaxSeenReward,
        ExpectedLen,
        FlowSize,
        RandomCheck,
        lambda: tf.keras.metrics.Mean(name='initflow'),
        PathLeak,
    ]

    if worker <= 1:
        with tf.summary.create_file_writer(log_dir).as_default():
            hp.hparams_config(
                hparams=HP,
                metrics=[hp.Metric(m, display_name=m) for m in ["loss"] + [m().name for m in Metrics]],
            )


def run_wrapper(hparams):
    T = time()
    worker = int(multiprocessing.current_process().name.split('-')[1])

    run_name = "run%s_" % worker + series_name + "-%d" % time()
    from train import train_test_model
    import tensorflow as tf
    run_dir = os.path.join(log_dir, run_name)
    # print(hparams)
    vals = train_test_model(hparams, log_dir=run_dir,test=TEST)
    if TEST:
        return vals
    for flow in range(POOL_SIZE):
        with tf.summary.create_file_writer(run_dir + 'F%s' % flow).as_default():
            hparams_split = {x:hparams[x] for x in hparams if x!='reg_fn_alpha'}
            hparams_split.update({'reg_fn_alpha' : hparams['reg_fn_alpha'][flow]})
            hp.hparams(hparams_split)
            for name in vals:
                if name=='loss':
                    tf.summary.scalar(name, vals[name], step=1)
                    continue
                tf.summary.scalar(name, vals[name][flow], step=1)

    tf.keras.backend.clear_session()

    return {name:vals[name].numpy() for name in vals}


def get_values(hyperparam):
    if isinstance(hyperparam,hp.Discrete):
        return hyperparam.values
    elif isinstance(hyperparam,hp.RealInterval):
        raise
        # return [hyperparam.sample_uniform() for _ in range(samples)]

from multiprocessing import Pool

class FalsePool():
    def __int__(self,processes=THREAD_SIZE,initializer=initialize):
        initialize()
    def map(self,*args,**kwargs):
        map(*args,**kwargs)
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
    assert(seed_repeat%POOL_SIZE==0)
    seed_repeat = seed_repeat//POOL_SIZE
    seeds = [7508, 9845, 6726, 9339, 8885, 4892, 1347, 5243, 7591, 3661,
           6626, 1007, 3951, 1215, 5597, 6864, 4676, 3922, 2490, 3927, 8842,
           3203, 8428, 3366, 5770, 3226, 6206, 9234, 5521, 4414, 5157, 7775,
           2762, 9248, 7612, 2678, 4066, 6574, 8923, 4360, 9276, 2752, 8101,
           3122, 9882, 8790, 1894, 3426, 8760, 8499, 1964, 8675, 1109, 7910,
           8877, 4682, 1623, 1086, 3062, 1716, 5139, 6708, 6799,4445][:seed_repeat]
    # print(len(seeds))

    a, b = REG_FN_alpha.domain.min_value, REG_FN_alpha.domain.max_value
    with open(LOG, 'a') as f:
        f.write("%s Hyperparameter cases. " %len(hparams_list))
    n_experiments =  seed_repeat*(np.log2((b-a)/THRESHOLD_IMPLOSIOM) +  np.log2((b-a)/THRESHOLD_MAX*N_SAMPLE)*N_SAMPLE + N_SAMPLE)
    with open(LOG, 'a') as f:
        f.write("Each conducting %s experiments. \n" % n_experiments)
    Total_time = time()
    with Pool(processes=THREAD_SIZE,initializer=initialize) as pool:
    # with FalsePool(processes=THREAD_SIZE,initializer=initialize) as pool:
        for hp_case, hparams in enumerate(hparams_list):
            case_time = time()
            experiments_counts = 0
            if hp_case < CASE_DONE:
                continue
            # search for flow size implosion domain (including instability)
            with open(LOG, 'a') as f:
                f.write('\n')
                f.write(str(datetime.datetime.now()))
                f.write("------------HP %s / %s" %(hp_case+1,len(hparams_list)))
            test_time = time()
            with open(LOG, 'a') as f:
                f.write(str(datetime.datetime.now()) + ', phase 1 start\n')
            a,b = REG_FN_alpha.domain.min_value, REG_FN_alpha.domain.max_value
            x = np.linspace(a, b, N_SAMPLE, dtype='float32')
            args = [{'pool_size': POOL_SIZE, SEED.name: seed, REG_FN_alpha.name: tuple([alpha]*POOL_SIZE), **hparams} for seed, alpha in
                    itertools.product(seeds, x)]
            list(pool.map(run_wrapper, args,chunksize=1))

            experiments_counts += len(args)
            EdeltaT = datetime.timedelta(seconds=(time() - test_time) / N_SAMPLE/seed_repeat*(n_experiments-N_SAMPLE*seed_repeat))
            with open(LOG, 'a') as f:
                f.write('. case ETA: %s seconds\n' % (datetime.datetime.now() + EdeltaT))
            i=0
            c = b

            with open(LOG, 'a') as f:
                f.write(str(datetime.datetime.now()) + ', phase 2 start\n')
            while b-a > THRESHOLD_IMPLOSIOM:
                c = (a+b)/2
                args = [
                    {
                        'pool_size': POOL_SIZE,
                        SEED.name: seed,
                        REG_FN_alpha.name: tuple([c]*POOL_SIZE),
                        **hparams
                    }
                    for seed in seeds
                ]
                results = list(pool.map(run_wrapper, args,chunksize=1))
                experiments_counts += len(args)
                flowsizes = np.min([np.min(res['FlowSize']) for res in results])
                if flowsizes < 1e-2:
                    b = c
                else:
                    a = c
            implosion_domain_min = c
            a,b = REG_FN_alpha.domain.min_value, implosion_domain_min

            i = 0
            former_best = (-1.,-1.)
            with open(LOG, 'a') as f:
                f.write(str(datetime.datetime.now()) + ', phase 3 start\n')
            while b-a > THRESHOLD_MAX:

                i += 1
                delta = np.abs((np.random.rand(2)*(b-a)/(N_SAMPLE-1)))
                x = np.linspace(a-delta[0],b+delta[1], N_SAMPLE, dtype='float32')
                args = [{'pool_size': POOL_SIZE, SEED.name: seed, REG_FN_alpha.name: tuple([alpha]*POOL_SIZE), **hparams} for seed,alpha in itertools.product(seeds,x)]
                packed_results = list(pool.map(run_wrapper, args,chunksize=1))
                experiments_counts += len(args)
                results = []
                for flow_index in range(POOL_SIZE):
                    new_results =  [({name: res[name][flow_index] for name in res if name != 'loss'},{'loss':res['loss']}) for res in packed_results]
                    for exp_index,(most_res,loss_res) in enumerate(new_results):
                        most_res.update(loss_res)
                        most_res.update({'flow_rank': flow_index})
                        most_res.update({'args': args[exp_index]})
                        results.append(most_res)
                indices = {alpha : [i for i,res in enumerate(results) if res['args'][REG_FN_alpha.name][0] == alpha] for alpha in x}
                # with open(LOG, 'a') as f:
                #     f.write('\n'+str(indices))
                    # f.write('ETA: %s seconds\n' % (datetime.datetime.now() + EdeltaT))
                EMaxSeenRew = [np.mean([results[i]['EMaxSeenRew'] for i in indices[alpha]]) for alpha in x]
                best_index = np.argmax(EMaxSeenRew)
                if EMaxSeenRew[best_index] > former_best[1]:
                    former_best = (x[best_index], EMaxSeenRew[best_index])
                a = (a+former_best[0])/2
                b = (b+former_best[0])/2
            deltat = time() - Total_time
            deltaT = datetime.timedelta(seconds=deltat)
            EdeltaT = datetime.timedelta(seconds= (time() - Total_time)/(hp_case+1)* (len(hparams_list)-hp_case-1) )
            with open(LOG,'a') as f:
                f.write('Elapsed time: %s seconds.' % (deltaT))
                f.write(' Experiments per hours %s:' % (experiments_counts / (time()-case_time) * 3600 * POOL_SIZE ))
                f.write('ETA: %s seconds\n' % (datetime.datetime.now()+EdeltaT))
with open(LOG, 'a') as f:
    f.write('THATS ALL FOLKS!\n')
