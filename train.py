import os.path

from Groups.symmetric_groups import SymmetricUniform
from Graphs.CayleyGraph import Symmetric
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import  MultiGFlowCayleyLinear
from GFlowBase.losses import MeanABError,Apower, Bpower,cutoff_fns
from GFlowBase.rewards import Reward
from GFlowBase.regularization import reg_post_choices,reg_fn_gen_choices
from TestingEnv import ReplayBuffer,FlowSizeStop,tuning_method,metrics,PandasRecord,MemoryUse#, LogProgress
import tensorflow as tf
from datetime import datetime
from time import time
from FlowNorm import normalization_nu_fns,normalization_flow_fns,normalization_reward_fns
from Hyperparameters.optimizer import optimizers,lr_schedule
import itertools
from copy import copy
import pandas
from logger_config import log_dict,log_tensorlist


def model_gen(hparams,logger):
    logger.info('Generate Model')
    group_dtype = hparams['group_dtype']
    # hparams.update({'group_dtype' : group_dtype})
    G = Symmetric(
        hparams['graph_size'],
        generators=hparams['graph_generators'],
        representation=[('natural', {})],
        inverse=hparams['initial_pos'],
        random_gen=eval(hparams['initial_pos'])(hparams['graph_size'], group_dtype=group_dtype),
        embedding=[(x, {}) for x in hparams['embedding']],
        group_dtype=group_dtype
    )
    logger.debug('G: %s' % str(G))



    reg_post = reg_post_choices[hparams['reg_proj']]
    reward_fn = hparams['rew_fn']
    reward_args = (hparams['graph_size'],)
    reward_kwargs = {'factor': hparams['rew_factor'], **hparams['rew_param']}
    heuristic_fn = hparams['heuristic_fn']
    heuristic_args = (hparams['graph_size'],)
    heuristic_kwargs = {'factor': hparams['heuristic_factor'], **hparams['heuristic_param']}
    reg_fn_gen = reg_fn_gen_choices[hparams['reg_fn_gen']]
    reg_fn = reg_fn_gen(
        alpha=(0.,) * hparams['pool_size'],
        logpmin=hparams['reg_fn_logmin'],
    )
    logger.debug('reg_fn: %s' % reg_fn)

    flow = MultiGFlowCayleyLinear(
        graph=G,
        reward=Reward(
            reward_fn=reward_fn,
            reward_args=reward_args,
            reward_kwargs=reward_kwargs,
            heuristic_fn=heuristic_fn,
            heuristic_args=heuristic_args,
            heuristic_kwargs=heuristic_kwargs,
        ),
        batch_size=hparams['batch_size'],
        FlowEstimatorGen=(CNNmultisplit_gen, hparams['flowestimator_opt']),
        path_length=hparams['length_cutoff'],
        initflow=hparams['initial_flow'],
        reg_post=reg_post,
        reg_fn=reg_fn,
        grad_batch_size=hparams['grad_batch_size'],
        reward_rescale_estimator=normalization_reward_fns[hparams['reward_rescale']],
        ncopy=hparams['pool_size'],
        logger=logger,
    )

    flow.initialize()
    return flow

def loss_gen(hparams,logger):
    logger.info('Generate Loss')
    B = Bpower(beta=hparams['B_beta'])
    A = Apower(alpha=hparams['loss_alpha'])
    logger.debug('B: %s' % str(B))
    loss_fn = MeanABError(
        A=A,
        B=B,
        normalization_fn=normalization_flow_fns[hparams['normalization_fn']],
        normalization_nu_fn=normalization_nu_fns[hparams['normalization_nu_fn']],
        cutoff=cutoff_fns[hparams['loss_cutoff']],
    )
    logger.debug('loss_fn: %s' % loss_fn)
    return loss_fn
def train_test_model(hparams,logger):
    logger.info('CALL TRAIN')
    tf.random.set_seed(hparams['seed'])

    # Initialize model
    flow = model_gen(hparams,logger)

    # Initialize ReplayBuffer
    seeded_uniform = tf.random.stateless_uniform(
        (hparams['epochs'], hparams['grad_batch_size'], hparams['batch_size'], hparams['length_cutoff'] - 1,1),
        (hparams['seed'], hparams['seed']),
        dtype='float32'
    )
    seeded_initial = flow.graph.sample(
        shape=(hparams['epochs'], hparams['grad_batch_size'], hparams['batch_size'],1),
        axis=-2
    )

    Replay = ReplayBuffer(
        epoch_per_train=hparams['epochs'],
        seeded_uniform=seeded_uniform,
        seeded_initial=seeded_initial,
        pool_size=hparams['pool_size'],
    )
    callback_hp_tune = tuning_method[hparams['tuning_method']](**hparams)
    pandas_record = PandasRecord(hparams, epoch_period=hparams['epochs'])
    Replay.reward = hparams['rew_fn']
    memory_use = MemoryUse()
    for m in metrics:
        flow.metric_list.append(m(nflow=hparams['pool_size']))

    flow.compile(
        optimizer=optimizers[hparams['optimizer']](hparams['learning_rate']),
        loss=loss_gen(hparams,logger),
    )
    log_tensorlist(logger,flow.trainable_variables,name='flow trainable')
    log_tensorlist(logger,flow.non_trainable_variables,name='flow non trainable')
    TOTAL_EPOCH = hparams['epochs'] * hparams['N_SAMPLE']
    flow.fit(
        np.zeros(hparams['step_per_epoch']),
        epochs=TOTAL_EPOCH,
        verbose=0,
        batch_size=1,
        callbacks=[
            pandas_record,
            # tf.keras.callbacks.TerminateOnNaN(),
            Replay,
            # tf.keras.callbacks.TensorBoard(log_dir=hparams['profile_logdir'],
                                           # histogram_freq = 1,
                                           # profile_batch = (2,200),
                                           # write_steps_per_second=True,
                                           # ),
            callback_hp_tune,
            memory_use,
            # LogProgress(logger)
        ]

    )
    logger.debug('results: %s' % str(pandas_record.results))
    logger.info('END TRAIN')
    return pandas_record.results,None,flow
