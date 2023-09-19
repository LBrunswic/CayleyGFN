from Graphs.CayleyGraph import Symmetric, path_representation
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from GFlowBase.regularization import *
from Groups.symmetric_groups import SymmetricUniform,Modal,rubick_generators,inversion,iteration_random
from metrics import FlowSizeStop, ReplayBuffer,ExpectedLen,ExpectedMaxSeenReward,MaxSeenReward,ExpectedReward,FlowSize,RandomCheck
import tensorflow as tf
from datetime import datetime




reg_post_choices = {
    'OrthReg' : proj_reg,
    'AddReg' :  straight_reg,
}


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



NORMALIZATION_FN='normalization_fn'
NORMALIZATION_NU_FN='normalization_nu_fn'
REG_FN_alpha='reg_fn_alpha'
REG_PROJ='reg_proj'
REG_FN_logpmin='reg_fn_logmin'
GRAD_BATCH_SIZE='grad_batch_size'
BATCH_SIZE='batch_size'
LENGTH_CUTOFF_FACTOR='length_cutoff_factor'
INIT_FLOW='initial_flow'
LR='learning_rate'
EPOCHS='epochs'
STEP_PER_EPOCH='step_per_epoch'
B_BETA='B_beta'
PATH_REDRAW='path_redraw'
NEIGHBORHOOD='neighborhood'
GRAPH_SIZE='graph_size'
GRAPH_GENERATORS='graph_generators'
GRAPH_INVERSE='inverse'
GRAPH_INITIAL_POSITION = 'initial_pos'
FE_OPT='flowestimator_opt'
SEED='seed'
EMBEDDING='embedding'
OPTIMIZER='optimizer'
LOSS='loss_base'
LOSS_ALPHA='loss_alpha'
REW_FN='rew_fn'
REW_PARAM='rew_param'
REW_FACTOR='rew_factor'
HEU_FN='heuristic_fn'
HEU_PARAM='heuristic_param'
HEU_FACTOR='heuristic_factor'
def train_test_model(hparams,log_dir=None):
    if log_dir is None:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tf.keras.utils.set_random_seed(hparams[SEED])
    G = Symmetric(
        hparams[GRAPH_SIZE],
        generators=hparams[GRAPH_GENERATORS],
        representation=[('natural', {})],
        inverse=hparams[GRAPH_INITIAL_POSITION],
        random_gen=eval(hparams[GRAPH_INITIAL_POSITION])(hparams[GRAPH_SIZE]),
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
    reward_fn = eval(hparams[REW_FN])(hparams[GRAPH_SIZE], factor=hparams[REW_FACTOR], **eval(hparams[REW_PARAM]))
    heuristic_fn = eval(hparams[HEU_FN])(hparams[GRAPH_SIZE], factor=hparams[HEU_FACTOR], **eval(hparams[HEU_PARAM]))

    flow = GFlowCayleyLinear(
        graph=G,
        reward=Reward(
            reward_fn=reward_fn,
            heuristic_fn=heuristic_fn,
        ),
        batch_size=hparams[BATCH_SIZE],
        FlowEstimatorGen=(dense_gen, eval(hparams[FE_OPT])),
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