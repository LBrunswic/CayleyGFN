from Graphs.CayleyGraph import Symmetric, path_representation
from GFlowBase.kernel import *
from GFlowBase.GFlowCayley import GFlowCayleyLinear
from GFlowBase.losses import *
from GFlowBase.rewards import *
from GFlowBase.regularization import *
from Groups.symmetric_groups import SymmetricUniform,Modal,rubick_generators,inversion,iteration_random
from metrics import FlowSizeStop,PathAccuracy, ReplayBuffer,ExpectedLen,ExpectedMaxSeenReward,MaxSeenReward,ExpectedReward,FlowSize,RandomCheck,PathLeak
import tensorflow as tf
from datetime import datetime


class PlaceHolder(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(PlaceHolder).__init__(**kwargs)


class One(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(One).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 and epoch > 0:
            self.model.reg_fn.alpha.assign(self.model.reg_fn.alpha*tf.exp(-1.))
class Two(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(One).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0 and epoch > 0:
            self.model.reg_fn.alpha.assign(self.model.reg_fn.alpha*tf.exp(-0.3))
class Short(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(Short).__init__(**kwargs)
    def on_epoch_end(self, epoch, logs=None):
        if epoch > 10:
            self.model.reg_fn.alpha.assign(self.model.reg_fn.alpha*tf.exp(-0.3))

reg_fn_alpha_schedules = {
    'none' : PlaceHolder,
    'one' : One,
    'two' : Two,
    'short' : Short,
}


reg_post_choices = {
    'OrthReg' : proj_reg,
    'AddReg' :  straight_reg,
}
reg_fn_gen_choices =   {
        'PathAccuracy' : PathAccuracy_gen,
        'LogPathLen' : LogPathLen_gen,
        'norm2' : Norm2_gen,
    }

cutoff_fns = {
    'none' : lambda flownu,finit: tf.ones_like([flownu[..., 0]]),
    'non_zero' : lambda flownu,finit: tf.cast(flownu[..., 2] >= tf.reduce_min(flownu[..., 2], axis=1,keepdims=True), 'float32'),
}

def schedule_one(epoch,lr):
    if epoch<20:
        return lr
    else:
        return lr*0.98
def schedule_two(epoch,lr):
    if epoch<20:
        return lr
    else:
        return lr*0.8

lr_schedule = {
    'none' : lambda epoch,lr: lr,
    'one' :  schedule_one,
    'two' :  schedule_two
}


normalization_fns =[
    None,
    lambda flownu,finit: 1e-3 + tf.reduce_sum(flownu[..., :4], axis=-1) / 2,
    lambda flownu,finit: 1e-3 + flownu[...,1],
    lambda flownu,finit: 1e-3 + finit,
    lambda flownu,finit: 1e-3 + tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1)) ,
    lambda flownu,finit: 1e-3 + tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1,keepdims=True),
    lambda flownu,finit: 1e-3 + 10*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1)) ,
    # lambda flownu,finit: 1e-3 + (finit+tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flownu[..., :4], axis=-1),axis=-1))),
]

normalization_nu_fns =[
    None,
    lambda flownu,finit: tf.reduce_mean(tf.reduce_sum(tf.math.exp(flownu[..., 4]),axis=1)),
    lambda flownu,finit: tf.reduce_sum(tf.math.exp(flownu[..., 4]),axis=1,keepdims=True),
    lambda flownu,finit: tf.reduce_mean(tf.reduce_sum(tf.math.exp(flownu[..., 4]),axis=1))
]

optimizers = {
        'Adam' : lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
        'Nesterov' : lambda lr: tf.keras.optimizers.SGD(learning_rate=lr,nesterov=True),
    }

@tf.function
def expected_reward(FoutStar,R,density,delta=1e-20):
    return tf.reduce_mean(tf.reduce_sum(density*R**2/(delta+FoutStar+R),axis=1),axis=0)

class RewardRescaleERew(tf.keras.Model):
    def __init__(self, name='RewardRescale', **kwargs):
        super(RewardRescaleERew, self).__init__(name=name, **kwargs)
        self.reward = self.add_weight(name='reward', initializer='ones',trainable=False,shape=())
    def update_state(self, Flownu):
        density = tf.math.exp(Flownu[..., 4])
        reward = Flownu[...,5]
        Foutstar=Flownu[..., 1]
        Erew = expected_reward(Foutstar,reward,density)
        self.reward.assign(Erew)
    def fn_call(self):
        return 3*self.reward

class RewardRescaleTrivial(tf.keras.Model):
    def __init__(self, name='RewardRescale', **kwargs):
        super(RewardRescaleTrivial, self).__init__(name=name, **kwargs)
    def update_state(self, Flownu):
        pass
    def fn_call(self):
        return 1.

reward_rescale = {
    'Trivial' : RewardRescaleTrivial,
    'ERew': RewardRescaleERew
}
Metrics = [
        ExpectedReward,
        ExpectedMaxSeenReward,
        MaxSeenReward,
        ExpectedLen,
        FlowSize,
        RandomCheck,
        lambda :tf.keras.metrics.Mean(name='initflow'),
        PathLeak,
        PathAccuracy
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
REWARD_RESCALE='reward_rescale'
REG_FN_GEN='reg_fn_gen'
LOSS_CUT='loss_cutoff'
LR_SCHEDULE='lr_schedule'
REG_FN_alpha_SCHEDULE='reg_fn_alpha_schedule'
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
    if hparams[B_BETA]>-900:
        B = Bpower(beta=tf.math.exp(hparams[B_BETA]))
    else:
        B = lambda x,y:1
    print(eval(hparams[LOSS])(alpha=hparams[LOSS_ALPHA]))
    loss_fn = MeanABError(
        A=eval(hparams[LOSS])(alpha=hparams[LOSS_ALPHA]),
        B=B,
        normalization_fn=normalization_fns[hparams[NORMALIZATION_FN]],
        normalization_nu_fn=normalization_nu_fns[hparams[NORMALIZATION_NU_FN]],
        cutoff=cutoff_fns[hparams[LOSS_CUT]],
    )
    reg_fn_gen = reg_fn_gen_choices[hparams[REG_FN_GEN]]
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
        reward_rescale_estimator=reward_rescale[hparams[REWARD_RESCALE]],
        # reg_fn=reg_withcutoff_fn
    )
    flow(0)
    # flow.FlowEstimator.kernel.summary()


    Replay = ReplayBuffer()
    Replay.reward = reward_fn


    for m in Metrics:
        flow.metric_list.append(m())


    flow.compile(
        optimizer=optimizers[hparams[OPTIMIZER]](hparams[LR]),
        loss=loss_fn,
    )



    flow.fit(
        np.zeros(hparams[STEP_PER_EPOCH]),
        initial_epoch=0,
        epochs=hparams[EPOCHS],
        verbose=0,
        batch_size=1,
        callbacks=[
            FlowSizeStop(),
            # tf.keras.callbacks.TerminateOnNaN(),
            Replay,
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.LearningRateScheduler(lr_schedule[hparams[LR_SCHEDULE]]),
            reg_fn_alpha_schedules[hparams[REG_FN_alpha_SCHEDULE]]()
        ]
    )
    return flow.evaluate()