
import tensorflow as tf
@tf.function
def meanphi(self,phi,paths,proba):
    PHI = phi(tf.reshape(paths[:, :-1], (-1, paths.shape[-1])))
    P = (proba[:, :-1] - proba[:, 1:])
    return tf.reduce_mean(tf.reduce_sum(tf.reshape(PHI,paths[:,:-1].shape[:2]) * P,axis=1),axis=0)

@tf.function
def Rbar_Error(paths_reward,  density, delta=1e-8):
    RRbar = tf.reduce_mean(tf.reduce_sum(paths_reward * density,axis=1),axis=0)
    RR = tf.reduce_sum(tf.math.square(paths_reward)) / tf.reduce_sum(paths_reward)
    return tf.abs(RRbar-RR)/(delta+RR)

@tf.function
def Rhat_Error(FIOR,total_R,initial_flow,delta=1e-8):
    return tf.reduce_mean(
        tf.math.abs(initial_flow+FIOR[:, :, 0]-FIOR[:, :, 1]-FIOR[:, :, 2])/(delta+total_R)
    )



class ExpectedLen(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='ExpectedLength', **kwargs):
        super(ExpectedLen, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.length = self.add_weight(name='length', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')


    @tf.function
    def expected_len(self,density):
        return tf.reduce_mean(tf.reduce_sum(density,axis=1))

    def update_state(self, Flownu, reg_gradients, sample_weight=None):
        density = tf.math.exp(Flownu[..., 4])
        if self.cutoff is None:
            Elen = self.expected_len(density)
        else:
            Elen = self.expected_len(density[:self.cutoff])
        self.length.assign_add(Elen)
        self.n.assign_add(1.)

    def result(self):
        return self.length / self.n

class ExpectedMaxSeenReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='EMaxSeenRew', **kwargs):
        super(ExpectedMaxSeenReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.max_reward = self.add_weight(name='max_reward', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')



    def update_state(self, Flownu, reg_gradients, sample_weight=None):

        self.max_reward.assign_add(tf.reduce_mean(tf.reduce_max(Flownu[...,2],axis=1)))
        self.n.assign_add(1.)

    def result(self):
        return self.max_reward / self.n

class MaxSeenReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='MaxSeenRew', **kwargs):
        super(MaxSeenReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.max_reward = self.add_weight(name='MaxSeenReward', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')

    def update_state(self, Flownu, reg_gradients, sample_weight=None):

        self.max_reward.assign_add(tf.reduce_max(Flownu[...,2]))
        self.n.assign_add(1.)

    def result(self):
        return self.max_reward / self.n

@tf.function
def expected_reward(FoutStar,R,density,delta=1e-20):
    return tf.reduce_mean(tf.reduce_sum(density*R**2/(delta+FoutStar+R),axis=1),axis=0)

class ExpectedReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='ExpectedReward', **kwargs):
        super(ExpectedReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.reward = self.add_weight(name='reward', initializer='zeros')



    def update_state(self, Flownu, reg_gradients, sample_weight=None):
        density = tf.math.exp(Flownu[..., 4])
        if self.cutoff is None:
            Erew = expected_reward(Flownu[...,1],Flownu[...,2],density)
        else:
            Erew = self.expected_reward(Flownu[:self.cutoff,:,1],Flownu[:self.cutoff,:,2],density[:self.cutoff])
        self.reward.assign(Erew)

    def result(self):
        return self.reward

class FlowSize(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='FlowSize', **kwargs):
        super(FlowSize, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.flow = self.add_weight(name='flow', initializer='zeros')



    def update_state(self, Flownu, reg_gradients, sample_weight=None):

        if self.cutoff is None:
            flow = tf.reduce_max(Flownu[...,1])
        else:
            flow = tf.reduce_max(Flownu[:self.cutoff,1])
        self.flow.assign(flow)

    def result(self):
        return self.flow

class RandomCheck(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='RandomCheck', **kwargs):
        super(RandomCheck, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.random = self.add_weight(name='RandomCheck', initializer='zeros')

    def update_state(self, Flownu, reg_gradients, sample_weight=None):
        self.random.assign(tf.random.uniform(()))
    def result(self):
        return self.random