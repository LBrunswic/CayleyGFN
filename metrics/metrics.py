
import tensorflow as tf


@tf.function
def meanphi(phi, paths, proba):
    phival = phi(tf.reshape(paths[:, :-1], (-1, paths.shape[-1])))
    p_t = (proba[:, :-1] - proba[:, 1:])
    return tf.reduce_mean(tf.reduce_sum(tf.reshape(phival, paths[:, :-1].shape[:2]) * p_t, axis=1), axis=0)


@tf.function
def rbar_error(paths_reward,  density, delta=1e-8):
    rrbar = tf.reduce_mean(tf.reduce_sum(paths_reward * density, axis=1), axis=0)
    rr = tf.reduce_sum(tf.math.square(paths_reward)) / tf.reduce_sum(paths_reward)
    return tf.abs(rrbar-rr)/(delta+rr)


@tf.function
def rhat_error(fior, total_r, initial_flow, delta=1e-8):
    return tf.reduce_mean(
        tf.math.abs(initial_flow+fior[:, :, 0]-fior[:, :, 1]-fior[:, :, 2])/(delta+total_r)
    )


class ExpectedLen(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='ExpectedLength', **kwargs):
        super(ExpectedLen, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.length = self.add_weight(name='length', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')

    @tf.function
    def expected_len(self, density):
        return tf.reduce_mean(tf.reduce_sum(density, axis=1))

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        density = tf.math.exp(flownu[..., 4])
        if self.cutoff is None:
            expexted_len = self.expected_len(density)
        else:
            expexted_len = self.expected_len(density[:self.cutoff])
        self.length.assign_add(expexted_len)
        self.n.assign_add(1.)

    def result(self):
        return self.length / self.n


class ExpectedMaxSeenReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='EMaxSeenRew', **kwargs):
        super(ExpectedMaxSeenReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.max_reward = self.add_weight(name='max_reward', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        reward = flownu[..., 5]
        self.max_reward.assign_add(tf.reduce_mean(tf.reduce_max(reward, axis=1)))
        self.n.assign_add(1.)

    def result(self):
        return self.max_reward / self.n


class MaxSeenReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='MaxSeenRew', **kwargs):
        super(MaxSeenReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.max_reward = self.add_weight(name='MaxSeenReward', initializer='zeros')
        # self.n = self.add_weight(name='n_sample', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        reward = flownu[..., 5]
        self.max_reward.assign(tf.reduce_max([tf.reduce_max(reward), self.max_reward]))
        # self.n.assign_add(1.)

    def result(self):
        return self.max_reward


@tf.function
def expected_reward_fn(foutstar, path_reward, density, delta=1e-20):
    return tf.reduce_mean(tf.reduce_sum(density*path_reward**2/(delta+foutstar+path_reward), axis=1), axis=0)


class ExpectedReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='ExpectedReward', **kwargs):
        super(ExpectedReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.reward = self.add_weight(name='reward', initializer='zeros')
    
    def update_state(self, flownu, reg_gradients, sample_weight=None):
        density = tf.math.exp(flownu[..., 4])
        reward = flownu[..., 2]
        rescale = tf.reduce_max(flownu[..., 5])/tf.reduce_max(flownu[..., 2])
        foutstar = flownu[..., 1]
        if self.cutoff is None:
            expected_reward = expected_reward_fn(foutstar, reward, density)*rescale
        else:
            expected_reward = rescale*self.expected_reward(
                foutstar[:self.cutoff],
                reward[:self.cutoff],
                density[:self.cutoff]
            )
        self.reward.assign(expected_reward)

    def result(self):
        return self.reward


class FlowSize(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='FlowSize', **kwargs):
        super(FlowSize, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.flow = self.add_weight(name='flow', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):

        if self.cutoff is None:
            flow = tf.reduce_max(flownu[..., 1])
        else:
            flow = tf.reduce_max(flownu[:self.cutoff, 1])
        self.flow.assign(flow)

    def result(self):
        return self.flow


class RandomCheck(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='RandomCheck', **kwargs):
        super(RandomCheck, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.random = self.add_weight(name='RandomCheck', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        self.random.assign(tf.random.uniform(()))

    def result(self):
        return self.random


class PathLeak(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='PathLeak', **kwargs):
        super(PathLeak, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.path_leak = self.add_weight(name='path_leak', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        logdensity_trainable = flownu[..., 4]
        expected_leak = tf.reduce_mean(logdensity_trainable[:, -1])
        self.path_leak.assign(expected_leak)

    def result(self):
        return self.path_leak

class PathAccuracy(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, name='PathAccuracy', **kwargs):
        super(PathAccuracy, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.accuracy = self.add_weight(name='path_accuracy', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        logdensity = flownu[..., 4]
        path_reward = flownu[..., 5]
        total_reward = tf.reduce_sum(path_reward,axis=1,keepdims=True)
        density = tf.concat([tf.exp(logdensity), tf.zeros_like(logdensity[:, :1])],axis=1)
        tau_distribution = density[...,:-1] - density[...,1:]
        accuracy = tf.reduce_mean(tf.reduce_sum(tf.math.log(1e-20 + tf.abs(tau_distribution - path_reward / total_reward)),axis=1))
        self.accuracy.assign(accuracy)

    def result(self):
        return self.accuracy
