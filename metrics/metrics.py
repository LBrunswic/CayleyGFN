
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
    def __init__(self, cutoff=None, nflow=1, name='ExpectedLength', **kwargs):
        super(ExpectedLen, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.length = self.add_weight(shape=(nflow,),name='ExpectedLength', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')

    @tf.function
    def expected_len(self, density):
        return tf.reduce_mean(tf.reduce_sum(density, axis=1),axis=0)

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        density = tf.math.exp(flownu[..., 4])
        if self.cutoff is None:
            expexted_len = self.expected_len(density)
        else:
            expexted_len = self.expected_len(density[:self.cutoff])
        self.length.assign_add(expexted_len)
        self.n.assign_add(1.)

    def reset_state(self):
        # pass
        self.length.assign(tf.zeros_like(self.length))
        self.n.assign(tf.zeros_like(self.n))

    def result(self):
        return self.length / self.n


class ExpectedMaxSeenReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, nflow=1, name='EMaxSeenRew', **kwargs):
        super(ExpectedMaxSeenReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.max_reward = self.add_weight(shape=(nflow,), name='EMaxSeenReward', initializer='zeros')
        self.n = self.add_weight(name='n_sample', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        reward = flownu[..., 5]
        self.max_reward.assign_add(tf.reduce_mean(tf.reduce_max(reward, axis=1),axis=0))
        self.n.assign_add(1.)
    def reset_state(self):
        pass
        self.max_reward.assign(tf.zeros_like(self.max_reward))
        self.n.assign(0)
    def result(self):
        return self.max_reward / self.n


class MaxSeenReward(tf.keras.metrics.Metric):
    def __init__(self, cutoff=None, nflow=1, name='MaxSeenRew', **kwargs):
        super(MaxSeenReward, self).__init__(name=name, **kwargs)
        self.cutoff = cutoff
        self.max_reward = self.add_weight(shape=(nflow,),name='MaxSeenReward', initializer='zeros')
        # self.n = self.add_weight(name='n_sample', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        reward = flownu[..., 5]
        self.max_reward.assign(tf.reduce_max(tf.stack([tf.reduce_max(tf.reduce_max(reward,axis=0), axis=0), self.max_reward]), axis=0))

    def reset_state(self):
        self.max_reward.assign(tf.zeros_like(self.max_reward))

    def result(self):
        return self.max_reward


@tf.function
def expected_reward_fn(foutstar, path_reward, density, delta=1e-20):
    return tf.reduce_mean(tf.reduce_sum(density*path_reward**2/(delta+foutstar+path_reward), axis=1), axis=0)


class ExpectedReward(tf.keras.metrics.Metric):
    def __init__(self, name='ExpectedReward', nflow=1, **kwargs):
        super(ExpectedReward, self).__init__(name=name, **kwargs)
        self.reward = self.add_weight(shape=(nflow,), name='Expected_reward', initializer='zeros')
    
    def update_state(self, flownu, reg_gradients, sample_weight=None):
        density = tf.math.exp(flownu[..., 4])
        reward = flownu[..., 2]
        rescale = tf.reduce_max(flownu[..., 5])/tf.reduce_max(flownu[..., 2])
        foutstar = flownu[..., 1]
        expected_reward = expected_reward_fn(foutstar, reward, density)*rescale
        self.reward.assign(expected_reward)

    def reset_state(self):
        pass

    def result(self):
        return self.reward


class FlowSize(tf.keras.metrics.Metric):
    def __init__(self, nflow=1,  name='FlowSize', **kwargs):
        super(FlowSize, self).__init__(name=name, **kwargs)
        self.flow = self.add_weight(shape=(nflow,), name='flow', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        flow = tf.reduce_max(tf.reduce_max(flownu[..., 1],axis=0),axis=0)
        self.flow.assign(flow)
    def reset_state(self):
        self.flow.assign(tf.zeros_like(self.flow))

    def result(self):
        return self.flow


class RandomCheck(tf.keras.metrics.Metric):
    def __init__(self, name='RandomCheck', nflow=None, **kwargs):
        super(RandomCheck, self).__init__(name=name, **kwargs)
        self.random = self.add_weight(name='RandomCheck', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        self.random.assign(tf.random.uniform(()))

    def result(self):
        return self.random


class PathLeak(tf.keras.metrics.Metric):
    def __init__(self, nflow=1, name='PathLeak', **kwargs):
        super(PathLeak, self).__init__(name=name, **kwargs)
        self.path_leak = self.add_weight(shape=(nflow,), name='path_leak', initializer='zeros')

    def update_state(self, flownu, reg_gradients, sample_weight=None):
        logdensity_trainable = flownu[..., 4]
        expected_leak = tf.reduce_mean(logdensity_trainable[:, -1], axis=0)
        self.path_leak.assign(expected_leak)

    def reset_state(self):
        self.path_leak.assign(tf.zeros_like(self.path_leak))
    def result(self):
        return self.path_leak


class InitFlowMetric(tf.keras.metrics.Metric):
    def __init__(self, nflow=1, name='initflow', **kwargs):
        super(InitFlowMetric, self).__init__(name=name, **kwargs)
        self.initflow = self.add_weight(shape=(nflow,), name=name, initializer='zeros')

    def update_state(self,init_estimate, sample_weight=None):
        self.initflow.assign(init_estimate)

    def reset_state(self):
        self.initflow.assign(tf.zeros_like(self.initflow))

    def result(self):
        return self.initflow
