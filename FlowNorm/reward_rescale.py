import tensorflow as tf

@tf.function
def expected_reward(FoutStar,R,density,delta=1e-20):
    return tf.reduce_mean(tf.reduce_sum(density*R**2/(delta+FoutStar+R),axis=1),axis=0)

class RewardRescaleERew(tf.keras.Model):
    def __init__(self, name='FlowNorm', **kwargs):
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
    def __init__(self, name='FlowNorm', **kwargs):
        super(RewardRescaleTrivial, self).__init__(name=name, **kwargs)
    def update_state(self, Flownu):
        pass
    def fn_call(self):
        return 1.


normalization_reward_fns = {
    'Trivial' : RewardRescaleTrivial,
    'ERew': RewardRescaleERew
}