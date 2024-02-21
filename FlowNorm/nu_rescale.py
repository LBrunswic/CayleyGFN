import tensorflow as tf

BaseModel =  tf.keras.Model

class SelfDensity(BaseModel):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    def call(self, flownu, init):
        logdensity_trainable = flownu[..., 4]
        return tf.math.exp(logdensity_trainable)

class SelfDensityModerationPathwiseExpectedLength(BaseModel):
    """ \sum_{t=0}^{\ell}  P(tay>t|s^b)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    def call(self, flownu, init):
        logdensity_trainable = flownu[..., 4]
        return tf.math.exp(logdensity_trainable)/(1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True))

class SelfDensityModerationExpectedLength(BaseModel):
    """\f$\frac{1}{b}\sum_{i=1}^b \sum_{t=0}^{\tau_{max}}  P(tay>t|s^b)\f$

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    def call(self, flownu, init):
        logdensity_trainable = flownu[..., 4]
        return tf.math.exp(logdensity_trainable)/tf.reduce_mean((1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)), axis=0, keepdims=True)

class TheoreticalPathwiseDensity(BaseModel):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    def call(self, flownu, init):
        R = flownu[..., 2]
        return tf.math.cumsum(R,axis=1,reverse=True)

class TheoreticalPathwiseDensityModerationPathwiseExpectedLength(BaseModel):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    def call(self, flownu, init):
        R = flownu[..., 2]
        return tf.math.cumsum(R,axis=1,reverse=True)/(1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True))

class TheoreticalPathwiseDensityModerationExpectedLength(BaseModel):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    def call(self, flownu, init):
        R = flownu[..., 2]
        return tf.math.cumsum(R,axis=1,reverse=True)/tf.reduce_mean((1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)), axis=0, keepdims=True)


normalization_nu_fns = [
    SelfDensity(),
    SelfDensityModerationExpectedLength(),
    SelfDensityModerationPathwiseExpectedLength(),
    TheoreticalPathwiseDensity(),
    TheoreticalPathwiseDensityModerationExpectedLength(),
    TheoreticalPathwiseDensityModerationPathwiseExpectedLength(),
]