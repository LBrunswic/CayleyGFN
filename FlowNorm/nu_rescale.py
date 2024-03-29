import tensorflow as tf

BaseModel =  tf.keras.Model

class SelfDensity(BaseModel):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    def __init__(self, name="SelfDensity", **kwargs):
        super(SelfDensity, self).__init__(name=name, **kwargs)

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
    def __init__(self, name="SelfDensityModerationPathwiseExpectedLength", **kwargs):
        super(SelfDensityModerationPathwiseExpectedLength, self).__init__(name=name, **kwargs)

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
    def __init__(self, name="SelfDensityModerationExpectedLength", **kwargs):
        super(SelfDensityModerationExpectedLength, self).__init__(name=name, **kwargs)

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
    def __init__(self, name="TheoreticalPathwiseDensity", **kwargs):
        super(TheoreticalPathwiseDensity, self).__init__(name=name, **kwargs)

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
    def __init__(self, name="TheoreticalPathwiseDensityModerationPathwiseExpectedLength", **kwargs):
        super(TheoreticalPathwiseDensityModerationPathwiseExpectedLength, self).__init__(name=name, **kwargs)

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
    def __init__(self, name="TheoreticalPathwiseDensityModerationExpectedLength", **kwargs):
        super(TheoreticalPathwiseDensityModerationExpectedLength, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        R = flownu[..., 2]
        return tf.math.cumsum(R,axis=1,reverse=True)/tf.reduce_mean((1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)), axis=0, keepdims=True)

class TheoreticalPathwiseDensityNormalized(BaseModel):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    def __init__(self, name="TheoreticalPathwiseDensityNormalized", **kwargs):
        super(TheoreticalPathwiseDensityNormalized, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        R = flownu[..., 2]
        nu = tf.math.cumsum(R,axis=1,reverse=True)
        # nu = nu / tf.reduce_sum(nu, axis=1, keepdims=True)
        nu = nu / (1e-10 + tf.reduce_sum(R, axis=0, keepdims=True))

        return nu


class TheoreticalPathwiseDensityNormalizedPlusSelfDensity(BaseModel):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """

    def __init__(self, name="TheoreticalPathwiseDensityNormalizedPlusSelfDensity", **kwargs):
        super(TheoreticalPathwiseDensityNormalizedPlusSelfDensity, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        R = flownu[..., 2]
        nu = tf.math.cumsum(R, axis=1, reverse=True)
        # nu = nu / tf.reduce_sum(nu, axis=1, keepdims=True)
        nu = nu / (1e-10 + tf.reduce_sum(R, axis=0, keepdims=True)) + 0.01*tf.math.exp(flownu[..., 4])

        return nu
normalization_nu_fns = [
    SelfDensity(),
    SelfDensityModerationExpectedLength(),
    SelfDensityModerationPathwiseExpectedLength(),
    TheoreticalPathwiseDensity(),
    TheoreticalPathwiseDensityModerationExpectedLength(),
    TheoreticalPathwiseDensityModerationPathwiseExpectedLength(),
    TheoreticalPathwiseDensityNormalized(),
    TheoreticalPathwiseDensityNormalizedPlusSelfDensity()
]