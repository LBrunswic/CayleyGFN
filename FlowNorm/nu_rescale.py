import tensorflow as tf




def self_density(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    logdensity_trainable = flownu[..., 4]
    return tf.math.exp(logdensity_trainable)


def self_density_moderation_pathwise_expected_length(flownu, init):
    """ \sum_{t=0}^{\ell}  P(tay>t|s^b)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    logdensity_trainable = flownu[..., 4]
    return tf.math.exp(logdensity_trainable)/(1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True))


def self_density_moderation_expected_length(flownu, init):
    """\f$\frac{1}{b}\sum_{i=1}^b \sum_{t=0}^{\tau_{max}}  P(tay>t|s^b)\f$

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    logdensity_trainable = flownu[..., 4]
    return tf.math.exp(logdensity_trainable)/tf.reduce_mean((1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)), axis=0, keepdims=True)

def theoretical_pathwise_density(flownu, init):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    R = flownu[..., 2]
    logdensity_trainable = flownu[..., 4]
    return tf.math.cumsum(R,axis=1,reverse=True)

def theoretical_pathwise_density_moderation_pathwise_expected_length(flownu, init):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    R = flownu[..., 2]
    logdensity_trainable = flownu[..., 4]
    return tf.math.cumsum(R,axis=1,reverse=True)/(1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True))


def theoretical_pathwise_density_moderation_expected_length(flownu, init):
    """ TODO

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)021
    """
    R = flownu[..., 2]
    logdensity_trainable = flownu[..., 4]
    return tf.math.cumsum(R,axis=1,reverse=True)/tf.reduce_mean((1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)), axis=0, keepdims=True)


normalization_nu_fns = [
    self_density,
    self_density_moderation_expected_length,
    self_density_moderation_pathwise_expected_length,
    theoretical_pathwise_density,
    theoretical_pathwise_density_moderation_expected_length,
    theoretical_pathwise_density_moderation_pathwise_expected_length,
    # batchwise_estimated_expected_density
]
