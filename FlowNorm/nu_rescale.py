import tensorflow as tf




def trivial_nu(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    return 1.

def batchwise_estimated_expected_density(flownu, init):
    """\f$\frac{1}{b}\sum_{i=1}^b \sum_{t=0}^{\ell}  P(tay>t|s^b)\f$

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    return tf.reduce_mean(tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True),axis=0,keepdims=True)

def expected_density(flownu, init):
    """ \sum_{t=0}^{\ell}  P(tay>t|s^b)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    return tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)






normalization_nu_fns = [
    trivial_nu,
    batchwise_estimated_expected_density,
    expected_density,
    batchwise_estimated_expected_density
]