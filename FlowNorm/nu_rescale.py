import tensorflow as tf




def trivial_nu(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    return 1.


def pathwise_expected_length(flownu, init):
    """ \sum_{t=0}^{\ell}  P(tay>t|s^b)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    return 1+tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True)


def expected_length(flownu, init):
    """\f$\frac{1}{b}\sum_{i=1}^b \sum_{t=0}^{\tau_{max}}  P(tay>t|s^b)\f$

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)
    """
    return 1+tf.reduce_mean(pathwise_expected_length(flownu, init), axis=0, keepdims=True)




normalization_nu_fns = [
    trivial_nu,
    expected_length,
    pathwise_expected_length,
    # batchwise_estimated_expected_density
]