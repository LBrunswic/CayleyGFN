import tensorflow as tf

eta = 1e-3


def trivial_nu(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)

    """
    return tf.ones_like(flownu[..., 0])

def f_out_star(flownu, init):
    """Normalization by Fout star

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    return flownu[..., 1]


def f_init(flownu, init):
    """Normalization by f_init

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    return flownu[..., 3]


def f_in_out(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    return tf.reduce_sum(flownu[..., :4], axis=-1)/2


def batchmean_total_f_in_out(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    scale = tf.reduce_mean(tf.reduce_sum(f_in_out(flownu,init), axis=1, keepdims=True), axis=0, keepdims=True)

    return eta + scale * trivial_nu(flownu,init)


def pathwise_total_f_in_out(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    scale = tf.reduce_sum(f_in_out(flownu, init), axis=1, keepdims=True)

    return eta + scale * trivial_nu(flownu,init)


def batchmean_normalized_f_in_out(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    scale = tf.reduce_sum(f_in_out(flownu, init), axis=1, keepdims=True)

    return eta + scale * trivial_nu(flownu,init)


def pathwise_normalized_f_in_out(flownu, init):
    """Trivial normalization (constant = 1.)

        args:
            flownu: tf tensor of shape (batch_size, path_length, n_flow, 5).
                The last axis contains (F_i^*,F_o^*,R/r, F(init->\cdot)/F_o(init), log P(tau>t|s_\dot), R)
            init: F_o(init)

        output shape:  (batch_size, path_length, n_flow)
    """
    scale = tf.reduce_sum(f_in_out(flownu, init), axis=1, keepdims=True)

    return eta + scale * trivial_nu(flownu,init)


normalization_flow_fns = [
    trivial_nu,
    f_in_out,
    f_out_star,
    f_init,
    batchmean_total_f_in_out,
    pathwise_total_f_in_out,
    pathwise_total_f_in_out,
    pathwise_total_f_in_out,
]