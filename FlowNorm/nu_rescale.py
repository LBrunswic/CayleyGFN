import tensorflow as tf


normalization_nu_fns = [
    lambda flownu, finit: 1.,
    lambda flownu, finit: tf.reduce_mean(tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1)),
    lambda flownu, finit: tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1, keepdims=True),
    lambda flownu, finit: tf.reduce_mean(tf.reduce_sum(tf.math.exp(flownu[..., 4]), axis=1))
]
