import tensorflow as tf

eta = 1e-3
normalization_flow_fns = [
    lambda flow_nu, f_init: 1.,
    lambda flow_nu, f_init: eta + tf.reduce_sum(flow_nu[..., :4], axis=-1) / 2,
    lambda flow_nu, f_init: eta + flow_nu[..., 1],
    lambda flow_nu, f_init: eta + f_init,
    lambda flow_nu, f_init: eta + tf.reduce_mean(
        tf.reduce_sum(tf.reduce_sum(flow_nu[..., :4], axis=-1), axis=2),
        axis=0
    ),
    lambda flow_nu, f_init: eta + tf.reduce_sum(tf.reduce_sum(flow_nu[..., :4], axis=-1), axis=-1, keepdims=True),
    lambda flow_nu, f_init: eta + 10*tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(flow_nu[..., :4], axis=-1), axis=-1)),
]
