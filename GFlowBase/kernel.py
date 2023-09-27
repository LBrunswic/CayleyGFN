
import tensorflow as tf
import numpy as np

class softmax_head(tf.keras.layers.Layer):
    def __init__(self,kernel):
        super(softmax_head, self).__init__()
        self.kernel = kernel

    def build(self, input_shape):
        self.kernel.build(input_shape)
        self.kernel_pathwise = tf.keras.layers.TimeDistributed(self.kernel)
        self.kernel_pathwise_multiedge = tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(self.kernel))

    def load_weights(self,*args,**kwargs):
        self.kernel.load_weights(*args,**kwargs)

    def call(self, inputs):
        V = self.kernel_pathwise(inputs)
        return tf.math.exp(V[...,-1:])*tf.nn.softmax(V[...,:-1])

    def call_statewise(self, inputs):
        V = self.kernel(inputs)
        return tf.math.exp(V[...,-1:])*tf.nn.softmax(V[...,:-1])

    def call_pathaction_wise(self,inputs):
        V = self.kernel_pathwise_multiedge(inputs)
        return tf.math.exp(V[...,-1:])*tf.nn.softmax(V[...,:-1])


class direct_head(tf.keras.layers.Layer):
    def __init__(self,kernel):
        super(direct_head, self).__init__()
        self.kernel = kernel

    def build(self, input_shape):
        self.kernel.build(input_shape)

    def load_weights(self,*args,**kwargs):
        self.kernel.load_weights(*args,**kwargs)

    def call(self, inputs):
        V = self.kernel(inputs)
        return tf.math.abs(V)


def dense_gen(ndirectactions,kernel_depth=2,width=64,final_activation=tf.math.abs,head=softmax_head,kernel_options={}):
    flow_kernel = tf.keras.Sequential(name='flow_kernel')
    for _ in range(kernel_depth):
        flow_kernel.add(
            tf.keras.layers.Dense(width,**kernel_options)
        )
    if head is softmax_head:
        nout= ndirectactions + 1
        head = softmax_head
    else:
        nout = ndirectactions
        head = direct_head
    flow_kernel.add(
        tf.keras.layers.Dense(
            nout,
            activation=final_activation,
            kernel_initializer=tf.keras.initializers.Constant(1e-3),
            bias_initializer=tf.keras.initializers.Constant(1.)
        )
    )
    return head(flow_kernel)
