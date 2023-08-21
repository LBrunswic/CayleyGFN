
import tensorflow as tf
import numpy as np

class total(tf.keras.layers.Layer):
    def __init__(self,kernel):
        super(total, self).__init__()
        self.kernel = kernel

    def build(self, input_shape):
        self.kernel.build(input_shape)

    def load_weights(self,*args,**kwargs):
        self.kernel.load_weights(*args,**kwargs)

    def call(self, inputs):
        V = self.kernel(inputs)
        return tf.math.exp(V[:,-1:])*tf.nn.softmax(V[:,:-1])
        # return tf.math.abs(V[:,-1:])*tf.nn.softmax(V[:,:-1])

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self,n=10,**kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.n =n
    def call(self,input):
        return tf.concat([
            tf.math.cos(i*input)
            for i in range(1,self.n+1)
        ]+
        [
            tf.math.sin(i*input)
            for i in range(1,self.n+1)
        ],
        axis=-1
    )

def dense_gen(ndirectactions,kernel_depth=2,width=64,final_activation=tf.math.abs,softmax_head=True,kernel_options={},encoding=-1):
    flow_kernel = tf.keras.Sequential(name='flow_kernel')
    if encoding>0:
        flow_kernel.add(PositionalEncoding(encoding))
    for _ in range(kernel_depth):
        flow_kernel.add(
            tf.keras.layers.Dense(width,**kernel_options)
        )
    if softmax_head:
        nout= ndirectactions + 1
        head = total
    else:
        nout = ndirectactions
        head = lambda x:x
    flow_kernel.add(
        tf.keras.layers.Dense(
            nout,
            activation=final_activation,
            kernel_initializer=tf.keras.initializers.Constant(1e-3),
            bias_initializer=tf.keras.initializers.Constant(1.)
        )
    )
    return head(flow_kernel)



def tranformer_gen(ndirectactions,kernel_depth=2,width=64,final_activation=tf.math.abs,softmax_head=True,kernel_options={},ke=-1):
    flow_kernel = tf.keras.Sequential(name='flow_kernel')
    if encoding>0:
        flow_kernel.add(PositionalEncoding(encoding))
    for _ in range(kernel_depth):
        flow_kernel.add(
            tf.keras.layers.Dense(width,**kernel_options)
        )
    if softmax_head:
        nout= ndirectactions + 1
        head = total
    else:
        nout = ndirectactions
        head = lambda x:x
    flow_kernel.add(
        tf.keras.layers.Dense(
            nout,
            activation=final_activation,
            kernel_initializer=tf.keras.initializers.Constant(1e-3),
            bias_initializer=tf.keras.initializers.Constant(1.)
        )
    )

    return head(flow_kernel)
