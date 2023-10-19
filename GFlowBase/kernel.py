
import tensorflow as tf
import numpy as np

class softmax_head(tf.keras.Model):
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


class multi_softmax_head(tf.keras.layers.Layer):
    def __init__(self,kernel):
        super(multi_softmax_head, self).__init__()
        self.kernel = kernel

    def build(self, input_shape):
        self.kernel.build(input_shape)
        # self.kernel_pathwise_multiedge = tf.keras.layers.TimeDistributed(self.kernel)

    @tf.function
    def call(self, inputs):
        f = self.kernel(inputs)
        return tf.math.exp(f[:, :, :,-1:]) * tf.nn.softmax(f[:, :, :, :-1], axis=3)

class multi_softmax_head_timedistribute(tf.keras.layers.Layer):
    def __init__(self,kernel):
        super(multi_softmax_head_timedistribute, self).__init__()
        self.kernel = kernel

    def build(self, input_shape):
        # self.kernel.build((None,input_shape[:-2],input_shape[-1]))
        self.kernels = {}
    @tf.function
    def call(self, inputs):
        if inputs.shape[2] not in self.kernels:
            self.kernels.update({inputs.shape[2] : tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(self.kernel))})
        f = self.kernels[inputs.shape[2]](inputs)
        return tf.math.exp(f[:, :, :,-1:]) * tf.nn.softmax(f[:, :, :, :-1], axis=3)



def dense_gen(ndirectactions,kernel_depth=2,width=64,final_activation=tf.math.abs,head=softmax_head,kernel_options={}):
    flow_kernel = tf.keras.Sequential(name='flow_kernel')
    for _ in range(kernel_depth):
        flow_kernel.add(
            tf.keras.layers.Dense(width,**kernel_options)
        )
    if head is softmax_head:
        nout= ndirectactions + 1
        head = softmax_head

    flow_kernel.add(
        tf.keras.layers.Dense(
            nout,
            activation=final_activation,
            kernel_initializer=tf.keras.initializers.Constant(1e-3),
            bias_initializer=tf.keras.initializers.Constant(1.)
        )
    )
    return head(flow_kernel)

class ChannelReshape(tf.keras.layers.Layer):
    def __init__(self, nsplit,out):
        super(ChannelReshape, self).__init__()
        self.kernel = tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(tf.keras.layers.Reshape((out,nsplit))))
        self.nsplit = nsplit
    @tf.function
    def call(self, inputs):
        # return self.kernel(inputs)
        return tf.stack(tf.split(inputs[..., 0, :], self.nsplit, axis=-1), axis=-1)


def CNNmulti_gen(nflow, ndirections, kernel_depth=2, width=64, embedding_dim=None, final_activation=tf.math.abs, kernel_options=None):
    if kernel_options is None:
        kernel_options = {}
    nout = ndirections+1
    shape = (None, None, embedding_dim, nflow)
    widths = [embedding_dim]+[width]*kernel_depth
    # print(widths)
    layers = [tf.keras.layers.Input(shape=shape)]
    if kernel_depth>0:
        layers.append(
            tf.keras.layers.Conv3D(
                    widths[1] * nflow,
                    (1, 1, widths[0]),
                    strides=(1, 1, 1),
                    groups=nflow,
                    **kernel_options
            )
        )
        for i, width in enumerate(widths[1:-1]):
            layers.append(
                tf.keras.layers.Conv3D(
                    widths[i + 1] * nflow,
                    (1, 1, 1),
                    strides=(1, 1, 1),
                    groups=nflow,
                    **kernel_options
                )
            )
        layers.append(
            tf.keras.layers.Conv3D(
                nout * nflow,
                (1, 1, 1),
                strides=(1, 1, 1),
                groups=nflow,
                activation=final_activation
            )
        )
    else:
        layers.append(
            tf.keras.layers.Conv3D(
                nout * nflow,
                (1, 1, embedding_dim),
                strides=(1, 1, 1),
                groups=nflow,
                activation=final_activation
            )
        )
    layers.append(ChannelReshape(nflow,nout))
    flow_kernel = tf.keras.models.Sequential(layers)
    return multi_softmax_head(flow_kernel)


def EinSummulti_gen(nflow, ndirections, kernel_depth=2, width=64, embedding_dim=None, final_activation=tf.math.abs, kernel_options=None):
    if kernel_options is None:
        kernel_options = {}
    nout = ndirections+1
    shape = (None, None, embedding_dim, nflow)
    widths = [width]*kernel_depth
    # print(widths)
    layers = [tf.keras.layers.Input(shape=shape)]
    for out_dim in widths:
        layers.append(
            tf.keras.layers.EinsumDense('...ec,oec->...oc', output_shape=(out_dim, nflow),bias_axes='oc')
        )
    layers.append(
        tf.keras.layers.EinsumDense('...ec,oec->...oc', output_shape=(nout, nflow),bias_axes='oc')
    )
    flow_kernel = tf.keras.models.Sequential(layers)
    return multi_softmax_head(flow_kernel)

def EinSummultiv2_gen(nflow, ndirections, kernel_depth=2, width=64, embedding_dim=None, final_activation=tf.math.abs, kernel_options=None):
    if kernel_options is None:
        kernel_options = {}
    nout = ndirections+1
    shape = (None, None, embedding_dim, nflow)
    widths = [width]*kernel_depth
    # print(widths)
    layers = [tf.keras.layers.Input(shape=shape)]
    for out_dim in widths:
        layers.append(
            tf.keras.layers.EinsumDense('uwec,oec->uwoc', output_shape=(None,None,out_dim, nflow),bias_axes='oc')
        )
    layers.append(
        tf.keras.layers.EinsumDense('uwec,oec->uwoc', output_shape=(None,None,nout, nflow),bias_axes='oc')
    )
    flow_kernel = tf.keras.models.Sequential(layers)
    return multi_softmax_head(flow_kernel)

class MultiDense(tf.keras.layers.Layer):
    def __init__(self, nsplit, out):
        super(MultiDense, self).__init__()
        self.nsplit = nsplit
        self.out = out
    def build(self, input_shape):
        n = len(input_shape)
        self.A = tf.Variable(tf.zeros((*([1]*(n-2)), input_shape[-2], self.out, input_shape[-1])))
        self.B = tf.Variable(tf.zeros((*([1]*(n-2)), self.out, input_shape[-1])))
    @tf.function
    def call(self, inputs):
        return tf.reduce_sum(tf.expand_dims(inputs,-2) * self.A, axis=3) + self.B

def DenseMultiDirect_gen(nflow, ndirections, kernel_depth=2, width=64, embedding_dim=None, final_activation=tf.math.abs, kernel_options=None):
    if kernel_options is None:
        kernel_options = {}
    nout = ndirections+1
    shape = (None, None, embedding_dim, nflow)
    widths = [width]*kernel_depth
    layers = [tf.keras.layers.Input(shape=shape)]
    for out_dim in widths:
        layers.append(
            MultiDense(nflow,out_dim)
        )
    layers.append(
            MultiDense(nflow,nout)
        )
    flow_kernel = tf.keras.models.Sequential(layers)
    return multi_softmax_head(tf.keras.layers.TimeDistributed(flow_kernel))


def CNNmultisplit_gen(nflow, ndirections, kernel_depth=2, width=64, embedding_dim=None, final_activation=tf.math.abs, kernel_options=None):
    if kernel_options is None:
        kernel_options = {}
    nout = ndirections+1
    shape = (None, None, embedding_dim,nflow)
    widths = [embedding_dim]+[width]*kernel_depth
    # print(widths)
    flow_kernel = []
    for i in range(nflow):
        layers = [tf.keras.layers.Input(shape=(*shape[:-1],1))]
        if kernel_depth>0:
            layers.append(
                tf.keras.layers.Conv3D(
                        widths[1],
                        (1,  1, widths[0]),
                        **kernel_options
                )
            )
            for i, width in enumerate(widths[1:-1]):
                layers.append(
                    tf.keras.layers.Conv3D(
                        widths[i + 1],
                        (1,  1, 1),
                        **kernel_options
                    )
                )
            layers.append(
                tf.keras.layers.Conv3D(
                    nout ,
                    (1, 1,1),
                    activation=final_activation
                )
            )
        else:
            layers.append(
                tf.keras.layers.Conv3D(
                    nout ,
                    (1, 1, embedding_dim),
                    activation=final_activation
                )
            )
        # layers.append(ChannelReshape(nflow,nout))
        flow_kernel.append(tf.keras.models.Sequential(layers))
    inputs = tf.keras.layers.Input(shape=shape)
    y = [flow_kernel[i](x)[...,0,:] for i,x in enumerate(tf.split(inputs, nflow, axis=-1))]
    outputs = tf.stack(y, axis=-1)
    return multi_softmax_head(tf.keras.Model(inputs=inputs,outputs=outputs))


def LocallyConnectedmulti_gen(nflow, ndirections, kernel_depth=2, width=64, embedding_dim=None, final_activation=tf.math.abs, kernel_options=None):
    if kernel_options is None:
        kernel_options = {}
    nout = ndirections+1
    shape = (None, None, None, nflow) #One dimension lacking, assumed last dimensions are swaped
    layers = []
    layers.append(
        tf.keras.layers.Permute((2,1))
    )
    for i in range(kernel_depth):
        layers.append(
            tf.keras.layers.LocallyConnected1D(
                width ,
                1,
                **kernel_options
            )
        )
    layers.append(
        tf.keras.layers.LocallyConnected1D(
            nout,
            1,
            activation=final_activation
        )
    )
    layers.append(
        tf.keras.layers.Permute((2,1))
    )
    core_model = tf.keras.Sequential(layers,name='core_estimator')
    return multi_softmax_head_timedistribute(core_model)
