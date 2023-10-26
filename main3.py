import numpy as np
import tensorflow as tf
from time import time
print("TensorFlow version:", tf.__version__)
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.enable_op_determinism()

# tf.random.set_seed(122344)
mnist = tf.keras.datasets.mnist


class ChannelReshape(tf.keras.layers.Layer):
    def __init__(self, nsplit):
        super(ChannelReshape, self).__init__()
        self.nsplit = nsplit

    def call(self, inputs):
        return tf.stack(tf.split(inputs[..., 0, :], self.nsplit, axis=-1), axis=-1)


N_case = 2**2
for ncopy in range(4,5):
    batch_size = 4
    # ncopy = 2
    path_length = None
    embdedding_dim = 16
    out = 3 # = width_{i+1} * ncopy
    shape = (batch_size,path_length,1,embdedding_dim,ncopy)


    widths = [
        embdedding_dim,
        64,
        64,
        64,
        # out,
    ]




    layers = []
    layers.append(
        tf.keras.layers.Conv3D(
            out * ncopy,
            (1, 1, embdedding_dim),
            strides=(1, 1, 1),
            padding='valid',
            groups=ncopy,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            activation=ChannelReshape(ncopy)
            # input_shape=shape[1:],
        ),
    )
    # for i,width in enumerate(widths[:-1]):
    #     layers.append(
    #         tf.keras.layers.Conv3D(
    #             widths[i+1]*ncopy,
    #             (1, 1,widths[i]),
    #             strides=(1, 1,1),
    #             padding='valid',
    #             groups=ncopy,
    #             kernel_initializer='glorot_uniform',
    #             bias_initializer='zeros',
    #             # input_shape=shape[1:],
    #         ),
    #     )
        # layers.append(tf.keras.layers.Reshape((-1,1,widths[i+1],ncopy)))
    # layers.append(
    #     tf.keras.layers.Conv3D(
    #         out * ncopy,
    #         (1, 1, 64),
    #         strides=(1, 1, 1),
    #         padding='valid',
    #         groups=ncopy,
    #         kernel_initializer='glorot_uniform',
    #         bias_initializer='zeros',
    #         # input_shape=shape[1:],
    #     ),
    # )
    model = tf.keras.models.Sequential(layers)

    path_length = 16
    shape = (batch_size,path_length,1,embdedding_dim,ncopy)
    model(tf.random.normal(shape))
    T = time()
    for i in range(N_case//ncopy):
        model(tf.random.normal(shape))
    print(ncopy, time() - T, model(tf.random.normal(shape)).shape)
    #
    # path_length = 8
    # shape = (batch_size, path_length,1,embdedding_dim, ncopy)
    # model(tf.random.normal(shape))
    # T = time()
    # for i in range(N_case // ncopy):
    #     model(tf.random.normal(shape))
    # print(ncopy, time() - T, model(tf.random.normal(shape)).shape)
    #
    # path_length = 1
    # shape = (batch_size, path_length,1, embdedding_dim, ncopy)
    # model(tf.random.normal(shape))
    # T = time()
    # for i in range(N_case // ncopy):
    #     model(tf.random.normal(shape))
    # print(ncopy, time() - T,model(tf.random.normal(shape)).shape)
    #
    # path_length = 128
    # shape = (batch_size, path_length, 1,embdedding_dim, ncopy)
    # model(tf.random.normal(shape))
    # T = time()
    # for i in range(N_case // ncopy):
    #     model(tf.random.normal(shape))
    # print(ncopy, time() - T, model(tf.random.normal(shape)).shape)
    #
    #
    #
    # path_length = 1024
    # shape = (batch_size, path_length,1, embdedding_dim, ncopy)
    # model(tf.random.normal(shape))
    # T = time()
    # for i in range(N_case // ncopy):
    #     model(tf.random.normal(shape))
    # print(ncopy, time() - T, model(tf.random.normal(shape)).shape)
# model.fit(x_train, y_train, epochs=5)
# model(x_train)
#
# for i in range(Ntest//ncopy):
#     model(x_train)
