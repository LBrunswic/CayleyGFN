import tensorflow as tf
import numpy as np

@tf.function
def mlog(x):
    return -tf.math.log(x)

@tf.function
def KL(x):
    return x*tf.math.log(x)

@tf.function
def chi2(x):
    return tf.math.square(1-x)

@tf.function
def Hellinger(x):
    return tf.math.square(1-tf.sqrt(x))

@tf.function
def TV(x):
    return tf.math.abs(1-x)

@tf.function
def LeCam(x):
    return (1-x)/(2*x+2)

@tf.function
def JensenShannon(x):
    return x*tf.math.log(2*x/(x+1)) + tf.math.log(2/(1+x))


@tf.function
def Bengio(x):
    return tf.square(tf.math.log(x))


@tf.function
def Alog(x):
    return tf.math.log(1+tf.math.abs(x))


def Alogsquare(alpha=2,delta=1.):
    @tf.function
    def aux(x):
        return tf.math.log(1+delta*tf.pow(tf.math.abs(x),alpha))
    return aux
def Apower(alpha=0.5):
    @tf.function
    def aux(x):
        return tf.math.pow(1+tf.square(tf.math.abs(x)),alpha)
    return aux

def Bpower(alpha=2.,beta=0.1,delta=1e-8):
    @tf.function
    def B(x, y):
        # return tf.math.log(1 + beta*tf.math.abs(tf.math.pow(x,alpha) -tf.math.pow(y,alpha)))
        return (1 + beta* tf.math.pow(delta + x + y,alpha))
    return B

class divergence(tf.keras.losses.Loss):
    def __init__(self, g=Bengio, delta=1e-10,name='plop',**kwargs):
        super(divergence, self).__init__(name=name, **kwargs)
        self.g = g
        self.delta = delta

    @tf.function
    def call(self, Flownu, bis):
        Finstar = Flownu[..., 0]
        Foutstar = Flownu[..., 1]
        R =  Flownu[..., 2]
        Finit = Flownu[..., 3]
        logdensity_trainable = Flownu[..., 4]
        density_fixed = tf.stop_gradient(tf.math.exp(logdensity_trainable))
        return tf.reduce_mean(tf.reduce_sum(density_fixed*self.g(tf.math.abs((self.delta+Finstar+Finit) / (self.delta+Foutstar+R))),axis=-1))


@tf.function
def Rbar_Error(paths_reward,  density, delta=1e-8):
    RRbar = tf.reduce_mean(tf.reduce_sum(paths_reward * density,axis=1),axis=0)
    RR = tf.reduce_sum(tf.math.square(paths_reward)) / tf.reduce_sum(paths_reward)
    return tf.math.log(delta+tf.abs(RRbar-RR)/(delta+RR))


class MeanABError(tf.keras.losses.Loss):
    def __init__(self, A=tf.math.square, B=lambda x,y: 1, normalization_fn=None, name='loss',**kwargs):
        super(MeanABError, self).__init__(name=name, **kwargs)
        self.A = A
        self.B = B
        if normalization_fn is None:
            self.normalization_fn  =lambda flownu,finit: tf.ones_like(flownu[..., 0])
        else:
            self.normalization_fn = normalization_fn

    @tf.function
    def call(self, Flownu, finit):
        Finstar = Flownu[..., 0]
        Foutstar = Flownu[..., 1]
        R =  Flownu[..., 2]
        Finit = Flownu[..., 3]
        logdensity_trainable = Flownu[..., 4]
        normalization = tf.stop_gradient(self.normalization_fn(Flownu,finit))
        density_fixed = tf.stop_gradient(tf.math.exp(logdensity_trainable))
        Ldelta = tf.reduce_mean(tf.reduce_sum(density_fixed*self.A((Foutstar+R-Finstar-Finit)/normalization)*self.B(Finstar/normalization, Foutstar/normalization),axis=-1))
        return Ldelta

