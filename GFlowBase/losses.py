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

def Bpower(alpha=0.5,beta=0.1,delta=1e-8):
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
        density_fixed = Flownu[..., 4]
        logdensity_trainable = Flownu[..., 5]
        density_one = Flownu[..., 6]
        return tf.reduce_mean(tf.reduce_sum(density_fixed*self.g(tf.math.abs((self.delta+Finstar+Finit) / (self.delta+Foutstar+R))),axis=-1))

@tf.function
def fall(x):
    return tf.nn.relu(1+x/20)

@tf.function
def Rbar_Error(paths_reward,  density, delta=1e-8):
    RRbar = tf.reduce_mean(tf.reduce_sum(paths_reward * density,axis=1),axis=0)
    RR = tf.reduce_sum(tf.math.square(paths_reward)) / tf.reduce_sum(paths_reward)
    return tf.math.log(delta+tf.abs(RRbar-RR)/(delta+RR))


class MeanABError(tf.keras.losses.Loss):
    def __init__(self, A=tf.math.square, B=lambda x,y: 1,name='plop',**kwargs):
        super(MeanABError, self).__init__(name=name, **kwargs)
        self.A = A
        self.B = B

    @tf.function
    def call(self, Flownu, bis):
        Finstar = Flownu[..., 0]
        Foutstar = Flownu[..., 1]
        R =  Flownu[..., 2]
        Finit = Flownu[..., 3]
        density_fixed = Flownu[..., 4]
        logdensity_trainable = Flownu[..., 5]
        density_one = Flownu[..., 6]

        # Ptau = density_fixed[:,:-1]-density_fixed[:,1:] # P(tau=t|path,t)
        # ER = tf.reduce_sum(density_fixed * R[:,:-1],axis=1) # E(R|path)
        # ERbar = Rbar_Error(R,tf.math.exp(logdensity_trainable))
        ER = tf.reduce_sum(R,axis=1) # E(R|path)
        # weights = ER #+ 0.0001
        weights = 1.
        Ldelta = tf.reduce_mean(weights*tf.reduce_sum(density_fixed*self.A(Foutstar+R-Finstar-Finit)*self.B(Finstar, Foutstar),axis=-1))
        Ebigtau = tf.reduce_mean(logdensity_trainable[:,-1]) #log P(tau > tmax)
        return Ldelta#+fall(Ebigtau)
