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
    def call(self, Flow,nu):
        Fout = Flow[:, 1]
        Fin = Flow[:, 0]
        return tf.reduce_sum(nu*self.g(tf.math.abs((self.delta+Fin) / (self.delta+Fout))))

class MeanABError(tf.keras.losses.Loss):
    def __init__(self, A=tf.math.square, B=lambda x,y: 1,name='plop',**kwargs):
        super(MeanABError, self).__init__(name=name, **kwargs)
        self.A = A
        self.B = B

    @tf.function
    def call(self, Flow, nu):
        Finstar = Flow[:,:, 0]
        Foutstar = Flow[:,:, 1]
        Finit = Flow[:,:, 2]
        R =  Flow[:,:, 3]
        return tf.reduce_mean(tf.reduce_sum(nu*self.A(Foutstar+R-Finstar-Finit)*self.B(Finstar, Foutstar),axis=-1))
