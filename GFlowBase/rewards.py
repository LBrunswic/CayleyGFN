import tensorflow as tf
import numpy as np
from itertools import product
import sys, os
from time import time



def R_zero(*arg,**kwarg):
    @tf.function
    def R(x):
        return tf.zeros(x.shape[0],dtype='float32')
    return R


def R_one(*arg,dtype= tf.float32,**kwarg):
    @tf.function
    def R(x):
        return tf.ones(x.shape[0],dtype='float32')
    return R


def R_first_k(scale,k,*arg,dtype= tf.float32,**kwarg):
    target =  tf.constant(np.arange(k).reshape(1,-1),dtype=dtype)
    @tf.function
    def R(x):
        return scale*tf.nn.relu(1-2*tf.norm(tf.cast(x[:,:k],'float32')-target,ord=1,axis=1))
    return R



def Manhattan(size,*arg,width=1,dtype= tf.float32,**kwarg):
    center = tf.constant(tf.cast(np.arange(width),dtype))
    # a = np.power(1e-4,1/size)
    # score = tf.constant(a**np.arange(size),dtype=dtype)
    # v = tf.constant(np.arange(width).reshape(1,1,-1),dtype=dtype)
    # hot = lambda x: tf.nn.relu(1+x)*tf.nn.relu(1-x)
    def R(x):
        # raw_score = tf.einsum('ijk,j->i', hot(tf.expand_dims(x,-1)-v), score )
        return (tf.linalg.norm(tf.cast(x,dtype)[:,:width]-center,ord=1,axis=1)+1e-1)**(-1)
        # return tf.nn.relu(1-0.01*tf.linalg.norm(tf.cast(x,dtype)[:,:width]-center,ord=1,axis=1))
        # return raw_score
    return R

def TwistedManhattan(size,*arg,width=1,scale=1e-4,factor=1.,dtype= tf.float32,**kwarg):
    # print(size,width,arg,factor,scale,dtype,kwarg)
    a = scale
    # score = factor*tf.constant(a**np.arange(size),dtype=dtype)
    score = factor*tf.constant(tf.math.exp(a*tf.range(size,dtype=dtype)),dtype=dtype)
    score = np.stack([np.roll(np.concatenate([score[::-1],score[1:]]),i)[size-1:size-1+width] for i in range(size)])
    print(score)
    v = tf.constant(np.arange(width).reshape(1,1,-1),dtype=dtype)
    hot = lambda x: tf.nn.relu(1+x)*tf.nn.relu(1-x)
    # print(score.shape)
    @tf.function
    def R(x):
        x = tf.cast(x,'float32')
        raw_score = tf.einsum('ijk,jk->i', hot(tf.expand_dims(x,-1)-v), score )
        return raw_score
    # print(R)
    return R

def H_first_one(*arg,dtype=tf.float32,**kwarg):
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,54), dtype=tf.float32)])
    def R(x):
        return 54/(1+tf.math.abs(x[:,0]))
    return R


class balance_add(tf.keras.Model):
    def __init__(self):
        super(balance_add, self).__init__()
        self.balance = tf.Variable(
            initial_value=np.array([1.,1.]),
            dtype=tf.float32,
            trainable=False
        )
    def call(self, r,h):
        return tf.tensordot(tf.stack([r,h],axis=-1),self.balance,axes=1)


def schedule_add(epoch=0):
    r = 1
    h = 1
    return np.array([r,h])

class Reward():
    def __init__(
            self,
            reward_fn=R_one(),
            heuristic_fn=R_zero(),
            memory=False,
            name='one',
            base_folder='Knowledge',
            folder=None,
            balance=balance_add(),
            schedule=schedule_add,
    ):
        self.name = name
        self.reward_fn = reward_fn
        self.heuristic_fn = heuristic_fn
        self.schedule = schedule
        self.balance = balance
        self.memory = memory
        self.update_reward()
        if folder is None:
            self.folder = os.path.join(base_folder,name)
        try:
            os.makedirs(self.folder)
        except FileExistsError as e:
            pass
    def update_reward(self,epoch=0):
        self.balance.set_weights([self.schedule(epoch=epoch)])

    def __call__(self, *args, **kwargs):
        return self.balance(self.reward_fn(*args,**kwargs),self.heuristic_fn(*args,**kwargs))
