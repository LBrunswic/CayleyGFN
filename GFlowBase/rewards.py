import tensorflow as tf
import numpy as np
from itertools import product
import sys, os
from time import time



def R_zero(*arg,**kwarg):
    @tf.function
    def R(x,*fargs,**fkwargs):
        return tf.zeros_like(x[:,0],dtype='float32')
    return R


def R_one(*arg,dtype= tf.float32,**kwarg):
    @tf.function
    def R(x,*fargs,**fkwargs):
        return tf.ones(x.shape[0],dtype='float32')
    return R





def TwistedManhattan(size,*arg,width=1,scale=-100,factor=1.,dtype= tf.float32,delta=1e-20,exp=False,mini=1,**kwarg):
    # print(size,width,arg,factor,scale,dtype,kwarg)
    a = scale
    # score = factor*tf.constant(a**np.arange(size),dtype=dtype)
    score = factor*tf.constant(tf.math.exp(a*tf.range(size,dtype=dtype)),dtype=dtype)
    score = np.stack([np.roll(np.concatenate([score[::-1],score[1:]]),i)[size-1:size-1+width] for i in range(size)])
    # print(score)
    v = tf.constant(np.arange(width),dtype=dtype)
    hot = lambda x: tf.nn.relu(1+x)*tf.nn.relu(1-x)
    indicator = lambda x: tf.cast(tf.greater(x, mini-delta), dtype=tf.float32)
    final = lambda x : x
    if exp:
        final = lambda x : x*tf.math.exp(x)
    else:
        final = lambda x : x
    # print(score.shape)
    # @tf.function
    def R(x,axis=-1):
        # print(x.shape)
        x = tf.cast(x,'float32')
        # print(x.shape)
        x = tf.experimental.numpy.swapaxes(x,-1,axis)
        # print(x.shape)
        x = tf.expand_dims(x,-1)
        # print(x.shape)
        raw_score = tf.einsum('...jk,jk->...', hot(x-v), score)
        tf.experimental.numpy.swapaxes(x, -1, axis)
        return delta+final(indicator(raw_score)*raw_score)
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

reward_map = {
    'R_zero' : R_zero,
    'TwistedManhattan': TwistedManhattan
}

class Reward():
    def __init__(
            self,
            reward_fn=R_one(),
            reward_args=(),
            reward_kwargs={},
            heuristic_fn=R_zero(),
            heuristic_args=(),
            heuristic_kwargs={},
            memory=False,
            name='one',
            base_folder='Knowledge',
            folder=None,
            balance=balance_add(),
            schedule=schedule_add,
    ):
        self.name = name
        self.reward_fn = reward_map[reward_fn](*reward_args,**reward_kwargs)
        self.heuristic_fn = reward_map[reward_fn](*heuristic_args,**heuristic_kwargs)
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

    def __call__(self, state_batch, *args, **kwargs):
        return self.reward_fn(state_batch, *args, **kwargs)
        # return self.balance(self.reward_fn(state_batch, *args, **kwargs), self.heuristic_fn(state_batch, *args, **kwargs))
