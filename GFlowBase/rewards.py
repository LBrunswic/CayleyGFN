import tensorflow as tf
import numpy as np
from itertools import product
import sys, os
from time import time


def R_zero(*arg,**kwarg):
    def R(x):
        return np.zeros(x.shape[0],dtype='float32')
    return R

def R_one(*arg,dtype= tf.float32,**kwarg):
    def R(x):
        return np.ones(x.shape[0],dtype='float32')
    return R

def R_first_one(scale,*arg,dtype= tf.float32,**kwarg):

    def R(x):
        return scale*tf.cast(x[:,0]==1,dtype)
    return R

def R_first_one(scale,*arg,dtype= tf.float32,**kwarg):

    def R(x):
        return scale*tf.cast(x[:,0]==1,dtype)
    return R



def Manhattan(size,*arg,width=1,dtype= tf.float32,**kwarg):
    center = tf.constant(tf.cast(np.arange(width),dtype))
    def R(x):
        return (tf.linalg.norm(tf.cast(x,dtype)[:,:width]-center,ord=1,axis=1)+1e-1)**(-1)

    return R

def H_first_one(*arg,dtype=tf.float32,**kwarg):
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,54), dtype=tf.float32)])
    def R(x):
        return 54/(1+tf.math.abs(x[:,0]))
    return R


def R_rubick(*arg,dtype=tf.float32,**kwarg):
    target = tf.constant(np.arange(54),dtype=dtype)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,54), dtype=tf.float32)])
    def R(x):
        return tf.nn.relu(1-2*tf.norm(x-target,ord=1,axis=1))
    return R

def H_rubick(*arg,dtype=tf.float32,**kwarg):
    target = tf.constant(np.arange(54),dtype=dtype)
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,54), dtype=tf.float32)])
    def R(x):
        return 1/(1+tf.norm(x-target,ord=1,axis=1))
    return R

# def R_main(x,THRESHOLD = 15000):
#     x = x.astype('int32')
#     batch_size = x.shape[0]
#     res = []
#     for i in range(batch_size):
#         res.append(base.score(order=x[i]))
#     return tf.constant(np.log(1+np.abs(np.array(res).astype('float32')-THRESHOLD)))

class balance_add(tf.keras.Model):
    def __init__(self):
        super(balance_add, self).__init__()
        self.balance = tf.Variable(initial_value=np.array([1.,1.]),dtype=tf.float32)
    def call(self, r,h):
        return tf.tensordot(tf.stack([r,h],axis=-1),self.balance,axes=1)


def schedule_add(epoch=0):
    r = 1
    h = 1/(epoch//10+1)
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
