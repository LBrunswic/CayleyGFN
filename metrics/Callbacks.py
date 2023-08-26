import sys
import os
import tensorflow as tf
import numpy as np
from time import time
from collections import Counter

from datetime import datetime

class FollowInitLoss(tf.keras.callbacks.Callback):
    def __init__(self, target_flow,delta=1e-8, save_name='initloss', folder='tests/',load=False):
        super(FollowInitLoss, self).__init__()
        self.Nval=3
        self.folder = folder
        self.save_name = save_name
        self.target_flow = target_flow
        self.values = []
        self.delta=delta
        os.makedirs(folder, exist_ok=True)
        if load:
            with open(os.path.join(self.folder, self.save_name+'.npy'), 'br') as f:
                self.pre_values = np.load(f)
        else:
            self.pre_values = np.zeros((0,self.Nval))

    def on_epoch_end(self, epoch, logs=None):
        length = len(set([tuple(x) for x in self.model.paths.numpy().reshape(-1, self.model.embedding_dim)]))
        self.values.append(np.concatenate(
            [
                self.model.initial_flow.numpy().reshape(-1)-self.target_flow,
                self.model.loss_val.numpy().reshape(-1),
                np.array(length,dtype='float32').reshape(-1)
            ]
        ))

    def on_train_end(self, logs=None):
        A = np.concatenate([
            self.pre_values.reshape((-1,self.Nval)),
            np.abs((np.array(self.values)).reshape((-1,self.Nval)))
        ],axis=0)
        with open(os.path.join(self.folder,self.save_name+'.npy'),'bw') as f:
            np.save(f,A)
        with open(os.path.join(self.folder,self.save_name+'_captions.npy'),'bw') as f:
            np.save(f,np.array(['init_diff','loss','exploration']))


class ReplayBuffer(tf.keras.callbacks.Callback):
    def __init__(self,
        model ,save_name='Errors',
        load=False,
        logger=None, folder='tests/',
    ):
        super(ReplayBuffer, self).__init__()
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.embedding_dim = model.embedding_dim
        self.nactions = model.graph.nactions
        # self.path_draw = path_draw
        self.save_name = save_name
        self.exploration = tf.Variable(1e-1,trainable=False)
        # self.epoch_length = epoch_length
        # self.load=load
        # self.paths = np.zeros((0, model.path_length, self.embedding_dim), dtype='int16')
        # self.states_dict = {}
        # self.states = np.zeros((0, self.embedding_dim), dtype='int16')
        # self._dataset = tf.zeros((0, 1 + self.nactions, self.embedding_dim))
        # self.reward = np.zeros((0,), dtype='float32')

        # self.states_density = np.zeros((0,), dtype='int32')



        # self.flush_memory(self.nactions, self.embedding_dim)

        if load:
            with open(os.path.join(self.folder,self.save_name+'.npy'),'br') as f:
                [
                    ErrorRhatVals,
                    ErrorRbarVals,
                    LossVals,
                    InitVals,
                    totalReward,
                    MaxReward,
                    Nstate,
                    Elen,
                    ERew
                ] = np.load(f)
                self.ErrorRhatVals = list(ErrorRhatVals)
                self.ErrorRbarVals = list(ErrorRbarVals)
                self.LossVals = list(LossVals)
                self.InitVals = list(InitVals)
                self.totalReward = list(totalReward)
                self.MaxReward = list(MaxReward)
                self.Nstate = list(Nstate)
                self.Elen = list(Elen)
                self.ERew = list(ERew)
        else:
            self.ErrorRhatVals = []
            self.ErrorRbarVals = []
            self.LossVals = []
            self.InitVals = []
            self.totalReward = []
            self.MaxReward = []
            self.Nstate = []
            self.Elen = []
            self.ERew = []
        # self.StatesVisited = []
        self.logger=logger
    def update_metrics_paths(self,delta=1e-8):

        T = time()
        self.model.gen_path_aux(exploration=0)
        self.model.update_reward()
        self.model.FinOutReward()
        self.model.compute_density(delta=delta,exploration=0.)
        if self.logger is not None:
            self.logger.info('Path gen : %s seconds' % (time()-T))
        T=time()

        self.ErrorRhatVals.append(self.Rhat_Error().numpy())
        self.ErrorRbarVals.append(self.Rbar_Error().numpy())
        self.LossVals.append(self.model.loss_val.numpy())
        self.InitVals.append(float(self.model.initial_flow.numpy()))
        self.totalReward.append(tf.reduce_sum(self.model.paths_reward).numpy())
        self.MaxReward.append(tf.reduce_max(self.model.paths_reward).numpy())
        self.Nstate.append(0)
        self.Elen.append(self.expected_len())
        self.ERew.append(self.expected_reward())
        # self.StatesVisited.append(self.states.shape[0])
        if self.logger is not None:
            # self.logger.info('self.states.shape=%s' % str(self.states.shape))
            self.logger.info('Metrics computed: %s seconds' % (time() - T))


    @tf.function
    def Rhat_Error(self,delta=1e-8):
        FIOR = self.model.FIOR
        total_R = tf.reduce_sum(self.model.paths_reward)
        return tf.reduce_mean(
            tf.math.abs(self.model.initial_flow+FIOR[:, :, 0]-FIOR[:, :, 1]-FIOR[:, :, 2])/(delta+total_R)
        )

    def expected_len(self):
        return tf.reduce_mean(tf.reduce_sum(self.model.density,axis=1))

    def expected_reward(self,delta=1e-8):
        FIOR = self.model.FIOR
        return np.mean(np.sum(self.model.density*FIOR[:,:,2]**2/(delta+FIOR[:,:,1]+FIOR[:,:,2]),axis=1),axis=0)

    def meanphi(self,phi,paths,proba):
        PHI = phi(tf.reshape(paths[:, :-1], (-1, paths.shape[-1])))
        P = (proba[:, :-1] - proba[:, 1:])
        return tf.reduce_mean(tf.reduce_sum(tf.reshape(PHI,paths[:,:-1].shape[:2]) * P,axis=1),axis=0)

    def Rbar_Error(self,delta=1e-8):
        RRbar = self.meanphi(self.model.reward, self.model.paths,self.model.density)
        RR = tf.reduce_sum(tf.math.square(self.model.paths_reward)) / tf.reduce_sum(self.model.paths_reward)
        return tf.abs(RRbar-RR)/(delta+RR)

    def check_nan(self):
        NaN = False
        for variables in self.model.variables:
            if np.sometrue(np.isnan(variables)):
                NaN = True
        if self.logger is not None:
            self.logger.info('The model contains NAN weights')

    def on_train_end(self, logs=None):
        self.check_nan()
        self.update_metrics_paths()
        Out = [
            self.ErrorRhatVals,
            self.ErrorRbarVals,
            self.LossVals,
            self.InitVals,
            self.totalReward,
            self.MaxReward,
            self.Nstate,
            self.Elen,
            self.ERew,
        ]
        A =  np.stack([np.array(x) for x in Out],axis=0)
        with open(os.path.join(self.folder,self.save_name+'.npy'),'bw') as f:
            np.save(f,A)
        with open(os.path.join(self.folder,self.save_name+'_captions.npy'),'bw') as f:
            np.save(f,np.array(['ErrorRhat','ErrorRbar','loss','init','totR','MaxR']))

    def on_train_begin(self, logs=None):
        T = time()
        self.model.update_training_distribution(exploration=self.exploration)
        
        if self.logger is not None:
            self.logger.info('Path gen : %s seconds' % (time()-T))


    # def on_epoch_end(self, epoch, logs=None):
    #     self.update_reward(epoch=epoch)
