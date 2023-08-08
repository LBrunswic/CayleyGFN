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
        model, epoch_length ,save_name='Errors',
        path_draw=True,
        load=False,
        logger=None, folder='tests/',
    ):
        super(ReplayBuffer, self).__init__()
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.embedding_dim = model.embedding_dim
        self.nactions = model.graph.nactions
        self.path_draw = path_draw
        self.save_name = save_name
        self.epoch_length = epoch_length
        self.load=load
        self.paths = np.zeros((0, model.path_length, self.embedding_dim), dtype='int16')
        self.states_dict = {}
        self.states = np.zeros((0, self.embedding_dim), dtype='int16')
        self._dataset = tf.zeros((0, 1 + self.nactions, self.embedding_dim))
        self.reward = np.zeros((0,), dtype='float32')
        self.states_density = np.zeros((0,), dtype='int32')
        self.flush_memory(self.nactions, self.embedding_dim)

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
    def update_metrics_paths(self, logs=None):

        T = time()
        if self.path_draw:
            self.gen_path(model=self.model,exploration=0)
        self.logger.info('Path gen : %s seconds' % (time()-T))
        T=time()
        self.compute_pathsFIOR()
        self.ErrorRhatVals.append(self.Rhat_Error().numpy())
        self.ErrorRbarVals.append(self.Rbar_Error().numpy())
        self.LossVals.append(self.model.loss_val.numpy())
        self.InitVals.append(float(self.model.initial_flow.numpy()))
        self.totalReward.append(tf.reduce_sum(self.reward).numpy())
        self.MaxReward.append(tf.reduce_max(self.reward).numpy())
        self.Nstate.append(self.states.shape[0])
        self.Elen.append(self.expected_len(self.paths))
        self.ERew.append(self.expected_reward(self.paths))
        # self.StatesVisited.append(self.states.shape[0])
        if self.logger is not None:
            self.logger.info('self.states.shape=%s' % str(self.states.shape))
            self.logger.info('Metrics computed: %s seconds' % (time() - T))

    def flush_memory(self,nactions,embedding_dim):
        self.states_dict = {}
        self.states = np.zeros((0, embedding_dim), dtype='int16')
        self._dataset = tf.zeros((0, 1 + nactions, embedding_dim))
        self.reward = np.zeros((0,), dtype='float32')
        self.states_density = np.zeros((0,), dtype='int32')

    def gen_path(self,model,exploration=1e-1):
        # print('self.states.shape=',self.states.shape)
        # print('self.states_dict.len=', len(self.states_dict.keys()))
        # print('self.states.dtype=',self.states.dtype)
        old_state_number = self.states.shape[0]
        if old_state_number>0:
            pass
            # print(type(list(self.states_dict.keys())[0][0]))

        model.gen_path_aux(exploration=exploration)
        new_states = model.paths.numpy().reshape(-1,model.embedding_dim).astype('int16')
        # print('new_states=', new_states.shape)
        new_states_tuples = [tuple(x) for x in new_states]
        # print('new_states_tuples type=', type(new_states_tuples[0][0]))
        new_states_dict_tuples = {new_states_tuples[i] : i for i in range(new_states.shape[0])} # keep states without redundancy!
        added_states_dict = {
            new_states_tuples[i]: i
            for i in range(len(new_states_tuples))
            if new_states_tuples[i] not in self.states_dict
        }
        previous_states_dict = {
            new_states_tuples[i]: i
            for i in range(len(new_states_tuples))
            if new_states_tuples[i] in self.states_dict
        }
        added_states = new_states[list(added_states_dict.values())]

        self.states = np.concatenate(
            [
                self.states,
                added_states,
            ],
        axis=0
        )

        self.reward = np.concatenate([
            self.reward,
            model.reward(added_states.astype('float32'))
        ])
        n_new_states = len(added_states)
        self.states_dict.update(zip(added_states_dict.keys(), range(old_state_number, old_state_number + n_new_states)))
        self.states_density = np.concatenate(
            [
                self.states_density,
                np.zeros(n_new_states)
            ]
        )
        counted_new_states = Counter(new_states_tuples)
        for key in counted_new_states:
            temp1 = self.states_dict[key]
            temp2 = counted_new_states[key]
            self.states_density[temp1] += temp2


        self._dataset = tf.concat(
            [
                self._dataset,
                model.build_dataset(added_states)
            ],
            axis=0
        )
    def compute_pathsFIOR(self):
        self.paths = self.model.paths
        self.FIOR = self.model.FinOutReward(self.model.paths)

    def Rhat_Error(self,delta=1e-8):
        total_R = tf.reduce_sum(self.reward).numpy()
        return tf.reduce_mean(
            tf.math.abs(self.model.initial_flow+self.FIOR[:, :, 0]-self.FIOR[:, :, 1]-self.FIOR[:, :, 2])/(delta+total_R)
        )

    # @tf.function
    def proba_aux(self,paths,delta=1e-8):
        batch_size, length = paths.shape[:2]
        proba = np.zeros((batch_size, length))
        proba[:, 0] = 1
        for i in range(1, length):
            proba[:, i] = proba[:, i - 1] * self.FIOR[:, i - 1, 1] / (
                    delta+self.FIOR[:, i - 1, 1] + self.FIOR[:, i - 1, 2]
            )
        return proba

    def expected_len(self,paths):
        proba = self.proba_aux(paths)
        return np.mean(np.sum(proba,axis=1))

    def expected_reward(self,paths,delta=1e-8):
        return np.mean(np.sum(self.proba_aux(paths)*self.FIOR[:,:,2]**2/(delta+self.FIOR[:,:,1]+self.FIOR[:,:,2]),axis=1),axis=0)

    def meanphi(self,phi,paths):
        proba = self.proba_aux(paths)
        PHI = phi(tf.reshape(paths[:, :-1], (-1, paths.shape[-1])))
        P = (proba[:, :-1] - proba[:, 1:])
        return tf.reduce_mean(tf.reduce_sum(tf.reshape(PHI,paths[:,:-1].shape[:2]) * P,axis=1),axis=0)

    def Rbar_Error(self,delta=1e-8):
        RRbar = self.meanphi(self.model.reward, self.paths)
        RR = tf.reduce_sum(tf.math.square(self.model.reward(self.states.astype('float32'))))/ tf.reduce_sum(self.model.reward(self.states.astype('float32')))
        return tf.abs(RRbar-RR)/(delta+RR)

    def check_nan(self):
        NaN = False
        for variables in self.model.variables:
            if np.sometrue(np.isnan(variables)):
                NaN = True
        self.logger.info('The model contains NAN weights')

    def update_reward(self,epoch):
        self.model.reward.update_reward(epoch=epoch)

    def on_train_end(self, logs=None):
        self.check_nan()
        self.flush_memory(self.model.graph.nactions, self.model.embedding_dim)
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

    def train_data(self,model=None,exploration=1e-2):
        if model is None:
            model = self.model
        self.gen_path(model,exploration=exploration)
        mu_initial_reward = np.concatenate(
            [
                self.states_density.reshape(-1, 1),
                np.ones_like(self.states_density).reshape(-1, 1),
                self.reward.reshape(-1, 1)
            ],
            axis=1
        )
        return self._dataset,mu_initial_reward

    def on_epoch_end(self, epoch, logs=None):
        self.update_reward(epoch=epoch)
