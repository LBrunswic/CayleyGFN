import sys
import os
import tensorflow as tf
import numpy as np
from time import time
import pandas
from datetime import datetime
from utils.utils import concat_dict_of_ndarray

class PandasRecord(tf.keras.callbacks.Callback):
    def __init__(self,hparams,loss, banch_seeded_uniform,bench_seeded_initial,record_period=10, **kwargs):
        self.epoch_period = hparams['epochs']
        self.loss = loss
        self.results = []
        self.hparams = hparams
        self.nflow = hparams['pool_size']
        self.seeded_uniform = banch_seeded_uniform
        self.seeded_initial = bench_seeded_initial
        self.record_period = record_period

    def on_epoch_end(self, epoch, logs=None):
        # print(epoch)
        if epoch % self.record_period == 0:
            res = self.model.evaluate(self.seeded_initial, self.seeded_uniform)
            episode = epoch//self.epoch_period
            true_epoch = 1+epoch % self.epoch_period
            res.update({
                key: [self.hparams[key]]*self.nflow
                for key in self.hparams
                if not isinstance(self.hparams[key],tuple)
            })
            res.update({'epoch': np.array([true_epoch]*self.nflow)})
            res.update({'episode': [episode] * self.nflow})
            loss_HP = self.loss.HP()
            res.update({key: [loss_HP[key]] * self.nflow for key in loss_HP})
            if 'path_strategy' in self.hparams:
                path_strategy = self.hparams['path_strategy'].HP()
                res.update({key: [path_strategy[key]] * self.nflow for key in path_strategy})

            for key in self.hparams:
                if isinstance(self.hparams[key],tuple):
                    res.update({key: [str(self.hparams[key])]*self.nflow })
            res.update({'reg_fn_alpha': self.model.reg_fn.alpha.numpy()})
            for key in res:
                if isinstance(res[key],tf.Tensor):
                    res[key] = res[key].numpy()
                elif isinstance(res[key],list):
                    res[key] = np.array(res[key])
                elif isinstance(res[key],str):
                    # print(key,res[key])
                    raise
                elif len(res[key].shape)>1:
                    pass

            self.results.append(res)
    def on_train_end(self, logs=None):
        self.results = pandas.DataFrame(concat_dict_of_ndarray(self.results))

class ReplayBuffer(tf.keras.callbacks.Callback):
    def __init__(self, seeded_uniform=None,seeded_initial=None, pool_size=1, path_strategy=None,batch_size=64, monitor="ReplayBuffer", folder='Knowledge',epochs=10):
        super().__init__()
        self.folder = folder
        self.episode_memory = []
        self.epoch_per_train = epochs
        self.pool_size = pool_size
        self.replay_strategy = path_strategy
        self.seeded_uniform = tf.Variable(seeded_uniform,trainable=False)
        self.seeded_initial = tf.Variable(seeded_initial,trainable=False)
        self.memory = None
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.model.compile_update_training_distribution()


    # def on_epoch_end(self, epoch, logs=None):
    #     print('ReplayBuffer', self.model.evaluate()['EMaxSeenRew'])
    def on_epoch_begin(self, epoch, logs=None):
        print('epoch', epoch)
        def schedule(epoch):
            if epoch%self.epoch_per_train <= 1:
                return (0., 1.)
            elif epoch%self.epoch_per_train < 6:
                return (0.90, 0.1)
            else:
                return (0.99, 0.01)
        self.update_training_distribution_callback(epoch)
        self.model.update_flow_init(schedule(epoch))


    def update_training_distribution_callback(self,epoch):

        initial = self.seeded_initial[epoch%self.epoch_per_train]
        initial = tf.broadcast_to(initial,(*initial.shape[:-1], self.pool_size))
        seeded_uniform = self.seeded_uniform[epoch%self.epoch_per_train]
        seeded_uniform = tf.broadcast_to(seeded_uniform, (*seeded_uniform.shape[:-1], self.pool_size))
        self.model.generate_update_training_distribution(initial, seeded_uniform)
        T = time()
        if epoch% self.epoch_per_train == 0:
            self.memory=None
        self.memorize(epoch)
        print('ReplayBuffer:Memorize', time() - T)
        T = time()
        if self.replay_strategy is not None and self.replay_strategy.name != 'baseline':
            self.model.update_training_distribution(*self.replay_strategy(self.memory, self.model.paths_true.shape, epoch,self.batch_size))
        print('ReplayBuffer:Strategy', time() - T)
    def memorize(self,epoch):
        if self.memory is None:
            paths_true = np.concatenate([self.model.paths_true[0, ..., 0].numpy()]*self.epoch_per_train)
            paths_embedded = np.concatenate([self.model.paths[0, ..., 0].numpy()]*self.epoch_per_train)
            paths_reward = np.concatenate([self.model.paths_reward[..., 0].numpy()]*self.epoch_per_train)
            path_init_flow = np.concatenate([self.model.path_init_flow[..., 0].numpy()]*self.epoch_per_train)
            self.memory = {
                'paths_true': paths_true,
                'paths_embedded': paths_embedded,
                'paths_reward': paths_reward,
                'path_init_flow': path_init_flow
            }
        else:
            self.memory['paths_true'][(epoch-1)*self.batch_size:epoch*self.batch_size] = self.model.paths_true[0, ..., 0].numpy()
            self.memory['paths_embedded'][(epoch-1)*self.batch_size:epoch*self.batch_size] = self.model.paths[0, ..., 0].numpy()
            self.memory['paths_reward'][(epoch-1)*self.batch_size:epoch*self.batch_size] = self.model.paths_reward[..., 0].numpy()
            self.memory['path_init_flow'][(epoch-1)*self.batch_size:epoch*self.batch_size] = self.model.path_init_flow[..., 0].numpy()


class fn_alpha_tune_grid(tf.keras.callbacks.Callback):
    def __init__(self, epochs=10, alpha_range=None, N_SAMPLE=16, pool_size=1, seed=1234,**kwargs):
        super(fn_alpha_tune_grid).__init__()
        self.epoch_per_train = epochs
        all_alpha = np.linspace(*alpha_range, N_SAMPLE, dtype='float32')
        alpha_list = [
            all_alpha[i * pool_size:(i + 1) * pool_size]
            for i in range(len(all_alpha) // pool_size)
        ]
        assert (all([len(x) == pool_size for x in alpha_list]))
        self.alpha_range = alpha_list
        self.current_experiments = -1
        self.metrics = []
        self.seed = seed
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.epoch_per_train == 0:
            self.current_experiments += 1
            self.model.reinitialize()
            self.model.reg_fn.alpha.assign(tf.math.exp(self.alpha_range[self.current_experiments]))


class fn_alpha_tune_search(tf.keras.callbacks.Callback):
    def __init__(self, epochs=10, flow_bang_threshold=10., flow_crush_threshold =0.01, reg_fn_alpha=None, N_SAMPLE=16, pool_size=1, seed=1234,**kwargs):
        super(fn_alpha_tune_grid).__init__()
        self.epoch_per_train = epochs
        self.min = [reg_fn_alpha[0]]*pool_size
        self.max = [reg_fn_alpha[1]]*pool_size
        self.flow_crush = [self.max]*pool_size
        self.flow_bang = [self.min]*pool_size
        self.sweet
        self.flow_bang_threshold = flow_bang_threshold
        self.flow_crush_threshold = flow_crush_threshold
        self.alpha_stack = []
        self.current_experiments = -1
        self.metrics = []
        self.seed = seed
        self.phase = 'crush'
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.epoch_per_train == 0:
            self.current_experiments += 1
            res = self.model.evaluate()
            res['FlowSize'] = res['FlowSize'].numpy()
            for i in range(self.pool_size):
                if res['FlowSize'][i] < self.flow_crush_threshold:
                    self.flow_crush[i] = self.model.reg_fn.alpha[i]
                if res['FlowSize'][i] > self.flow_bang_threshold:
                    self.flow_bang[i] = self.model.reg_fn.alpha[i]


            self.model.reinitialize()
            self.model.reg_fn.alpha.assign(self.alpha_range[self.current_experiments])

class FlowSizeStop(tf.keras.callbacks.Callback):
    def __init__(self, monitor="FlowSize", min_val=5*1e-1,max_val=1e4):
        super().__init__()
        self.monitor = monitor
        self.min_val = min_val
        self.max_val = max_val
        self.stopped = False

    def on_train_begin(self, logs=None):
        self.stopped = False
    def on_epoch_end(self, epoch, logs=None):
        if epoch < 20:
            return
        if self.get_monitor_value(logs) < self.min_val or self.get_monitor_value(logs) > self.max_val:
            self.model.stop_training = True
            self.stopped = True

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        return monitor_value


class MemoryUse(tf.keras.callbacks.Callback):
    def on_train_end(self, epoch, logs=None):
        if tf.config.list_physical_devices('GPU'):
          print(tf.config.experimental.get_memory_info('GPU:0')['peak']//2**20)


class LogProgress(tf.keras.callbacks.Callback):
    def __init__(self,logger):
        super().__init__()
        self.logger = logger
    def on_epoch_end(self, epoch, logs=None):
        self.logger.info('epoch %d' % (epoch, ))

tuning_method = {
    'fn_alpha_tune_grid' : fn_alpha_tune_grid,
    'fn_alpha_tune_search' : fn_alpha_tune_search,
}

