import sys
import os
import tensorflow as tf
import numpy as np
from time import time
import pandas
from datetime import datetime
from utils.utils import concat_dict_of_ndarray

class PandasRecord(tf.keras.callbacks.Callback):
    def __init__(self,hparams,epoch_period = 10):
        self.epoch_period = epoch_period
        self.results = []
        self.hparams = hparams
        self.nflow = hparams['pool_size']
    def on_epoch_end(self, epoch, logs=None):
        # print(epoch)
        res = self.model.evaluate()
        episode = epoch//self.epoch_period
        true_epoch = 1+epoch % self.epoch_period
        res.update({
            key: [self.hparams[key]]*self.nflow
            for key in self.hparams
            if not isinstance(self.hparams[key],tuple)
        })
        res.update({'epoch': np.array([true_epoch]*self.nflow)})
        res.update({'episode': [episode] * self.nflow})
        res.update({'reg_fn_alpha': self.model.reg_fn.alpha.numpy()})
        for key in self.hparams:
            if isinstance(self.hparams[key],tuple):
                res.update({key: [str(self.hparams[key])]*self.nflow })

        for key in res:
            if isinstance(res[key],tf.Tensor):
                res[key] = res[key].numpy()
            if isinstance(res[key],list):
                res[key] = np.array(res[key])
            if len(res[key].shape)>1:
                pass
                # print(key,res[key].shape)
                # print(res[key])
        self.results.append(res)
    def on_train_end(self, logs=None):
        for x in self.results:
            print('range:', x['alpha_range'])
            print('embedding:', x['embedding'])
        self.results = pandas.DataFrame(concat_dict_of_ndarray(self.results))

class ReplayBuffer(tf.keras.callbacks.Callback):
    def __init__(self, seeded_uniform=None,seeded_initial=None, pool_size=1,monitor="ReplayBuffer", folder='Knowledge',epoch_per_train=10):
        super().__init__()
        self.folder = folder
        self.episode_memory = []
        self.epoch_per_train = epoch_per_train
        self.pool_size = pool_size
        self.seeded_uniform = tf.Variable(seeded_uniform,trainable=False)
        self.seeded_initial = tf.Variable(seeded_initial,trainable=False)


    def on_train_begin(self, logs=None):
        self.compile_update_training_distribution()


    # def on_epoch_end(self, epoch, logs=None):
    #     print('ReplayBuffer', self.model.evaluate()['EMaxSeenRew'])
    def on_epoch_begin(self, epoch, logs=None):

        def schedule(epoch):
            if epoch%self.epoch_per_train == 0:
                return (0., 1.)
            elif epoch%self.epoch_per_train < 6:
                return (0.90, 0.1)
            else:
                return (0.99, 0.01)
        self.update_training_distribution_callback(epoch)
        self.model.update_flow_init(schedule(epoch))


    @tf.function
    def update_training_distribution_callback_aux(self, initial,seeded_uniform):
        print('RECOMPILE UPDATE',initial.shape)
        for j in tf.range(self.model.grad_batch_size):
            true_paths, embedded_paths = self.gen_path_model(
                tf.concat([seeded_uniform[j], tf.cast(initial[j], 'float32')], axis=1))
            self.model.paths_true[j].assign(true_paths)
            self.model.paths[j].assign(embedded_paths)
            self.model.update_training_distribution_gflow()

    def update_training_distribution_callback(self,epoch):
        initial = self.seeded_initial[epoch%self.epoch_per_train]
        initial = tf.broadcast_to(initial,(*initial.shape[:-1], self.pool_size) )
        seeded_uniform = self.seeded_uniform[epoch%self.epoch_per_train]
        seeded_uniform = tf.broadcast_to(seeded_uniform, (*seeded_uniform.shape[:-1], self.pool_size))
        self.update_training_distribution_callback_aux(initial, seeded_uniform)



    def compile_update_training_distribution(self):
        print('COMPILE update train dist')
        self.permute = tf.keras.layers.Permute((2, 1))

        # self.embedding_layer = tf.keras.layers.Lambda(lambda x:self.model.graph.embedding_fn(x,axis=-2))
        self.embedding_layer = self.model.graph.embedding_fn
        self.Categorical = lambda F_r: tf.gather(
            self.model.graph.actions,
            tf.argmin(
                tf.cumsum(F_r[:, :-1], axis=1) / tf.reduce_sum(F_r[:, :-1], axis=1,
                                                                                   keepdims=True) < F_r[:, -1:],
                axis=1
            )
        )

        self.flow_base = tf.keras.Sequential([
            tf.keras.layers.Reshape((1, 1, self.model.embedding_dim, self.model.ncopy)),
            self.model.FlowEstimator,
            tf.keras.layers.Lambda(lambda x: x[:, 0, 0], name='Squeeze')
        ])



        self.apply_action = tf.keras.layers.Lambda(
            lambda gathered, pos: tf.linalg.matvec(gathered, pos, name='ApplyAction'))

        seeded_uniform_positions = tf.keras.layers.Input(
            shape=(self.model.path_length - 1 + self.model.graph.group_dim, self.model.ncopy))
        seeded_uniform, position = tf.split(seeded_uniform_positions, [self.model.path_length - 1, self.model.graph.group_dim],
                                            axis=1)
        positions = [tf.cast(position, self.model.graph.group_dtype)]  # bjc
        embedded_positions = [self.embedding_layer(positions[-1])]  # (bec)
        actions = []
        seeded_uniform_split = tf.split(seeded_uniform, self.model.path_length - 1, axis=1)
        for i in range(self.model.path_length - 1):
            F_r = tf.concat([self.flow_base(embedded_positions[-1]), seeded_uniform_split[i]], axis=1)
            gathered = tf.cast(self.Categorical(F_r), self.model.graph.group_dtype ) # bcij
            prev_pos = tf.cast(tf.transpose(positions[-1], perm=(0, 2, 1)), self.model.graph.group_dtype )
            positions.append(
                tf.cast(self.permute(tf.linalg.matvec(gathered, prev_pos, name='ApplyAction')),self.model.graph.group_dtype)  # bcij,bcj
            )
            embedded_positions.append(self.embedding_layer(positions[-1]))

        outputs = tf.stack(positions, axis=1), tf.stack(embedded_positions, axis=1)
        self.gen_path_model = tf.keras.Model(inputs=seeded_uniform_positions, outputs=outputs, name='Gen')

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
            self.model.reg_fn.alpha.assign(self.alpha_range[self.current_experiments])


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

