import sys
import os
import tensorflow as tf
import numpy as np
from time import time
import pandas
from datetime import datetime


class PandasRecord(tf.keras.callbacks.Callback):
    def __init__(self,hparams,alpha_range,epoch_period = 10):
        self.epoch_period = epoch_period
        self.results = None
        self.hparams = hparams
        self.nflow = hparams['pool_size']
        self.alpha_range = alpha_range
    def on_epoch_end(self, epoch, logs=None):
        # print(epoch)
        res = self.model.evaluate()
        episode = epoch//self.epoch_period
        true_epoch = 1+epoch % self.epoch_period
        res.update({key: [self.hparams[key]]*self.nflow for key in self.hparams if key != 'reg_fn_alpha'})
        res.update({'epoch': [true_epoch]*self.nflow})
        res.update({'reg_fn_alpha': self.alpha_range[episode]})
        res.update({'episode': [episode]*self.nflow })
        res = pandas.DataFrame(res)

        if self.results is None:
            self.results = res
        else:
            res.index = res.index + self.results.index.stop
            print('new results')
            print(type(self.results))
            print(type(res))
            self.results = pandas.concat([self.results,res])


class ReplayBuffer(tf.keras.callbacks.Callback):
    def __init__(self, monitor="ReplayBuffer", folder='Knowledge',epochs=10):
        super().__init__()
        self.folder = folder
        self.episode_memory = []
        self.epochs = epochs


    def on_train_begin(self, logs=None):
        self.compile_update_training_distribution()

    # def on_epoch_end(self, epoch, logs=None):
    #     print('ReplayBuffer', self.model.evaluate()['EMaxSeenRew'])
    def on_epoch_begin(self, epoch, logs=None):

        def schedule(epoch):
            if epoch%self.epochs == 0:
                return (0., 1.)
            elif epoch%self.epochs < 6:
                return (0.90, 0.1)
            else:
                return (0.99, 0.01)
        self.update_training_distribution_callback()
        self.model.update_flow_init(schedule(epoch))

    @tf.function
    def update_training_distribution_callback_aux(self, initial):
        print('RECOMPILE UPDATE',initial.shape)
        for j in tf.range(self.model.grad_batch_size):
            seeded_uniform = tf.random.uniform(
                shape=(self.model.batch_size, self.model.path_length - 1, self.model.ncopy), dtype='float32')
            true_paths, embedded_paths = self.gen_path_model(
                tf.concat([seeded_uniform, tf.cast(initial[j], 'float32')], axis=1))
            self.model.paths_true[j].assign(true_paths)
            self.model.paths[j].assign(embedded_paths)
            self.model.update_training_distribution_gflow()

    def update_training_distribution_callback(self):
        initial = self.model.graph.sample(shape=(self.model.grad_batch_size, self.model.batch_size, self.model.ncopy), axis=-2)
        self.update_training_distribution_callback_aux(initial)



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
    def __init__(self, epoch_per_train=10, alpha_range=None, seed=1234,**kwargs):
        super(fn_alpha_tune_grid).__init__()
        self.epoch_per_train= epoch_per_train
        self.alpha_range = alpha_range
        self.current_experiments = -1
        self.metrics = []
        self.seed = seed
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.epoch_per_train == 0:
            self.current_experiments += 1
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


fn_alpha_tune = {
    'fn_alpha_tune_grid' : fn_alpha_tune_grid
}