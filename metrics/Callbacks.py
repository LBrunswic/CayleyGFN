import sys
import os
import tensorflow as tf
import numpy as np
from time import time

from datetime import datetime

class ReplayBuffer(tf.keras.callbacks.Callback):
    def __init__(self, monitor="ReplayBuffer", folder='Knowledge'):
        super().__init__()
        self.folder = folder
        self.episode_memory = []
    def generate_episode(self,model):
        return
    def gen(self):
        while True:
            yield

    def on_epoch_begin(self, epoch, logs=None):
        def schedule(epoch):
            if epoch<10:
                return [
                    (0.,1.),
                    (0.9,0.1),
                    (0.9,0.1),
                    (0.9,0.1),
                    (0.9,0.1),
                    (0.9,0.1),
                    (0.99, 0.01),
                    (0.99, 0.01),
                    (0.99, 0.01),
                    (0.99, 0.01),
                    (0.99, 0.01),
                    (0.99, 0.01),
                    (0.99, 0.01),
                ][epoch]
            else:
                return (0.99, 0.01)
        # print('Update training dist...')
        T = time()
        self.model.update_training_distribution(exploration=0.,alpha=schedule(epoch-1))
        # print('done in %s seconds' % (time()-T))


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

