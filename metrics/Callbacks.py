import sys
import os
import tensorflow as tf
import numpy as np
from time import time
from collections import Counter
from Graphs.CayleyGraph import path_representation

from datetime import datetime

class ReplayBuffer(tf.keras.callbacks.Callback):
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
        print('Update training dist...')
        T = time()
        self.model.update_training_distribution(exploration=0.,alpha=schedule(epoch-1) )
        print('done in %s seconds' % (time()-T))


class FlowSizeStop(tf.keras.callbacks.Callback):
    def __init__(self, monitor="FlowSize", min_val=1e-4,max_val=1e4):
        super().__init__()
        self.monitor = monitor
        self.min_val = min_val
        self.max_val = max_val
        self.stopped = False

    def on_train_begin(self, logs=None):
        self.stopped = False
    def on_epoch_end(self, epoch, logs=None):
        if self.get_monitor_value(logs) < self.min_val or self.get_monitor_value(logs) > self.max_val:
            self.model.stop_training = True
            self.stopped = True

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        # if monitor_value is None:
        #     logging.warning(
        #         "Early stopping conditioned on metric `%s` "
        #         "which is not available. Available metrics are: %s",
        #         self.monitor,
        #         ",".join(list(logs.keys())),
        #     )
        return monitor_value