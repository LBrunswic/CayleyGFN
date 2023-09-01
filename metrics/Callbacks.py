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
        self.model.update_training_distribution(exploration=0.)
    def on_epoch_end(self,epoch,logs=None):
        flow = self.model

        F=(flow.path_density[...,1:].numpy()*8).astype(int)
        path_representation(
            flow.paths_true[:14,:30],
            flow=F,
            reward=self.reward,
            # reward=lambda x:np.all(x==np.arange(48)),
            filename='epoch_%s' % epoch,
            previous = 'epoch_%s' % (epoch-1),
            next = 'epoch_%s' % (epoch+1),
            folder = 'results',
        )
