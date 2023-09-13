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

    # def on_epoch_end(self,epoch,logs=None):
        # flow = self.model

        # F=(flow.path_density_foo()[...,1:].numpy()*8).astype(int)
        # path_representation(
        #     flow.paths_true[:14,:30],
        #     flow=F,
        #     reward=self.reward,
        #     # reward=lambda x:np.all(x==np.arange(48)),
        #     filename='epoch_%s' % epoch,
        #     previous = 'epoch_%s' % (epoch-1),
        #     next = 'epoch_%s' % (epoch+1),
        #     folder = 'results',
        # )
