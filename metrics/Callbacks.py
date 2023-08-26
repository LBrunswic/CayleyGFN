import sys
import os
import tensorflow as tf
import numpy as np
from time import time
from collections import Counter

from datetime import datetime

class ReplayBuffer(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.update_training_distribution()
