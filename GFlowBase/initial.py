import tensorflow as tf
import numpy as np
def initial_dot_gen(center):
    def initial(x):
        is_initial = np.all(np.abs(x - center) < 1e-3, axis=1).astype('float32')
        return is_initial
    return initial
