import tensorflow as tf
@tf.function
def straight_reg(self,loss_gradients,reg_gradients,scaling=1.0):
    n_train = self.n_train
    res = [loss_gradients[i] + scaling*reg_gradients[i] for i in range(n_train)]
    return res


@tf.function
def proj_reg(self, loss_gradients, reg_gradients,scaling=1.0):
    n_train = self.n_train
    res = [
        loss_gradients[i]
        + scaling * (reg_gradients[i] - projection_orth(reg_gradients[i], loss_gradients[i]))
        for i in range(n_train)
    ]
    return res

@tf.function
def scaled_proj_reg(self, loss_gradients, reg_gradients,scaling=1.0):
    n_train = self.n_train
    res = [
        loss_gradients[i]
        + scaling * tf.linalg.norm(loss_gradients[i]) * tf.math.l2_normalize(
            reg_gradients[i] - projection_orth(reg_gradients[i], loss_gradients[i])
        )
        for i in range(n_train)
    ]
    return res


@tf.function
def projection_orth(u, v):
    """Compute the orthogonal projection of u on v
    """

    v_norm = tf.math.l2_normalize(v)
    return tf.reduce_sum(u*v_norm)*v_norm


class LogPathLen_gen(tf.keras.Model):
    def __init__(self, alpha,logpmin,name='LogPathLen_fn', **kwargs):
        super(LogPathLen_gen, self).__init__(name=name, **kwargs)
        self.alpha = tf.Variable(tf.math.exp(alpha), trainable=False, dtype='float32')
        self.logpmin = tf.Variable(logpmin, trainable=False, dtype='float32')
    @tf.function
    def call(self,Flownu):
        logdensity_trainable = Flownu[..., 4]
        Ebigtau = tf.reduce_mean(logdensity_trainable[:, -1],axis=0)
        return tf.reduce_sum(tf.nn.relu(self.logpmin + Ebigtau))

class LogEPathLen_gen(tf.keras.Model):
    def __init__(self, alpha,logpmin,name='LogPathLen_fn', **kwargs):
        super(LogEPathLen_gen, self).__init__(name=name, **kwargs)
        self.alpha = tf.Variable(tf.math.exp(alpha), trainable=False, dtype='float32')
        self.logpmin = tf.Variable(logpmin, trainable=False, dtype='float32')
    @tf.function
    def call(self,Flownu):
        logdensity_trainable = Flownu[..., 4]
        Ebigtau = tf.math.log(tf.reduce_mean(tf.exp(logdensity_trainable[:, -1]), axis=0))
        return tf.reduce_sum(tf.nn.relu(self.logpmin + Ebigtau))



class PathAccuracy_gen(tf.keras.Model):
    def __init__(self, alpha, logpmin, name='PathAccuracy_fn',**kwargs):
        super(PathAccuracy_gen, self).__init__(name=name, **kwargs)
        self.alpha = tf.Variable(tf.math.exp(alpha),trainable=False,dtype='float32')
        self.logpmin = tf.Variable(logpmin,trainable=False,dtype='float32')
    @tf.function
    def call(self, flownu):
        logdensity = flownu[..., 4]
        path_reward = flownu[..., 5]
        total_reward = tf.reduce_sum(path_reward, axis=1, keepdims=True)
        batch_size, path_length = logdensity.shape
        density = tf.concat([tf.exp(logdensity), tf.zeros_like(logdensity[:, :1])],axis=1)
        tau_distribution = density[..., :-1] - density[..., 1:]
        accuracy = tf.reduce_mean(tf.reduce_sum(
            tf.stop_gradient(density[:, :-1]) *
            tf.nn.relu(self.logpmin+tf.math.log(1e-20 + tf.abs(tau_distribution - path_reward / total_reward))),
            axis=1
        ))
        return accuracy


class Norm_gen(tf.keras.Model):
    """Regularization class implementing a scaled L2 norm

    """
    def __init__(self,logpmin=None, alpha=2, beta=1, **kwargs):
        super(Norm_gen, self).__init__(name='Norm2_fn', **kwargs)
        self.alpha = tf.Variable(alpha, trainable=False, dtype='float32')
        self.beta = tf.Variable(beta, trainable=False, dtype='float32')

    @tf.function
    def call(self,Flownu):
        return tf.reduce_sum(
            tf.reduce_mean(tf.linalg.norm(Flownu[..., 0] + Flownu[..., 1], ord=self.alpha, axis=1)**self.beta, axis=0)
        )


reg_post_choices = {
    'OrthReg': proj_reg,
    'ScaledOrthReg': scaled_proj_reg,
    'AddReg':  straight_reg,
}

reg_fn_gen_choices = {
    'LogPathLen': LogPathLen_gen,
    'LogEPathLen': LogEPathLen_gen,
    'norm2': Norm_gen,
}
