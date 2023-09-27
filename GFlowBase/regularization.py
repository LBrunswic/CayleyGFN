import tensorflow as tf
@tf.function
def straight_reg(self,loss_gradients,reg_gradients):
    n_train =  self.n_train
    res = [loss_gradients[i]+reg_gradients[i]  for i in range(n_train)]
    return res


@tf.function
def proj_reg(self,loss_gradients,reg_gradients):
    n_train =  self.n_train
    res = [
        loss_gradients[i] + reg_gradients[i] -  projection_orth(reg_gradients[i],loss_gradients[i])
        for i in range(n_train)
    ]
    return res


@tf.function
def projection_orth(u,v):
    v_norm = tf.math.l2_normalize(v)
    return tf.reduce_sum(u*v_norm)*v_norm


class LogPathLen_gen(tf.keras.Model):
    def __init__(self,alpha,logpmin,name='LogPathLen_fn',**kwargs):
        super(LogPathLen_gen, self).__init__(name=name, **kwargs)
        self.alpha = tf.Variable(tf.math.exp(alpha), trainable=False, dtype='float32')
        self.logpmin = tf.Variable(logpmin, trainable=False, dtype='float32')
    @tf.function
    def call(self,Flownu):
        logdensity_trainable = Flownu[..., 4]
        Ebigtau = tf.reduce_mean(logdensity_trainable[:,-1])
        return self.alpha*tf.nn.relu(self.logpmin+Ebigtau)



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
            tf.stop_gradient(density[:,:-1]) *
            tf.nn.relu(self.logpmin+tf.math.log(1e-20 + tf.abs(tau_distribution - path_reward / total_reward))),
            axis = 1
        ))
        return self.alpha*accuracy


class Norm2_gen(tf.keras.Model):
    def __init__(self, alpha, logpmin, **kwargs):
        super(Norm2_gen, self).__init__(name='Norm2_fn', **kwargs)
        self.alpha = tf.Variable(tf.math.exp(alpha), trainable=False, dtype='float32')
        self.logpmin = tf.Variable(logpmin, trainable=False, dtype='float32')
    @tf.function
    def call(self,Flownu):
        return self.alpha * tf.reduce_mean(tf.linalg.norm(Flownu[..., 0]+Flownu[..., 1],ord=2,axis=1))

