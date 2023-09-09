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
def reg_fn_gen(alpha,logpmin):
    alpha = tf.math.exp(alpha)
    @tf.function
    def reg_fn(Flownu):
        logdensity_trainable = Flownu[..., 4]
        Ebigtau = tf.reduce_mean(logdensity_trainable[:,-1])
        return alpha*tf.nn.relu(logpmin+Ebigtau)
    return reg_fn



@tf.function
def reg_withcutoff_fn(Flownu):
    logdensity_trainable = Flownu[..., 4]
    density_fixed = tf.stop_gradient(tf.math.exp(logdensity_trainable))
    Expected_length = tf.reduce_mean(tf.reduce_sum(density_fixed,axis=1))
    cutoff = tf.cast(tf.reduce_min([3*Expected_length,density_fixed.shape[1]])-1,dtype='int32')
    Ebigtau = tf.reduce_mean(logdensity_trainable[:,cutoff])
    return tf.nn.relu(40+Ebigtau)