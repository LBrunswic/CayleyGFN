import tensorflow as tf


@tf.function
def mlog(x):
    return -tf.math.log(x)


@tf.function
def KL(x):
    return x * tf.math.log(x)


@tf.function
def chi2(x):
    return tf.math.square(1 - x)


@tf.function
def Hellinger(x):
    return tf.math.square(1 - tf.sqrt(x))


@tf.function
def TV(x):
    return tf.math.abs(1 - x)


@tf.function
def LeCam(x):
    return (1 - x) / (2 * x + 2)


@tf.function
def JensenShannon(x):
    return x * tf.math.log(2 * x / (x + 1)) + tf.math.log(2 / (1 + x))


@tf.function
def Bengio(x):
    return tf.square(tf.math.log(x))


@tf.function
def Alog(x):
    return tf.math.log(1 + tf.math.abs(x))


def Alogsquare(alpha=2, delta=1.):
    @tf.function
    def aux(x):
        return tf.math.log(1 + delta * tf.pow(tf.math.abs(x), alpha))
    return aux


class Apower(tf.keras.models.Model):
    def __init__(self, alpha=2., **kwargs):
        super(Apower, self).__init__(**kwargs)
        self.alpha = tf.Variable(alpha, trainable=False)
    def HP(self):
        return {'alpha' :self.alpha.numpy()}

    def call(self, x):
        return tf.math.pow(1e-20 + tf.math.abs(x), self.alpha)


class Bpower(tf.keras.models.Model):
    def __init__(self, alpha=2., beta=0.1, delta=1e-8, **kwargs):
        super(Bpower, self).__init__(**kwargs)
        self.alpha = tf.Variable(alpha, trainable=False)
        self.delta = tf.Variable(delta, trainable=False)
        if beta is not None:
            self.beta = tf.Variable(tf.exp(beta), trainable=False)
        if beta is None:
            self.beta = tf.Variable(0., trainable=False)
    def HP(self):
        return {
            'alpha': self.alpha.numpy(),
            'delta': self.delta.numpy(),
            'beta': self.beta.numpy(),
        }

    def call(self, x, y):
        return (1 + self.beta * tf.math.pow(self.delta + x + y, self.alpha))


class divergence(tf.keras.losses.Loss):
    def __init__(self, g=Bengio, delta=1e-10, name='plop', **kwargs):
        super(divergence, self).__init__(name=name, **kwargs)
        self.g = g
        self.delta = delta

    @tf.function
    def call(self, Flownu, bis):
        Finstar = Flownu[..., 0]
        Foutstar = Flownu[..., 1]
        R = Flownu[..., 2]
        Finit = Flownu[..., 3]
        logdensity_trainable = Flownu[..., 4]
        density_fixed = tf.stop_gradient(tf.math.exp(logdensity_trainable))
        return tf.reduce_mean(tf.reduce_sum(density_fixed * self.g(tf.math.abs((self.delta + Finstar + Finit) / (self.delta + Foutstar + R))), axis=-1))


@tf.function
def Rbar_Error(paths_reward, density, delta=1e-8):
    RRbar = tf.reduce_mean(tf.reduce_sum(paths_reward * density, axis=1), axis=0)
    RR = tf.reduce_sum(tf.math.square(paths_reward)) / tf.reduce_sum(paths_reward)
    return tf.math.log(delta + tf.abs(RRbar - RR) / (delta + RR))


class MeanABError(tf.keras.losses.Loss):
    def __init__(self,
            A=tf.math.square,
            B=lambda x, y: 1,
            normalization_fn=None,
            normalization_nu_fn=None,
            cutoff=None,
            name='loss',
            **kwargs
        ):
        super(MeanABError, self).__init__(name=name, **kwargs)
        self.A = A
        self.B = B
        self.normalization_fn = normalization_fn
        self.normalization_nu_fn = normalization_nu_fn
        self.cutoff = cutoff

    def HP(self):
        HP = {}
        HP_A = self.A.HP()
        HP.update({f'A_{key}' : HP_A[key]  for key in HP_A})
        HP_B = self.B.HP()
        HP.update({f'B_{key}' : HP_B[key]  for key in HP_B})
        HP.update({'normalization_fn': self.normalization_fn.name})
        HP.update({'normalization_nu_fn': self.normalization_nu_fn.name})
        HP.update({'cutoff': self.cutoff})
        return HP

    @tf.function
    def call(self, Flownu, finit):
        Finstar = Flownu[..., 0]
        Foutstar = Flownu[..., 1]
        R = Flownu[..., 2]
        Finit = Flownu[..., 3]
        logdensity_trainable = Flownu[..., 4]
        Fin = Finstar + Finit
        Fout = Foutstar + R
        cutoff = 1.
        normalization = tf.stop_gradient(self.normalization_fn(Flownu, finit))
        normalization_nu = self.normalization_nu_fn(Flownu, finit)
        density_fixed = tf.stop_gradient(normalization_nu)
        Ldelta = tf.reduce_sum(
            tf.reduce_mean(
                tf.reduce_sum(
                    cutoff
                    * density_fixed
                    * self.A((Fout - Fin) / normalization)
                    * self.B(Fin / normalization, Fout / normalization),
                    axis=1
                ),
                axis=0
            )
        )
        print(Ldelta.shape)
        return Ldelta


cutoff_fns = {
    'none': lambda flownu, finit: 1.,
}
