import tensorflow as tf

eta = 1e-3


class TrivialNu(tf.keras.Model):
    def __init__(self, name="TrivialNu", **kwargs):
        super(TrivialNu, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        """
        Trivial normalization (constant = 1.)
        """
        return tf.ones_like(flownu[..., 0])


class FOutStar(tf.keras.Model):
    def __init__(self, name="FOutStar", **kwargs):
        super(FOutStar, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        """
        $\omega = F_{out}^star$
        """
        return flownu[..., 1]


class FInit(tf.keras.Model):
    def __init__(self, name="FInit", **kwargs):
        super(FInit, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        """
        $\omega = f_{init} $
        """
        return flownu[..., 3]


class FInOut(tf.keras.Model):
    def __init__(self, name="FInOut", **kwargs):
        super(FInOut, self).__init__(name=name, **kwargs)

    def call(self, flownu, init):
        """
        \omega =  $\frac{F_{in}^*+F_{out}^*}{2}$
        """
        return tf.reduce_sum(flownu[..., :4], axis=-1) / 2


class PathwiseTotalFInOut(tf.keras.Model):
    def __init__(self, name="PathwiseTotalFInOut", **kwargs):
        super(PathwiseTotalFInOut, self).__init__(name=name, **kwargs)
        self.f_in_out = FInOut()
        self.trivial_nu = TrivialNu()

    def call(self, flownu, init):
        """
        Trivial normalization (constant = 1.)
        """
        scale = tf.reduce_sum(self.f_in_out(flownu, init), axis=1, keepdims=True)
        return eta + scale * self.trivial_nu(flownu, init)


class BatchmeanTotalFInOut(tf.keras.Model):
    def __init__(self, name="BatchmeanTotalFInOut", **kwargs):
        super(BatchmeanTotalFInOut, self).__init__(name=name, **kwargs)
        self.pathwise_total_f_in_out = PathwiseTotalFInOut()

    def call(self, flownu, init):
        """
        Trivial normalization (constant = 1.)
        """
        return tf.reduce_mean(self.pathwise_total_f_in_out(flownu, init), axis=0,
                              keepdims=True)


class PathwiseNormalizedFInOut(tf.keras.Model):
    def __init__(self, name="PathwiseNormalizedFInOut", **kwargs):
        super(PathwiseNormalizedFInOut, self).__init__(name=name, **kwargs)
        self.f_in_out = FInOut()
        self.trivial_nu = TrivialNu()

    def call(self, flownu, init):
        """
        Trivial normalization (constant = 1.)
        """
        f_in_out_val = self.f_in_out(flownu, init)
        normalization = tf.reduce_sum(tf.math.exp(flownu[..., 4]),
                                      axis=1, keepdims=True)
        scale = tf.reduce_sum(f_in_out_val / normalization, axis=1,
                              keepdims=True)
        return eta + scale * self.trivial_nu(flownu, init)


class BatchmeanNormalizedFInOut(tf.keras.Model):
    def __init__(self, name="BatchmeanNormalizedFInOut", **kwargs):
        super(BatchmeanNormalizedFInOut, self).__init__(name=name, **kwargs)
        self.pathwise_normalized_f_in_out = PathwiseNormalizedFInOut()
        self.trivial_nu = TrivialNu()

    def call(self, flownu, init):
        """
        Trivial normalization (constant = 1.)
        """
        return tf.reduce_mean(self.pathwise_normalized_f_in_out(flownu, init),
                              axis=0, keepdims=True)


normalization_flow_models = [
    TrivialNu(),
    FInOut(),
    FOutStar(),
    FInit(),
    BatchmeanTotalFInOut(),
    PathwiseTotalFInOut(),
    BatchmeanNormalizedFInOut(),
    PathwiseNormalizedFInOut(),
]