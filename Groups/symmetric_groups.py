import numpy as np
import tensorflow as tf

def permutation_matrix(sigma):
    n = len(sigma)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, sigma[i]] = 1
    return P

def translation_matrix(v):
    n = len(v)
    P = np.eye((n+1, n+1))
    P[:-1,-1] = v
    return P

def cycle_dec_to_array(n,cycles=[],start=0,dtype='float32'):
    res = np.arange(n,dtype=dtype)
    for omega in cycles:
        ell = len(omega)
        for i in range(ell):
            res[omega[i]-start] = omega[(i+1)%ell]-start
    return res




def rubick_generators(n,dtype='int32'):
    if n!=48:
        raise NotImplementedError('The size of the permutation group should be 48')
    generators = [
        cycle_dec_to_array(n,sigma,start=1,dtype=dtype)
        for sigma in [
            [(1, 3, 8, 6),(2, 5, 7, 4),(9, 33, 25, 17),(10, 34, 26, 18),(11, 35, 27, 19)],
            [(9, 11, 16, 14),(10, 13, 15, 12),(1, 17, 41, 40),(4, 20, 44, 37),(6, 22, 46, 35)],
            [(17, 19, 24, 22),(18, 21, 23, 20),(6, 25, 43, 16),(7, 28, 42, 13),(8, 30, 41, 11)],
            [(25, 27, 32, 30),(26, 29, 31, 28),(3, 38, 43, 19),(5, 36, 45, 21),(8, 33, 48, 24)],
            [(33, 35, 40, 38),(34, 37, 39, 36),(3, 9, 46, 32),(2, 12, 47, 29),(1, 14, 48, 27)],
            [(41, 43, 48, 46),(42, 45, 47, 44),(14, 22, 30, 38),(15, 23, 31, 39),(16, 24, 32, 40)],
        ]
    ]
    involutions = [False]*len(generators)
    diameter = 30
    return generators,involutions,diameter


def iteration_random(batch_size,n,depth,generators):
    base = tf.broadcast_to(tf.range(n),(batch_size,n))
    indices = tf.range(len(generators))
    p = tf.zeros(shape=(1,len(generators)))
    batch_indices = tf.broadcast_to(tf.reshape(tf.range(batch_size),(batch_size,1,1)),(batch_size,n,1))
    def kernel(x):
        c = tf.expand_dims(tf.random.categorical(p,batch_size),-1)[0]
        I = tf.expand_dims(tf.gather_nd(params=generators,indices=c),-1)
        I = tf.concat([batch_indices,I],axis=-1)
        return tf.gather_nd(params=x,indices=I)
    for i in range(depth):
        base = kernel(base)
    return base



def inversion(sigma):
    return np.unique(sigma,return_index=True)[1]


class SymmetricUniform:
    def __init__(self,n,group_dtype='int32',density_dtype='float32'):
        self.n = n
        self.group_dtype = group_dtype
        self.density_dtype = density_dtype
        # self.g = tf.random.Generator.from_seed(1234)
        # self.g = tf.random.default_rng(seed=1234)
        self.batch = {}
    def sample(self,shape):
        n = self.n
        return tf.cast(tf.argsort(tf.random.normal((*shape,n)),axis=-1),dtype=self.group_dtype)

    def density(self, position_batch, axis=-1):
        return tf.ones((*position_batch.shape[:axis],*position_batch.shape[axis+1:]),dtype=self.density_dtype)


class Modal:
    def __init__(self,n,modes,logits=None):
        self.n = n
        print(n)
        self.Nmodes = len(modes)
        self.dtype = 'int32'
        self.modes = tf.Variable(tf.reshape(modes,(-1,self.n)),trainable=False)
        if logits is None:
            self.logits =  tf.Variable(tf.zeros((1,self.Nmodes)))
        else:
            self.logits =  tf.Variable(tf.reshape(logits,(1,-1)))
        print(self.logits)
    @tf.function
    def sample(self,batch_size):
        return tf.gather(self.modes, tf.random.categorical(self.logits,batch_size,dtype=self.dtype)[0])
    def set_cutoff_logits(self,p):
        self.update_logits(
            [0.]*p + [-100.]*(self.Nmodes-p)
        )
    def update_logits(self,logits):
        self.logits.assign(tf.reshape(logits,(1,-1)))
    @tf.function
    def density(self,position_batch):
        A = tf.reshape(position_batch,(*position_batch.shape[:-1],1,self.n))
        B = self.modes
        B = tf.reshape(B,(1,1,*B.shape))
        return tf.einsum('...i,i',tf.cast(tf.reduce_all(A==B,axis=-1),'float32'),tf.math.exp(self.logits)[0])
