import numpy as np
import itertools
import tensorflow as tf
from Groups.symmetric_groups import *
from scipy.linalg import block_diag
import os

D_TYPE = 'float32'
class CayleyGraphLinearEmb():
    def __init__(self,
                 direct_actions,
                 inverse_actions,
                 diameter,
                 embedding=None,
                 random_gen=None,
                 group_dim=None,
                 embedding_dim=None,
                 rep_list=None,
                 generators=None,
                 name='cayley_graph',
                 group_dtype='int32',
                 representation_dtype='float32',
            ):

        self.nactions = len(direct_actions)
        self.actions = tf.constant(direct_actions,dtype=group_dtype)
        # print(self.actions)
        self.generators = generators
        self.reverse_actions = tf.constant(inverse_actions,dtype=group_dtype)
        # print(self.reverse_actions)

        self.group_dim = group_dim
        self.embedding_dim = tf.constant(embedding_dim) # always a (p,)
        self.group_dtype = group_dtype
        self.representation_dtype = representation_dtype
        self.representation = rep_list[0][0]


        self.name = name
        self.diameter = diameter
        self.random_gen = random_gen
        self.embedding_fn = embedding

    def iter(self,depth):
        N = self.group_dim
        base = np.arange(N,dtype=self.group_dtype).reshape(1,-1)
        for i in range(depth):
            old = [tuple(x) for x in base]
            old_set = set(old)
            new = list(set([tuple(x) for x in base[...,self.generators].reshape(-1,N) if tuple(x) not in old_set]))
            base = np.array(old+new)
        return tf.constant(np.array([self.representation(g) for g in  base],dtype=self.group_dtype))

    def sample(self, shape,axis=-1):
        if isinstance(shape,int):
            shape = (shape,)
        samples = self.random_gen.sample(shape)
        size = len(shape)+1
        if axis < 0:
            axis = size + axis
        sigma = list(range(axis)) + [size-1] + list(range(axis, size-1))
        return tf.transpose(samples, sigma)

    def density(self, state_batch, axis=-1):
        return self.random_gen.density(state_batch, axis=axis)

    def embedding(self,state_batch,axis=-1):
        state_batch_swap = tf.experimental.numpy.swapaxes(state_batch, -1, axis)
        res_swap = self.embedding_fn(state_batch_swap)
        return tf.experimental.numpy.swapaxes(res_swap, -1, axis)

    def __str__(self):
        return self.name

D = 2
def cycle(length,start,n):
    assert(start>=0 and start<n)
    sigma = list(range(start))+list(range(start+1,start+length)) + [start] + list(range(start+length,n))
    sigma = [x%n for x in sigma]
    if start == n-1:
        sigma = sigma[1:]
    # print(sigma)
    return sigma
SymmetricGenerators = {
    'trans_cycle_a' : lambda n: ([[1,0]+list(range(2,n)), list(range(1,n))+[0]],[True,False],n),
    'cycles_a' : lambda n: ([cycle(k,n-k,n) for k in range(2,n+1)],[True]+[False for k in range(3,n+1)],n),
    'transpositions' : lambda n: ([cycle(2,i,n) for i in range(n)], [True]*(n),n),
    'Hypergrid' : lambda n: ([
        cycle(n//D,i*(n//D),n)  for i in range(D)], [False]*(D),n),
    'all' : lambda n: (list(itertools.permutations(list(range(n)))), [True]*np.math.factorial(n),1),
    'Rubicks': rubick_generators,
}
Representations = {
    # name -> size ->  (morphism, dim)
    'natural' : lambda n : (permutation_matrix,n),
    'affinenatural' : lambda n : (translation_matrix,n+1),
}

rubick_gen = tf.constant(rubick_generators(48)[0])

pi = np.math.pi
def hot(n,*args,omega=1,**kwarg):
    def aux(paths):
        return tf.reshape(tf.one_hot(paths,n),(*paths.shape[:-1], n*n,))
    return aux,n*n
pi = np.math.pi
def hotalpha(n,*args,omega=1,dtype=D_TYPE,**kwarg):
    alpha = tf.reshape(tf.math.exp(-10*np.arange(n,dtype=dtype)),(n,1))
    Id = tf.eye(n)*alpha
    def aux(paths):
        return tf.cast(tf.reshape(tf.gather(Id,paths),(*paths.shape[:-1], n*n,) ),dtype)
    return aux,n*n

Embeddings = {
    'natural': lambda n,*args,dtype=D_TYPE,**kwarg: (lambda x: tf.cast(x,dtype)/tf.cast(n,dtype),n),
    'cos': lambda n,*args,omega=1,dtype=D_TYPE,**kwarg: (lambda x: tf.math.cos(2*omega*pi*tf.cast(x,dtype)/n),n),
    'sin': lambda n,*args,omega=1,dtype=D_TYPE,**kwarg: (lambda x: tf.math.sin(2*omega*pi*tf.cast(x,dtype)/n),n),
    'hot': hot,
    'hotalpha': hotalpha
}

def Symmetric(
    n,
    generators='trans_cycle_a',
    representation = [('natural',{})],
    inverse = False,
    random_gen = None,
    embedding = [('natural',{})],
    group_dtype='int32'
):
    assert(random_gen is not None)
    generators,involutions,diameter = SymmetricGenerators[generators](n)
    if inverse:
        inverse_generators = []
        for i,gen in enumerate(generators):
            if not involutions[i]:
                inverse_generators.append(inversion(gen))
        generators += inverse_generators
    generators = tf.constant(generators)
    R_list = [Representations[rep](n) for rep,options in representation]

    direct_actions = tf.cast(tf.stack([
        block_diag(*[rho(gen) for rho,_ in R_list])
        for gen in generators
    ]),group_dtype)
    inverse_actions = tf.cast(tf.stack([
        block_diag(*[rho(inversion(gen)) for rho,_ in R_list])
        for gen in generators
    ]),group_dtype)


    E_list = [Embeddings[emb_choice](n,**emb_kwarg) for emb_choice,emb_kwarg in embedding]
    embedding_dim = sum([d for _,d in E_list])
    @tf.function
    def iota(x):
         return tf.concat([emb(x) for emb,_  in E_list],axis=-1)

    return CayleyGraphLinearEmb(direct_actions, inverse_actions, diameter, generators=generators,rep_list=R_list,random_gen=random_gen,embedding=iota,embedding_dim=embedding_dim, group_dim=n, name='Sym%s_%s' % (n,generators.numpy()),group_dtype=group_dtype,representation_dtype=D_TYPE)

