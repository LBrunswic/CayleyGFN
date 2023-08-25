import numpy as np
import itertools
import tensorflow as tf
from Groups.symmetric_groups import *

class CayleyGraphLinearEmb():
    def __init__(self,
                 initial,
                 direct_actions,
                 diameter,
                 random_gen=None,
                 name='cayley_graph',
                 dtype='float32',
            ):
        self.initial = tf.constant(initial,dtype=dtype)
        self.embedding_dim = tf.constant(initial.shape[0]) # always a (p,)
        self.actions = tf.constant(direct_actions,dtype=dtype)
        self.nactions = len(direct_actions)
        self.reverse_actions = tf.constant(np.linalg.inv(direct_actions),dtype=dtype)
        self.dtype = dtype
        self.name = name
        self.diameter = diameter
        self.random_gen = random_gen

    def random(self, batch_size):
        if self.random_gen is None:
            return tf.convert_to_tensor(np.full((batch_size, self.embedding_dim), self.initial))
        else:
            return self.random_gen(batch_size)
    def __str__(self):
        return self.name

def Symmetric(n,Gen='trans_cycle_a', inverse = False,dtype='float32',k=1):
    def aux(sigma):
        return np.unique(sigma,return_index=True)[1]
    def permutation_matrix(sigma):
        n = len(sigma)
        P = np.zeros((n, n))
        for i in range(n):
            P[i, sigma[i]] = 1

        S = np.zeros((k*n,k*n))
        for i in range(k):
            S[i*n:(i+1)*n,i*n:(i+1)*n] = P
        return S
    if k<n:
        initial = np.arange(k*n)-k*n/2
    elif k==n:
        initial = np.eye(n).reshape(-1)
    elif k==n+1:
        initial = np.concatenate([np.eye(n).reshape(-1),np.arange(n)])

    if Gen == 'trans_cycle_a':
        generators = [
            [1,0]+list(range(2,n)),
            list(range(1,n))+[0],
        ]
        if inverse:
            generators.append(aux(generators[1]))
        diameter = n
    elif Gen == 'transpositions':
        generators = [
            list(range(i))+[i+1,i]+list(range(i+2,n))
            for i in range(n-1)
        ]
        diameter = n
    elif Gen == 'all':
        generators = list(itertools.permutations(list(range(n))))
        diameter = 1
    elif Gen == '3_cycles':
        generators = [
            [1,2,0] + list(range(3, n)),
            # [1,0,2] + list(range(3, n)),
            list(range(1, n)) + [0],
            # [n-1]+list(range(0, n-1)),
        ]
        if inverse:
            generators.append(aux(generators[0]))
            generators.append(aux(generators[1]))
        diameter = int(n*np.log(1+n))
    elif Gen == 'large_cycles':
        p = 5
        generators = [
            list(range(0,i))+list(range(i+1,i+p)) + [i] + list(range(i+p,n))
            for i in range(0,n-p,4)
        ] + [list(range(0,n-p+1)) + list(range(n-p+2,n)) + [n-p+1]]
        if inverse:
            a = [aux(x) for x in generators]
            generators += a
        diameter = int(n)
        print(generators)
    else:
        raise NotImplementedError('The set of generator is unknown')
    direct_actions = np.zeros((len(generators),k*n,k*n))
    for i in range(len(generators)):
        direct_actions[i] = permutation_matrix(generators[i])
    random_gen = lambda b:random_perm(b,n)
    # random_gen = lambda b:random_perm_extremal(b,n)
    # random_gen = lambda b: np.concatenate([random_perm_extremal(b//2,n),random_perm(b-b//2,n)])
    return CayleyGraphLinearEmb(initial,direct_actions,diameter,random_gen=random_gen,name='Sym%s_%s' % (n,Gen))


def RubicksCube(width=3,Gen='default', inverse = False,dtype='float32',melange=200):
    n = 48
    def aux(sigma):
        return np.unique(sigma,return_index=True)[1]
    initial = np.arange(n)
    if Gen == 'default':
        generators = [
            cycle_dec_to_array(n,sigma,start=1,dtype=int)
            for sigma in [
                [(1, 3, 8, 6),(2, 5, 7, 4),(9, 33, 25, 17),(10, 34, 26, 18),(11, 35, 27, 19)],
                [(9, 11, 16, 14),(10, 13, 15, 12),(1, 17, 41, 40),(4, 20, 44, 37),(6, 22, 46, 35)],
                [(17, 19, 24, 22),(18, 21, 23, 20),(6, 25, 43, 16),(7, 28, 42, 13),(8, 30, 41, 11)],
                [(25, 27, 32, 30),(26, 29, 31, 28),(3, 38, 43, 19),(5, 36, 45, 21),(8, 33, 48, 24)],
                [(33, 35, 40, 38),(34, 37, 39, 36),(3, 9, 46, 32),(2, 12, 47, 29),(1, 14, 48, 27)],
                [(41, 43, 48, 46),(42, 45, 47, 44),(14, 22, 30, 38),(15, 23, 31, 39),(16, 24, 32, 40)],
            ]
        ]
        if inverse:
            generators.append(aux(generators[1]))
        diameter = 20
    else:
        raise NotImplementedError('The set of generator is unknown')
    direct_actions = np.zeros((len(generators),n,n),dtype='float32')
    for i in range(len(generators)):
        direct_actions[i] = permutation_matrix(generators[i])
    generators = np.array(generators)
    def random_gen(batch_size):
        base = np.full((batch_size,n),np.arange(n))
        indices = np.arange(len(generators),dtype=int)
        def kernel(x):
            c = np.random.choice(indices,size=batch_size)
            I = generators[c]
            return x[np.arange(batch_size).reshape(-1,1),I]
        for i in range(melange):
            base = kernel(base)
        return base
    return CayleyGraphLinearEmb(initial,direct_actions,diameter,random_gen=random_gen,name='Sym%s_%s' % (n,Gen))
#
#
# def build_representation(G):
#     def aux(x):
#         return str(list(tf.cast(x,tf.int32).numpy()))
#     def aux2(x):
#         return tf.convert_to_tensor(eval(x),dtype='float32')
#     dot = graphviz.Digraph(comment=G.name,strict=True,format='png')
#     node_explored = []
#     node_unexplored = []
#     node_unexplored.append(aux(G.initial))
#     Sa = aux(G.initial)
#     dot.node(Sa, Sa)
#     while node_unexplored:
#         Sa = node_unexplored.pop()
#         node_explored.append(Sa)
#         Next = list(tf.einsum('ijk,k->ij',G.actions,aux2(Sa)))
#         for s in Next:
#             sa = aux(s)
#             if sa not in node_explored + node_unexplored:
#                 dot.node(sa)
#                 node_unexplored.append(sa)
#             dot.edge(Sa,sa)
#     return dot,node_explored
