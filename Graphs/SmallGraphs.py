import numpy as np
import itertools

class SmallGraph():
    def __init__(self,adjacency_matrix,name):
        self.adjacency_matrix = adjacency_matrix
        self.Nstate = self.adjacency_matrix.shape[0]
        self.name=name

def HyperGrid(dimension,width,stationary=False):
    adjacency = np.zeros(tuple([width]*(2*dimension)))
    for I in  itertools.product(range(width),repeat = dimension):
        if stationary:
            adjacency[tuple(list(I) + list(I))] = 1
        for c in range(dimension):
            J = list(I)
            J[c] += 1
            if J[c]<width:
                adjacency[tuple(list(I)+J)]=1
            J[c]-= 2
            if J[c]>=0:
                adjacency[tuple(list(I)+J)]=1
    adjacency.reshape((-1,width**dimension))
    name = "HGD%sW%s" % (dimension,width)
    return SmallGraph(adjacency.reshape((-1,width**dimension)),name)


def HyperGridAcyclic(dimension,width,stationary=False):
    adjacency = np.zeros(tuple([width]*(2*dimension)))
    for I in  itertools.product(range(width), repeat = dimension):
        if stationary:
            adjacency[tuple(list(I) + list(I))] = 1
        for c in range(dimension):
            J = list(I)
            J[c] += 1
            if J[c]<width:
                adjacency[tuple(list(I)+J)]=1
            # J[c]-= 2
            # if J[c]>=0:
            #     adjacency[tuple(list(I)+J)]=1
    adjacency.reshape((-1,width**dimension))
    name = "acyclicHGD%sW%s" % (dimension, width)
    return SmallGraph(adjacency.reshape((-1,width**dimension)),name)



from itertools import permutations


def Symmetric(n,width,Gen='trans_cycle_a',dtype='float32'):
    def permutation_matrix(sigma):
        n = len(sigma)
        P = np.zeros((n, n),dtype='uint32')
        for i in range(n):
            P[i, sigma[i]] = 1
        return P

    initial = np.arange(n)
    if Gen == 'trans_cycle_a':
        generators = [
            [1,0]+list(range(2,n)),
            list(range(1,n))+[0],
            # [n-1]+list(range(0, n-1)),
        ]
        diameter = n
    else:
        raise NotImplementedError('The set of generator is unknown')
    direct_actions = np.zeros((len(generators),n,n),dtype='uint32')
    for i in range(len(generators)):
        direct_actions[i] = permutation_matrix(generators[i])

    states = np.array(list(permutations(list(range(n)))))
    pre_adjacency = np.einsum('aij,sj->sai',direct_actions,states)
    B = np.equal(
        np.expand_dims(pre_adjacency,2),
        states.reshape((1,1,states.shape[0],states.shape[1]))
    )
    B = np.all(B,axis=-1)
    adjacency = np.sum(B,axis=1)
    # print(states)
    return SmallGraph(adjacency,name='Sym%s_%s' % (n,Gen))