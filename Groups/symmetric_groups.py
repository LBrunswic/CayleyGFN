import numpy as np

def permutation_matrix(sigma):
    n = len(sigma)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, sigma[i]] = 1
    return P

def cycle_dec_to_array(n,cycles=[],start=0,dtype='float32'):
    res = np.arange(n,dtype=dtype)
    for omega in cycles:
        ell = len(omega)
        for i in range(ell):
            res[omega[i]-start] = omega[(i+1)%ell]-start
    return res

def random_perm(batch_size, n):
    perm = np.zeros((batch_size, n), dtype=int)
    admissible = np.full_like(perm, np.arange(n), dtype=int)
    for i in range(n):
        indices = np.full((batch_size, n - i), np.arange(n - i), dtype=int)
        choice = np.random.randint(0, n - i, (batch_size, 1))
        perm[:, i] = admissible[np.where(indices == choice)]
        admissible = admissible[np.where(indices != choice)].reshape((batch_size, -1))
    return perm

def random_perm_extremal(batch_size, n):
    return np.concatenate([random_perm(batch_size,n-1)+1,np.zeros((batch_size,1))],axis=1)

def random_perm_from_gen(batch_size, generators,iteration=100,stalk=None):
    n = generators[0].shape[0]
    generators = generators.numpy()
    g = generators.shape[0]
    perm = np.zeros((batch_size, n), dtype=int)
    if stalk is None:
        stalk = np.arange(n)
    admissible = np.full_like(perm, stalk, dtype=int)
    moves = np.random.randint(g,size=(batch_size,iteration))
    for i in range(batch_size):
        for j in range(iteration):
            admissible[i] = generators[moves[i,j]]@admissible[i]
    return admissible


def rubick_generators(n):
    if n!=48:
        raise NotImplementedError('The size of the permutation group should be 48')
    return [
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

def iteration_random(batch_size,n,depth,generators):
    base = np.full((batch_size,n),np.arange(n))
    indices = np.arange(len(generators),dtype=int)
    def kernel(x):
        c = np.random.choice(indices,size=batch_size)
        I = generators[c]
        return x[np.arange(batch_size).reshape(-1,1),I]
    for i in range(melange):
        base = kernel(base)
    return base

def inversion(sigma):
    return np.unique(sigma,return_index=True)[1]
