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
