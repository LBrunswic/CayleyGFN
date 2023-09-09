import numpy as np
import matplotlib.pyplot as plt

SIZE = 54

def R(x):
    return SIZE*(x[:,0]==1)+1e-2


def foo(x):
    return np.linalg.norm(x,axis=0)<1



actions = np.array([
    [1,0]+list(range(2,SIZE)),
    list(range(1,SIZE))+[0],
    [SIZE-1]+list(range(SIZE-1))
],dtype=int)

def MH(MAX_STEP = 1000,actions=actions,foo=R,batch_size = 10):
    START = random_perm(batch_size,54)[0]
    base = np.arange(len(actions))
    def kernel(x):
        c = np.random.choice(base,size=batch_size)
        return x[np.arange(SIZE).reshape(-1,1),actions[c]]
    MarkovChain = np.zeros((MAX_STEP,batch_size,SIZE),int)
    MarkovChain[0]=START
    for i in range(MAX_STEP-1):
        proposal = kernel(MarkovChain[i])
        if np.all(np.random.uniform()<foo(proposal)/foo(MarkovChain[i])):
            MarkovChain[i+1]=proposal
        else:
            MarkovChain[i+1]= MarkovChain[i]
    return MarkovChain
