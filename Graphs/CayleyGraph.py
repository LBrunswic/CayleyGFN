import numpy as np
import itertools
import tensorflow as tf
from Groups.symmetric_groups import *
from scipy.linalg import block_diag

class CayleyGraphLinearEmb():
    def __init__(self,
                 direct_actions,
                 diameter,
                 random_gen=None,
                 name='cayley_graph',
                 dtype='float32',
            ):

        self.actions = tf.constant(direct_actions,dtype=dtype)
        self.embedding_dim = tf.constant(direct_actions.shape[-1]) # always a (p,)
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


SymmetricGenerators = {
    'trans_cycle_a' : lambda n: ([[1,0]+list(range(2,n)), list(range(1,n))+[0]],[True,False],n),
    'transpositions' : lambda n: ([list(range(i))+[i+1,i]+list(range(i+2,n)) for i in range(n-1)], [True]*(n-1),n),
    'all' : lambda n: (list(itertools.permutations(list(range(n)))), [True]*np.math.factorial(n),1),
    'Rubicks': rubick_generators
}
Representations = {
    # name -> size ->  (morphism, dim)
    'natural' : lambda n : (permutation_matrix,n),
}
RandomGenerators = {
    # name -> batch_size -> size -> (batch_size,size)
    'uniform': random_perm,
    'uniform_extremal': random_perm_extremal,
    'deterministic': lambda b,n : np.full((b,n),np.arange(n)),
    'rubick': lambda b,n : rubick_random(b)
}

pi = np.math.pi
Embeddings = {
    'natural': lambda *args,**kwarg: lambda x : x,
    'cos':  lambda n,*args,omega=1,**kwarg: lambda x :tf.math.cos(2*omega*pi*x/n),
    'sin':  lambda n,*args,omega=1,**kwarg: lambda x :tf.math.sin(2*omega*pi*x/n),
}

def Symmetric(
    n,
    generators='trans_cycle_a',
    representation = [('natural',{})],
    inverse = False,
    random_gen = 'uniform',
    embedding = [('natural',{})],
    dtype='float32'
):
    generators,involutions,diameter = SymmetricGenerators[generators](n)
    if inverse:
        inverse_generators = []
        for i,gen in enumerate(generators):
            if not involutions[i]:
                inverse_generators.append(inversion(gen))
        generators += inverse_generators

    assert(len(representation)==len(embedding))

    R_list = [Representations[rep](n) for rep in representation]
    dim = sum([d for rho,d in R_list])
    direct_actions = tf.cast(tf.stack([
        block_diag(*[rho(gen) for rho,_ in R_list])
        for gen in generators
    ]),dtype)

    E_list = [Embeddings[emb_choice](n,**emb_kwarg) for emb_choice,emb_kwarg in embedding]
    @tf.function
    def iota(x):
         return tf.concat([emb(tf.cast(x,dtype)) for emb  in E_list],axis=-1)
    random_gen = RandomGenerators[random_gen]
    initial = lambda b: iota(random_gen(b,n))

    return CayleyGraphLinearEmb(direct_actions,diameter,random_gen=initial,name='Sym%s_%s' % (n,generators))

from pyvis.network import Network
import networkx as nx
from networkx.readwrite import json_graph


def graph_representation(n,generators_choice='transpositions',inverse=False,reward=lambda x:x[0]==0):
    colors = ['blue','red','green','purple','black','cyan']
    generators = SymmetricGenerators[generators_choice](n)[0]
    if inverse:
        for i,inv in enumerate(SymmetricGenerators[generators_choice](n)[1]):
            if not inv:
                generators.append(inversion(generators[i]))
    generators = np.array(generators)
    def iter(base,g,N):
        old = [tuple(x) for x in base]
        new = [tuple(x) for x in base[...,g].reshape(-1,N)]
        return np.array(list(set(old+new)))

    base = np.arange(n).reshape(1,-1)
    oldshape=0
    while True:
        base = iter(base,generators,n)
        if oldshape == base.shape:
            break
        else:
            oldshape = base.shape
    net = nx.DiGraph()
    # net = Network('1000px','2000px',is_directed=True)
    def name(x):
        return ''.join([str(u) for u in x])
    for x in base:
        net.add_node(name(x), size=20, color=colors[reward(x)])
    for x in base:
        for i,y in enumerate(x[generators]):
            net.add_edge(name(x),name(y),color=colors[i])

    nx.nx_pydot.write_dot(net,'plop.dot')
    with open('plop.dot','r') as f:
        dot = ' '.join(f.readlines())
    template = """
        <!DOCTYPE html>
    <html lang="en">
      <head>
        <title> Graph </title>

        <script
          type="text/javascript"
          src="vis.js"
        ></script>

        <style type="text/css">
          #mynetwork {
            width: 2200px;
            height: 1400px;
            border: 3px solid lightgray;
          }
        </style>
      </head>
      <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
          // create an array with nodes


          // create a network
          var container = document.getElementById("mynetwork");
          var dot = "%s";
          var options = {};
          var data = vis.parseDOTNetwork(dot);
          var network = new vis.Network(container, data, options);
        </script>
      </body>
    </html>
    """
    with open('graph.html','w') as f:
        f.write(template % dot.replace('\n',''))
    return net


from itertools import permutations
import matplotlib as mpl
def path_representation(paths,reward=lambda x:x[0]==0):
    cmap = mpl.colormaps['turbo']
    colors = ['gold','blue','red','green','yellow','cyan']
    batch_size, length,n= paths.shape
    net = nx.MultiDiGraph()
    if batch_size>len(colors):
        NotImplementedError('too many paths to draw')

    def name(x):
        return ''.join([str(u) for u in x])

    for x in permutations(np.arange(n,dtype=int)):
        net.add_node(name(x), size=20, color=colors[reward(x)])
    for i,path  in enumerate(paths):
        for t in range(length-1):
            x = path[t]
            y = path[t+1]
            net.add_edge(name(x),name(y),color=colors[i])

    nx.nx_pydot.write_dot(net,'plop.dot')
    with open('plop.dot','r') as f:
        dot = ' '.join(f.readlines())
    template = """
        <!DOCTYPE html>
    <html lang="en">
      <head>
        <title> Graph </title>

        <script
          type="text/javascript"
          src="vis.js"
        ></script>

        <style type="text/css">
          #mynetwork {
            width: 2200px;
            height: 1400px;
            border: 3px solid lightgray;
          }
        </style>
      </head>
      <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
          // create an array with nodes


          // create a network
          var container = document.getElementById("mynetwork");
          var dot = "%s";
          var options = {};
          var data = vis.parseDOTNetwork(dot);
          var network = new vis.Network(container, data, options);
        </script>
      </body>
    </html>
    """
    with open('graph.html','w') as f:
        f.write(template % dot.replace('\n',''))
    return net
