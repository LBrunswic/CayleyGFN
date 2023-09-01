import numpy as np
import itertools
import tensorflow as tf
from Groups.symmetric_groups import *
from scipy.linalg import block_diag
import os


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
        self.embedding = embedding

    def iter(self,depth):
        N = self.group_dim
        base = np.arange(N,dtype='int32').reshape(1,-1)
        for i in range(depth):
            old = [tuple(x) for x in base]
            old_set = set(old)
            new = list(set([tuple(x) for x in base[...,self.generators].reshape(-1,N) if tuple(x) not in old_set]))
            base = np.array(old+new)
        return tf.constant(np.array([self.representation(g) for g in  base],dtype='int32'))

    def sample(self, batch_size):
        return self.random_gen.sample(batch_size)

    def density(self, position_batch):
        return self.random_gen.density(position_batch)

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

rubick_gen = tf.constant(rubick_generators(48)[0])

pi = np.math.pi
def hot(n,*args,omega=1,**kwarg):
    Id = tf.eye(n)
    def aux(paths):
        # print((*paths.shape[:-1], n*n,))
        # print(tf.gather(Id,paths))
        x = tf.gather(Id,paths)
        k = len(x.shape)
        # print((*x.shape[:-2],k-2,k-1))
        transpose_shape = (*tuple(range(k-2)),k-2,k-1)
        return tf.cast(
            tf.reshape(
                tf.transpose(x,transpose_shape),
                (*paths.shape[:-1], n*n,)
            ),
        'float32')
    return aux,n*n
pi = np.math.pi
def hotalpha(n,*args,omega=1,**kwarg):
    alpha = tf.reshape(tf.math.exp(-10*np.arange(n,dtype='float32')),(n,1))
    Id = tf.eye(n)*alpha
    def aux(paths):
        return tf.cast(tf.reshape(tf.gather(Id,paths),(*paths.shape[:-1], n*n,) ),'float32')
    return aux,n*n
Embeddings = {
    'natural': lambda n,*args,**kwarg: (lambda x : x,n),
    'cos': lambda n,*args,omega=1,**kwarg: (lambda x :tf.math.cos(2*omega*pi*tf.cast(x,'float32')/n),n),
    'sin': lambda n,*args,omega=1,**kwarg: (lambda x :tf.math.sin(2*omega*pi*tf.cast(x,'float32')/n),n),
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
    dtype='float32'
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
    ]),'int32')
    inverse_actions = tf.cast(tf.stack([
        block_diag(*[rho(inversion(gen)) for rho,_ in R_list])
        for gen in generators
    ]),'int32')


    E_list = [Embeddings[emb_choice](n,**emb_kwarg) for emb_choice,emb_kwarg in embedding]
    embedding_dim = sum([d for _,d in E_list])
    @tf.function
    def iota(x):
         return tf.concat([emb(x) for emb,_  in E_list],axis=-1)

    return CayleyGraphLinearEmb(direct_actions,inverse_actions,diameter,generators=generators,rep_list = R_list,random_gen=random_gen,embedding=iota,embedding_dim=embedding_dim, group_dim=n, name='Sym%s_%s' % (n,generators.numpy()))

from pyvis.network import Network
import networkx as nx
from networkx.readwrite import json_graph


def graph_representation(n,generators_choice='transpositions',filename='graph',inverse=False,reward=lambda x:x[0]==0):
    colors = ['blue','red','green','purple','black','cyan','orange','orangered','yellowgreen']
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
    with open('%s.html' % filename,'w') as f:
        f.write(template % dot.replace('\n',''))
    return net


from itertools import permutations
import matplotlib as mpl
def path_representation(paths,filename='graph',folder = 'results', previous=None,next=None ,reward=lambda x:x[0]==0,flow=None):
    paths = paths.numpy()
    cmap = mpl.colormaps['turbo']
    colors = ['gold','red','green','cyan','purple','blue','orange','orangered','yellowgreen','darkgreen','pink','lightcyan','darkred','hotpink','peachpuff','cornflowerblue']
    batch_size, length,n= paths.shape
    net = nx.MultiDiGraph()
    if batch_size>len(colors):
        NotImplementedError('too many paths to draw')
    if flow is None:
        flow = np.ones_like(paths[...,0],dtype=int)*10
    def name(x):
        R = str(int(reward(x.reshape(1,-1)).numpy().reshape(-1)))
        state = "."+'.'.join([str(u) for u in x])
        return state
        return R+state
        # print()
        # print('_______________________')
        # print(x.shape)
        # print(reward(x))
        # res = str(int(reward(x.reshape(1,-1))))
        # print(res)
        # return res

    # for x in permutations(np.arange(n,dtype=int)):
    #     net.add_node(name(x), size=20, color=colors[reward(x)])
    for i,path  in enumerate(paths):
        for t in range(length-1):
            x = path[t]
            y = path[t+1]
            if t == 0:
                net.add_node(
                    name(x), size=20,
                    # color=colors[reward(x)],
                    penwidth =4)
            else:
                net.add_node(name(x), size=20,
                # color=colors[reward(x)]
                )
            net.add_node(name(y), size=20,
             # color=colors[reward(y)]
             )
            # print(flow[i,t])
            net.add_edge(name(x),name(y),color=colors[i], penwidth = flow[i,t])
    for i,path  in enumerate(paths):
        x = path[0]
        net.add_node(name(x), size=20,color=colors[i],penwidth=10)

    nx.nx_pydot.write_dot(net,'plop.dot')
    with open('plop.dot','r') as f:
        dot = ' '.join(f.readlines())
    links=''
    if previous is not None:
        links += '<a href="%s.html">previous</a>' % previous
    if next is not None:
        links += '<a href="%s.html">next</a>' % next

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

        %s

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
    with open('%s.html' % os.path.join(folder,filename),'w') as f:
        f.write(template % (links,dot.replace('\n','')))
    return net
