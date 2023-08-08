import numpy as np
import os
import matplotlib.pyplot as plt
import math
from itertools import permutations
from decimal import Decimal
from Graphs.CayleyGraph import Symmetric
import graphviz
from Groups.symmetric_groups import *
plt.rcParams['text.usetex'] = True
import matplotlib as mpl
def _aux_plot(y1,y2,label1,label2,path):
    assert(y1.shape[1]==y2.shape[1])
    assert(len(label1)==(y1.shape[0]))
    assert(len(label2)==(y2.shape[0]))
    Nepoch = y1.shape[1]
    n_graph1 = y1.shape[0]
    n_graph2 = y2.shape[0]
    x = np.arange(Nepoch)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epochs')
    color = 'tab:red'
    ax1.set_ylabel(r'$\log$', position=(0, 0.35), rotation=90, color=color, alpha=1)
    ax1.tick_params(axis='y', rotation=45, labelcolor=color)
    leftcolormap = mpl.cm.get_cmap('autumn')
    def c1(x,m=0.2,M=0.7):
        return leftcolormap(m+x/n_graph1*(M-m))
    for i in range(n_graph1):
        ax1.plot(x, y1[i], color=c1(i), label=label1[i])
    ax2 = ax1.twinx()
    color = 'blue'
    ax2.set_ylabel(r'$\log$', position=(0.1, 0.35), rotation=0,
                   color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    rightcolormap = mpl.cm.get_cmap('winter')
    def c2(x, m=0.2, M=0.7):
        return rightcolormap(m + x / n_graph1 * (M - m))
    for i in range(n_graph2):
        ax2.plot(x, y2[i], color=c2(i), label=label2[i])
    box = ax1.get_position()

    ax1.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05),
               fancybox=True, shadow=True, ncol=1)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05),
               fancybox=True, shadow=True, ncol=1)

    fig.tight_layout()
    plt.savefig(path)
    plt.clf()

def plot(init_target = 1 ,folder='tests/',file_name='Errors.npy',image_name='Errors.png'):
    with open(os.path.join(folder,file_name),'rb') as f:
        A = np.load(f)
        print(A.shape)
        ERhat,ERbar,loss,init,totR,MaxR = A
    print(ERhat)
    print(init)
    A[0] =  np.log(np.clip(A[0],1e-8,5))
    A[3] = np.log( np.abs(A[3]-init_target)/(1e-8+init_target))
    A[2] = np.log(np.clip(A[2],1e-8,5))
    _aux_plot(
        y1=A[2:4],
        y2=A[:1],
        label1=['loss','init'],
        label2=[r'$\mathcal E_{R}$'],
        path=os.path.join(folder,'LIRhat.png')
    )
    return ERhat,ERbar,loss,init,totR,MaxR

def represent_symmetric_R_first_one(size,generators,flow):

    dot = graphviz.Digraph(comment='S%s' % size, strict=True, format='png',engine='fdp')
    N = math.factorial(size)
    if size<10:
        def aux(x):
            return "".join([str(a) for a in x])
    else:
        def aux(x):
            return str(tuple(x))

    # FIRST CIRCLE

    if N>10**3:
        raise OverflowError('Number of states is too large to be represented')
    Circles = [[]]
    Circles[0] = np.array([[0] + list(x) for x in permutations(list(range(1, size)), size - 1)])
    reached = {tuple(x) for x in Circles[0]}
    i = 0
    while len(reached) < N:
        new = np.einsum('aij,sj->asi', generators, Circles[i]).reshape(-1, size)
        Circles.append([])
        for x in new:
            if tuple(x) not in reached:
                reached.add(tuple(x))
                Circles[-1].append(x)
        Circles[-1] = np.array(Circles[-1])
        i += 1
    # print([x.shape for x in Circles])
    n_circle = len(Circles)

    Radii = size ** 0.8 * np.arange(1 + 1 * n_circle, 1, -1) * 0.83
    theta = [2 * math.pi / len(c) for c in Circles]

    for i in range(len(Circles)):
        circle = Circles[i]
        for j in range(circle.shape[0]):
            pos = Radii[i] * np.array([math.cos(theta[i]*j+theta[i]*i/3),math.sin(theta[i]*j+theta[i]*i/2)])
            dot.node(
                aux(circle[j]),
                # pos = "%s,%s" % tuple(pos)
            )
    for circle in Circles:
        Fout = flow(circle).numpy()
        i=0
        for x in circle:
            j=0
            for g in generators:
                # f = '%.2E' % Decimal(float(Fout[i,j]))
                # f = 'f=%s' % float(Fout[i,j])
                # f = 'texlbl= "$%s$"' % i
                # print(f)
                a = aux(x)
                b = aux(g@x)
                dot.edge(a, b,color="red:blue")
                j+=1
            i+=1
    return dot

def represent_symmetric_R_first_one_at(size,model,initial='random',iteration=4,engine='fdp'):

    dot = graphviz.Digraph(comment='S%s' % size, strict=True, format='png',engine=engine)
    N = math.factorial(size)
    if size<10:
        def aux(x):
            return "".join([str(a) for a in x])
    else:
        def aux(x):
            return str(tuple(x))
    color = ['red', 'blue', 'green', 'yellow', 'aqua', 'aquamarine1', 'chocolate4', 'chocolate1', 'dodgerblue4',
             'purple2', 'purple4']
    if initial == 'close':
        if size<7:
            raise NotImplementedError
        p = 4
        initial = np.concatenate([random_perm(11, p),np.full((11,size-p),np.arange(p,size))],axis=1)
    elif initial == 'close_gen':
        stalk = np.array(list(range(1,size))+[0])
        print(stalk)
        initial = random_perm_from_gen(11,model.graph.reverse_actions,iteration=iteration,stalk=stalk)
    paths = model.gen_true_path(initial=initial)[:len(color)]
    # FIRST CIRCLE



    edges={}
    node={}
    node_all = set()
    for j in range(len(paths)):
        if isinstance(paths[j],np.ndarray):
            path = paths[j].astype('int')
        else:
            path = paths[j].numpy().astype('int')
        if aux(path[0]) not in node:
            node[aux(path[0])] = []
        node[aux(path[0])].append(color[j])
        ell = path.shape[0]
        for i in range(1,ell):
            a = aux(path[i-1])
            b = aux(path[i])
            node_all.add(a)
            node_all.add(b)
            if (a,b) not in edges:
                edges[(a, b)] = []
            edges[(a, b)].append(color[j])
    node_all = list(node_all)
    node_cor={}
    for i in range(len(node_all)):
        node_cor[node_all[i]] = str(i)
    for a,b in edges:
        dot.node(node_cor[a])
        dot.node(node_cor[b])
        dot.edge(node_cor[a], node_cor[b], color=':'.join(edges[(a,b)]))

    for a in node_all:
        if eval(a)[0] == 1 and a in node:
            dot.node(node_cor[a], style='filled', fillcolor=':'.join(node[a]), shape='doublecircle')
        elif eval(a)[0] == 1:
            dot.node(node_cor[a], shape='doublecircle')
        elif a in node:
            dot.node(node_cor[a], style='filled',fillcolor=':'.join(node[a]))

    return dot,node_all,node_cor,edges


