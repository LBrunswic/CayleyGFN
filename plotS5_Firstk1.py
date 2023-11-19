import os

import numpy as np

from utils import extract
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from datetime import datetime
import sys
import os
#sns.set_theme(style="darkgrid")


timestamp = datetime.now()

DATA_FOLDER = 'RESULTS'
print(os.listdir(DATA_FOLDER))
# DATA_FOLDER = 'data'
OUTFOLDER = 'images/new'
# OUTFOLDER = 'images/best'

# FOLDER = 'data'
os.makedirs(OUTFOLDER,exist_ok=True)

#   ___Definition des Graphes_________
#   A. Problems
#       GRAPH : S15
#       GENERATORS : trans_cycle_a < cycles_a < transposition
#       REWARDS = S15_TwistedManhattan_W1 , S15_TwistedManhattan_W2exp
#   B. Embeddings
#       light :  naturalrescaled + cos + sin
#       linear : permutation matrix




def load_data(FOLDER):
    FILES = [os.path.join(FOLDER, x) for x in os.listdir(FOLDER)]
    data = []
    for filename in FILES:
        data.append(pandas.read_csv(filename))
    data = pandas.concat(data)
    data = data[[x for x in data.columns if 'Unnamed' not in x]]
    return data
print('LOADING DATA...',end='')
data = load_data(DATA_FOLDER)
print('DONE!')

def data_get(filter, data):
    data_here = pandas.DataFrame(data)
    for column in filter:
        data_here = data_here[data_here[column] == filter[column]]
    return data_here

def data_get_best(filter, data,x='reg_fn_alpha',y='EMaxSeenRew'):
    pass
    data_here = pandas.DataFrame(data)
    for column in filter:
        data_here = data_here[data_here[column] == filter[column]]
    data_here = data_here.sort_values(by=[x,y])
    data_here = data_here.drop_duplicates(subset=[x], keep='last')
    return data_here


def concat_filters(filters):
    base_filter = {}
    for filter in filters:
        base_filter.update(filter)
    return base_filter
def make_graph(
    x,y,title,name,
    data_here = None,
    comparison = None,
    preTTT = lambda x:x,
    cmap = ['red','blue','green','orange','purple','gold','darkgreen','cyan'],
    y_range = (0,1,0.05,1),
    log_scale=False,
):
    # data_here = data_get(filter,data)
    # print(filter)
    # print(data_here)
    data_here=pandas.DataFrame(data_here)
    data_here.insert(len(data_here.columns), title, [''] * len(data_here))
    if isinstance(comparison,list) and all([isinstance(x,str) for x in comparison]):
        for column in comparison:
            data_here[title] = data_here[title] + data_here[column].astype('str') + ' '
    elif isinstance(comparison,str):
        data_here[title] = data_here[comparison]
    else:
        raise NotImplementedError('comparison should be a string or list of string')

    data_here = preTTT(data_here)
    plt.clf()
    #sns.set_theme(style="darkgrid")
    if log_scale:
        extra = np.log10
    else:
        extra = lambda x:x

    data_here[y] = np.clip(extra(data_here[y].values)/y_range[3],y_range[0],y_range[1])

    f, axs = plt.subplots(nrows=1, ncols=1, figsize=(32, 18))
    print(data_here)
    a = sns.lineplot(
        data=data_here,
        x=x,
        y=y,
        hue=title,
        # legend=True,
        ax=axs,
        # palette=cmap,

    )
    plt.yticks(np.arange(y_range[0], y_range[1], step=y_range[2]))
    # sns.move_legend(
    #     axs, "lower center",
    #     bbox_to_anchor=(0.5, -.13), ncol=3, framealpha=1., fontsize='xx-large'
    # )
    a.get_figure().savefig(os.path.join(OUTFOLDER,'%s.png' % name))
    plt.clf()
    f.clf()


base_filter = {
    'optimizer':'Adam',
    'grad_batch_size': 1,
    # 'batch_size': 1024,
    # 'length_cutoff':30,
    'initial_flow':1e-3,
    'loss_base':'Apower',
    'loss_alpha':2,
    'path_redraw':0,
    'neighborhood':0,
}
NoBETA = {
    'B_beta':-1000.
}
NoALPHA = {
    'reg_fn_alpha':-1000.
}
graph_filters = [#ordered by difficulty
    {
        'graph_size':15,
        'graph_generators':'trans_cycle_a',
        'inverse':True,
        'initial_pos':'SymmetricUniform'
    },
    {
        'graph_size': 15,
        'graph_generators': 'cycles_a',
        'inverse': True,
        'initial_pos': 'SymmetricUniform'
    },
    {
        'graph_size':15,
        'graph_generators':'transpositions',
        'inverse':True,
        'initial_pos':'SymmetricUniform'
    },
    {
        'graph_size':5,
        'graph_generators':'trans_cycle_a',
        'inverse':True,
        'initial_pos':'SymmetricUniform'
    },
    {
        'graph_size': 5,
        'graph_generators': 'cycles_a',
        'inverse': True,
        'initial_pos': 'SymmetricUniform'
    },
    {
        'graph_size':5,
        'graph_generators':'transpositions',
        'inverse':True,
        'initial_pos':'SymmetricUniform'
    },
    {
        'graph_size':10,
        'graph_generators':'trans_cycle_a',
        'inverse':True,
        'initial_pos':'SymmetricUniform'
    },
    {
        'graph_size': 10,
        'graph_generators': 'cycles_a',
        'inverse': True,
        'initial_pos': 'SymmetricUniform'
    },
    {
        'graph_size':10,
        'graph_generators':'transpositions',
        'inverse':True,
        'initial_pos':'SymmetricUniform'
    },
]
reward_filters = [
    {
        'rew_fn': 'TwistedManhattan',
        # 'rew_param': str({'width': 1, 'scale': -100, 'exp': False, 'mini': 0}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 2, 'scale': -100, 'exp': True, 'mini': 0}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 1, 'scale': -100, 'exp': True, 'mini': 0}),
        'rew_factor': 1.0
    },
]


REG_PROJ = [
    {'reg_proj' : 'AddReg'},
    {'reg_proj' : 'OrthReg'},
]
REWARD_RESCALE = [
    {'reward_rescale' : 'Trivial'},
    {'reward_rescale' : 'ERew'}
]
normalization_filters = [i for i in range(8)]
normalization_nu_filters = [0,2]

train_filter = {
    'big_short' : {
        'epochs' : 20,
        'step_per_epoch' : 5,
    },
    'short': {
        # 'epochs': 10,
        'step_per_epoch': 5,
        'epoch':100
    },
    'long': {
        # 'epochs': 10,
        'step_per_epoch': 5,
        'epoch': 100,
        'learning_rate':2*1e-3
    }
}


def groupTTT(epsilon=0.5):
    def aux(data_here):
        data_copy = pandas.DataFrame(data_here)
        data_copy['reg_fn_alpha'] = np.floor(data_copy['reg_fn_alpha'].values/epsilon)*epsilon
        return data_copy
    return aux

def groupTTTlog(epsilon=0.5):
    def aux(data_here):
        data_copy = pandas.DataFrame(data_here)
        data_copy['reg_fn_alpha'] =np.floor(data_copy['reg_fn_alpha'].values/epsilon)*epsilon
        return data_copy
    return aux

# groupTTT = lambda epsilon: (lambda x:x)

for omega in normalization_filters:
    for nu in normalization_nu_filters:
        for graph_filter in graph_filters:
            print('MAKE GRAPH 1...',end='')
            FILTER = concat_filters([{'normalization_fn':omega},{'normalization_nu_fn':nu},base_filter,graph_filter,NoBETA,REWARD_RESCALE[0],reward_filters[0],train_filter['long']])
            data1 = data_get(FILTER,data)
            if len(data1)==0:
                continue
            make_graph(
                'reg_fn_alpha',
                'EMaxSeenRew',
                r'\gamma',
                'S%sG%sW1F1_EMaxSeenRew_%s_%s' % (graph_filter['graph_size'],graph_filter['graph_generators'],omega,nu),
                data_here=data1,
                # y_range = (0,1,0.05,np.exp(1)),
                comparison=['normalization_fn','normalization_nu_fn','reg_fn_gen','reg_proj'],
                preTTT=groupTTT(epsilon=0.5)
            )
            # make_graph(
            #     'reg_fn_alpha',
            #     'FlowSize',
            #     r'\gamma',
            #     'S%sG%sW1F1_FlowSize_%s_%s' % (
            #     graph_filter['graph_size'], graph_filter['graph_generators'], omega, nu),
            #     data_here=data1,
            #     y_range = (-10,10,2,1.),
            #     comparison=['normalization_fn', 'normalization_nu_fn', 'reg_fn_gen', 'reg_proj'],
            #     log_scale=True,
            # )
            # make_graph(
            #     'reg_fn_alpha',
            #     'ExpectedLength',
            #     r'\gamma',
            #     'S%sG%sW1F1_ExpectedLength_%s_%s' % (
            #     graph_filter['graph_size'], graph_filter['graph_generators'], omega, nu),
            #     data_here=data1,
            #     y_range = (0,30,2,1.),
            #     comparison=['normalization_fn', 'normalization_nu_fn', 'reg_fn_gen', 'reg_proj'],
            #     # p
            # )

            print('DONE!')


for omega in normalization_filters:
    for nu in normalization_nu_filters:
        for graph_filter in graph_filters:
            print('MAKE GRAPH...',omega,nu,graph_filter, end='')
            FILTER = concat_filters([{'normalization_fn':omega},{'normalization_nu_fn':nu},base_filter,graph_filter,NoBETA,REWARD_RESCALE[0],reward_filters[0],train_filter['long']])
            data1 = data_get_best(FILTER,data, x='reg_fn_alpha',y='EMaxSeenRew')
            if len(data1)==0:
                print('None')
                continue
            make_graph(
                'reg_fn_alpha',
                'EMaxSeenRew',
                r'\gamma',
                'S%sG%sW1F1_BestEMaxSeenRew_%s_%s' % (graph_filter['graph_size'],graph_filter['graph_generators'],omega,nu),
                data_here=data1,
                # y_range = (0,1,0.05,np.exp(1)),
                comparison=['normalization_fn','normalization_nu_fn','reg_fn_gen','reg_proj'],
                # preTTT=groupTTT(epsilon=0.5)
            )
            # make_graph(
            #     'reg_fn_alpha',
            #     'FlowSize',
            #     r'\gamma',
            #     'S%sG%sW1F1_BestFlowSize_%s_%s' % (
            #     graph_filter['graph_size'], graph_filter['graph_generators'], omega, nu),
            #     data_here=data1,
            #     y_range = (-10,10,2,1.),
            #     comparison=['normalization_fn', 'normalization_nu_fn', 'reg_fn_gen', 'reg_proj'],
            #     log_scale=True,
            # )
            # make_graph(
            #     'reg_fn_alpha',
            #     'ExpectedLength',
            #     r'\gamma',
            #     'S%sG%sW1F1_BestExpectedLength_%s_%s' % (
            #     graph_filter['graph_size'], graph_filter['graph_generators'], omega, nu),
            #     data_here=data1,
            #     y_range = (0,30,2,1.),
            #     comparison=['normalization_fn', 'normalization_nu_fn', 'reg_fn_gen', 'reg_proj'],
            #     # p
            # )

            print('DONE!')
raise
