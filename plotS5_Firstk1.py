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
DATA_FOLDER = '/home/maxbrain/DATA/TaskForce/Results/CayleyGFN/Hyperparameters/generate_S15G3W1'
print(os.listdir(DATA_FOLDER))
# DATA_FOLDER = 'data'
OUTFOLDER = 'images/Hyperparameters/generate_S15G3W1'
# OUTFOLDER = 'images/best'
CMAP = None
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




def load_data(FOLDER,FILES=None):
    if FILES is None:
        FILES = [os.path.join(FOLDER, x) for x in os.listdir(FOLDER) if '.csv' in x]
    else:
        FILES = [os.path.join(FOLDER, x) for x in FILES]
    data = []
    for filename in FILES:
        print(filename)
        data.append(pandas.read_csv(filename))
        print(data[-1].columns)
    data = pandas.concat(data)
    data = data[[x for x in data.columns if 'Unnamed' not in x]]

    return data
print('LOADING DATA...',end='')
data = load_data(DATA_FOLDER)
print(data)
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
    cmap = CMAP,
    y_range = (0,1,0.05,1),
    log_scale=False,
):
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
    print(data_here)
    plt.clf()
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
        palette=cmap,

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
    # 'optimizer':'Adam',
    'grad_batch_size': 1,
    # 'batch_size': 1024,
    # 'length_cutoff':30,
    # 'initial_flow':1e-3,
    'loss_base':'Apower',
    # 'loss_alpha':2,
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
        'rew_param': str({'width': 1, 'scale': -100, 'exp': False, 'mini': 0.0}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 5, 'scale': -100, 'exp': True, 'mini': 2.0}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 2, 'scale': -100, 'exp': True, 'mini': 1.0}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 2, 'scale': -100, 'exp': True, 'mini': 2.0}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 5, 'scale': -100, 'exp': True, 'mini': 0.}),
        'rew_factor': 1.0
    },
    {
        'rew_fn': 'TwistedManhattan',
        'rew_param': str({'width': 6, 'scale': -100, 'exp': True, 'mini': 0.}),
        'rew_factor': 1.0
    },
]
KERNEL_OPT = [
    {'flowestimator_opt': str({
            'options': {'kernel_depth': 2, 'width': 32, 'final_activation': 'linear'},
            'kernel_options': {'kernel_initializer': 'Orthogonal', 'activation': 'tanh', 'implementation': 1}})
    }
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
normalization_nu_filters = [0]

train_filter = {
    'precise' : {
        'epoch' : 20,
        'learning_rate': 0.001,
        'step_per_epoch': 50,
        # 'step_per_epoch' : 5,
    },
    'short': {
        # 'epochs': 10,
        # 'step_per_epoch': 5,
        'epoch':10
    },
    'long': {
        # 'epochs': 10,
        # 'step_per_epoch': 5,
        'epoch': 100,
        # 'learning_rate':2*1e-3
    }
}

SEEDS = []

def groupTTT(epsilon=0.5):
    def aux(data_here):
        data_copy = pandas.DataFrame(data_here)
        data_copy['reg_fn_alpha'] = np.floor(data_copy['reg_fn_alpha'].values/epsilon)*epsilon
        return data_copy
    return aux

def probagroupTTT(epsilon=0.5,success=0.8,value='EMaxSeenRew'):
    def aux(data_here):
        data_copy = pandas.DataFrame(data_here)
        data_copy['reg_fn_alpha'] = np.floor(data_copy['reg_fn_alpha'].values/epsilon)*epsilon
        data_copy[value] = data_copy[value]>success
        return data_copy

    return aux


def groupTTTlog(epsilon=0.5):
    def aux(data_here):
        data_copy = pandas.DataFrame(data_here)
        data_copy['reg_fn_alpha'] =np.floor(data_copy['reg_fn_alpha'].values/epsilon)*epsilon
        return data_copy
    return aux
#
groupTTT = lambda epsilon: (lambda x:x)
for reg_fn_gen in ['norm2','LogPathLen']:
    for reg_proj in ['OrthReg','AddReg']:
        for graph_filter in graph_filters:
            FILTER = concat_filters([{'reg_fn_gen':reg_fn_gen,'reg_proj':reg_proj},graph_filter,NoBETA,train_filter['precise'],reward_filters[0]])
            data1 = data_get(FILTER,data)
            if len(data1)==0:
                continue
            make_graph(
                'reg_fn_alpha',
                'EMaxSeenRew',
                r'\gamma',
                'S%sG%sW1F1_EMaxSeenRew_%s_%s' % (graph_filter['graph_size'],graph_filter['graph_generators'],reg_proj,reg_fn_gen),
                data_here=data1,
                y_range = (0,1,0.1,np.exp(0)),
                comparison=['normalization_fn','normalization_nu_fn','reg_fn_gen','reg_proj','batch_size','step_per_epoch','epoch'],
                # preTTT=probagroupTTT(epsilon=0.1)
                # preTTT=probagroupTTT(epsilon=0.1, success=2400,value='MaxSeenRew')
                        )

for normalization_fn in [0]:
    for normalization_nu_fn in [5]:
        for graph_filter in graph_filters:
            FILTER = concat_filters([{'normalization_fn':normalization_fn,'normalization_nu_fn':normalization_nu_fn},graph_filter,NoBETA,train_filter['precise'],reward_filters[0]])
            data1 = data_get(FILTER,data)
            if len(data1)==0:
                print(normalization_fn,normalization_nu_fn)
                continue
            make_graph(
                'reg_fn_alpha',
                'EMaxSeenRew',
                r'\gamma',
                'S%sG%sW1F1_EMaxSeenRew_%s_%s' % (graph_filter['graph_size'],graph_filter['graph_generators'],normalization_fn,normalization_nu_fn),
                data_here=data1,
                y_range = (0,1,0.1,np.exp(0)),
                comparison=['normalization_fn','normalization_nu_fn','reg_fn_gen','reg_proj','batch_size','step_per_epoch','epoch'],
                # preTTT=probagroupTTT(epsilon=0.1)
                # preTTT=probagroupTTT(epsilon=0.1, success=2400,value='MaxSeenRew')
                        )
