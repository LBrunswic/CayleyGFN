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

FOLDER = 'data'
os.makedirs('images',exist_ok=True)

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
data = load_data(FOLDER)
print('DONE!')

def data_get(filter, data):
    data_here = pandas.DataFrame(data)
    for column in filter:
        data_here = data_here[data_here[column] == filter[column]]
    return data_here

def concat_filters(filters):
    base_filter = {}
    for filter in filters:
        base_filter.update(filter)
    return base_filter
def make_graph(
    x,y,title,name,
    filter = None,
    comparison = None,
    preTTT = lambda x:x,
    cmap = ['red','blue','green','orange','purple','gold','darkgreen','cyan'],
    y_range = (0,1,0.05,1),
    log_scale=False,
):
    data_here = data_get(filter,data)
    print(filter)
    # print(data_here)
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
    a.get_figure().savefig('images/%s.png' % name)
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

FlowEstimator_filter = [
    {
        'flowestimator_opt' : str({
            'options': {
                'kernel_depth' : 1,
                'width' : 32,
                'final_activation' : 'linear',
            },
            'kernel_options': {

                'kernel_initializer' : 'Orthogonal',
                'activation': 'tanh',
                'implementation': 1,
            }
        }),

    },

    {
        'flowestimator_opt':   str({
            'options': {
                'kernel_depth': 2,
                'width': 64,
                'final_activation': 'linear',
            },
            'kernel_options': {
                'kernel_initializer': 'Orthogonal',
                'activation': 'tanh',
                'implementation': 1,
            }
        })
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
normalization_filters = [i for i in range(6)
]
normalization_nu_filters = [i for i in range(3)
]

train_filter = {
    'big_short' : {
        'epochs' : 20,
        'step_per_epoch' : 5,
    },
    'short': {
        # 'epochs': 10,
        'step_per_epoch': 5,
        'epoch':20
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
            FILTER = concat_filters([{'normalization_fn':omega},{'normalization_nu_fn':nu},base_filter,graph_filter,NoBETA,REWARD_RESCALE[0],reward_filters[0],train_filter['short']])
            data1 = data_get(FILTER,data)
            if len(data1)==0:
                continue
            make_graph(
                'reg_fn_alpha',
                'EMaxSeenRew',
                r'\gamma',
                'S%sG%sW1F1_EMaxSeenRew_%s_%s' % (graph_filter['graph_size'],graph_filter['graph_generators'],omega,nu),
                filter=FILTER,
                # y_range = (0,1,0.05,np.exp(1)),
                comparison=['normalization_fn','normalization_nu_fn','reg_fn_gen','reg_proj'],
                # preTTT=groupTTT(epsilon=0.5)
            )
            make_graph(
                'reg_fn_alpha',
                'FlowSize',
                r'\gamma',
                'S%sG%sW1F1_FlowSize_%s_%s' % (
                graph_filter['graph_size'], graph_filter['graph_generators'], omega, nu),
                filter=FILTER,
                y_range = (-10,10,2,1.),
                comparison=['normalization_fn', 'normalization_nu_fn', 'reg_fn_gen', 'reg_proj'],
                log_scale=True,
            )
            make_graph(
                'reg_fn_alpha',
                'ExpectedLength',
                r'\gamma',
                'S%sG%sW1F1_ExpectedLength_%s_%s' % (
                graph_filter['graph_size'], graph_filter['graph_generators'], omega, nu),
                filter=FILTER,
                y_range = (0,30,2,1.),
                comparison=['normalization_fn', 'normalization_nu_fn', 'reg_fn_gen', 'reg_proj'],
                # p
            )

            print('DONE!')
raise

def pack_reg_fn_alpha(data,step=0.5,column='reg_fn_alpha'):
    res = pandas.DataFrame(data)
    res[column] = res[column].div(step).floordiv(1).mul(step)
    return res

# raise
hue_display_name1='normalizations'
data.insert(len(data.columns),hue_display_name1,['']*len(data))
link = np.array(['_']*len(data))
columns = ['rew_factor','normalization_fn','normalization_nu_fn']
for column in  columns:
    data[hue_display_name1] = data[hue_display_name1] + data[column].astype('str') + link

hue_display_name2='reward_fact-rescale'
data.insert(len(data.columns),hue_display_name2,['']*len(data))
link = np.array(['_']*len(data))
columns = ['rew_factor','reward_rescale']
for column in  columns:
    data[hue_display_name2] = data[hue_display_name2] + data[column].astype('str') + link


data['EMaxSeenRew'] = data['EMaxSeenRew'].values / data['rew_factor'].values
data['ExpectedReward'] = data['ExpectedReward'].values / data['rew_factor'].values

dataS15 = data[data['graph_size']==15]
dataS15A = dataS15[dataS15['optimizer']=='Adam']
dataS15A_noBeta = dataS15A[dataS15A['B_beta']<-900.]
dataS15A_YesBeta = dataS15A[dataS15A['B_beta']>-900.]

# dataS15A_normreg = dataS15A[dataS15A['reg_fn_gen']==norm2]
# dataS15N = dataS15[dataS15['optimizer']=='Nesterov']
dataS15A_norm = dataS15A_noBeta[dataS15A_noBeta['normalization_fn']==4]
dataS15A_norm = dataS15A_norm[dataS15A_norm['normalization_nu_fn']==2]

#___________________________________Norm Reg HP_fit__________________________________

hue_display_name3='Norm regularizations'
datacurve1 = dataS15A_YesBeta[dataS15A_YesBeta['reg_fn_alpha']<-900.]
datacurve2 = dataS15A_noBeta[dataS15A_noBeta['reg_fn_gen']=='norm2']
datacurve2 = datacurve2[datacurve2['reg_proj']=='AddReg']
datacurve1.insert(len(datacurve1.columns),hue_display_name3,['beta=omega,gamma=0']*len(datacurve1))
datacurve1.insert(len(datacurve1.columns),'omega',datacurve1['B_beta'])
datacurve2.insert(len(datacurve2.columns),hue_display_name3,['beta=0,gamma=omega']*len(datacurve2))
datacurve2.insert(len(datacurve2.columns),'omega',datacurve2['reg_fn_alpha'])

here_data = pandas.concat([
    datacurve1,
    datacurve2
])
here_data1 = pack_reg_fn_alpha(here_data,step=0.2, column='omega')

f,axs = plt.subplots(nrows=1,ncols=2,figsize=(32,18),sharey=True)
cmap = ['red','blue']# sns.color_palette("Paired")
a = sns.lineplot(
    data = here_data1,
    x='omega',
    y='ExpectedReward',
    hue=hue_display_name3,
    legend=True,
    ax=axs[0],
    palette=cmap
)
plt.yticks(np.arange(0, 1, step=0.05))

here_data = here_data[here_data['omega']<-2]
here_data = here_data[here_data['omega']>-4.5]
here_data = pack_reg_fn_alpha(here_data,step=0.05, column='omega')

a = sns.lineplot(
    data = here_data,
    x='omega',
    y='ExpectedReward',
    hue=hue_display_name3,
    legend=True,
    ax=axs[1],
    palette=cmap
)
sns.move_legend(
    axs[0], "lower center",
    bbox_to_anchor=(0.5, -.13), ncol=3,  framealpha=1.,fontsize='xx-large'
)
plt.yticks(np.arange(0, 1, step=0.05))
a.get_figure().savefig('images/S15_Beta.png')
plt.clf()

#___________________________________Norm Reg HP_fit + orthreg__________________________________

hue_display_name3='Norm regularizations'
here_data = dataS15A_noBeta[dataS15A_noBeta['reg_fn_gen']=='norm2']

# here_data.insert(len(here_data.columns),hue_display_name3,['beta=omega,gamma=0']*len(datacurve1))



here_data1 = pack_reg_fn_alpha(here_data,step=0.2, column='reg_fn_alpha')

f,axs = plt.subplots(nrows=1,ncols=2,figsize=(32,18),sharey=True)
cmap = ['blue','darkgreen']# sns.color_palette("Paired")
a = sns.lineplot(
    data = here_data1,
    x='reg_fn_alpha',
    y='ExpectedReward',
    hue='reg_proj',
    legend=True,
    ax=axs[0],
    palette=cmap
)
plt.yticks(np.arange(0, 1, step=0.05))

here_data = here_data[here_data['reg_fn_alpha']<-2]
here_data = here_data[here_data['reg_fn_alpha']>-4.5]
here_data = pack_reg_fn_alpha(here_data,step=0.05, column='reg_fn_alpha')

a = sns.lineplot(
    data = here_data,
    x='reg_fn_alpha',
    y='ExpectedReward',
    hue='reg_proj',
    legend=True,
    ax=axs[1],
    palette=cmap
)
sns.move_legend(
    axs[0], "lower center",
    bbox_to_anchor=(0.5, -.13), ncol=3,  framealpha=1.,fontsize='xx-large'
)
plt.yticks(np.arange(0, 1, step=0.05))
a.get_figure().savefig('images/S15_norm_orth.png')
plt.clf()

#___________________________________Normalizations comparisons__________________________________

here_data = dataS15A_noBeta[dataS15A_noBeta['reg_proj']=='AddReg']
here_data = here_data[here_data['reward_rescale']=='Trivial']
here_data = pack_reg_fn_alpha(here_data)
f,axs = plt.subplots(nrows=2,ncols=1,figsize=(32,18))
cmap = sns.color_palette("Paired")
a = sns.lineplot(
    data = here_data,
    x='reg_fn_alpha',
    y='EMaxSeenRew',
    hue=hue_display_name1,
    legend=False,
    ax=axs[0],
    palette=cmap
)
plt.yticks(np.arange(0, 1, step=0.05))

a = sns.lineplot(
    data = here_data,
    x='reg_fn_alpha',
    y='ExpectedReward',
    hue=hue_display_name1,
    legend=True,
    ax=axs[1],
    palette=cmap
)
sns.move_legend(
    axs[1], "lower center",
    bbox_to_anchor=(0.5, -.3), ncol=3,  framealpha=1.,fontsize='xx-large'
)
plt.yticks(np.arange(0, 1, step=0.05))
a.get_figure().savefig('images/S15normalizations.png')
plt.clf()



#_________________________________Reward Rescale_______________________________________


dataS15A_norm = pack_reg_fn_alpha(dataS15A_norm)
f,axs = plt.subplots(nrows=2,ncols=1,figsize=(32,18))
cmap = sns.color_palette("Paired")
a = sns.lineplot(
    data = dataS15A_norm[dataS15A_norm['reg_proj']=='AddReg'],
    x='reg_fn_alpha',
    y='EMaxSeenRew',
    hue=hue_display_name2,
    legend=False,
    ax=axs[0],
    palette=cmap
)
plt.yticks(np.arange(0, 1, step=0.05))

a = sns.lineplot(
    data = dataS15A_norm[dataS15A_norm['reg_proj']=='AddReg'],
    x='reg_fn_alpha',
    y='ExpectedReward',
    hue=hue_display_name2,
    legend=True,
    ax=axs[1],
    palette=cmap
)
sns.move_legend(
    axs[1], "lower center",
    bbox_to_anchor=(0.5, -.3), ncol=3,  framealpha=1.,fontsize='xx-large'
)
plt.yticks(np.arange(0, 1, step=0.05))
a.get_figure().savefig('images/S15reward_rescale.png')

