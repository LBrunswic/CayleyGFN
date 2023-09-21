import os

import numpy as np

from utils import extract
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
from datetime import datetime
import sys
import os
sns.set_theme(style="darkgrid")



timestamp = datetime.now()

FOLDER = 'data/complete_log'



def load_data(FOLDER):
    FILES = [os.path.join(FOLDER, x) for x in os.listdir(FOLDER)]
    data = []
    for filename in FILES:
        data.append(pandas.read_csv(filename))
    data = pandas.concat(data)
    data = data[[x for x in data.columns if 'Unnamed' not in x]]
    column_type = {
        'normalization_fn': 'int32',
        'normalization_nu_fn': 'int32',
        'reg_fn_alpha': 'float32',
        'reg_proj': 'str',
        'reg_fn_logmin': 'float32',
        'grad_batch_size': 'int32',
        'batch_size': 'int32',
        'length_cutoff_factor': 'int32',
        'initial_flow': 'float32',
        'learning_rate': 'float32',
        'epochs': 'int32',
        'step_per_epoch': 'int32',
        'B_beta': 'float32',
        'path_redraw': 'int32',
        'neighborhood': 'int32',
        'graph_size': 'int32',
        'graph_generators': 'str',
        'inverse': 'bool',
        'flowestimator_opt': 'str',
        'seed': 'int32',
        'embedding': 'str',
        'optimizer': 'str',
        'loss_base': 'str',
        'reg_fn_gen': 'str',
        'reward_rescale': 'str',
    }
    for column in column_type:
        data[column] = data[column].astype(column_type[column])
    return data

data = load_data(FOLDER)

def pack_reg_fn_alpha(data,step=0.5):
    data['reg_fn_alpha'] = data['reg_fn_alpha'].div(step).floordiv(1).mul(step)
    return data
data = pack_reg_fn_alpha(data)
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


f,axs = plt.subplots(nrows=1,ncols=2,figsize=(32,18),sharey=True)
cmap = ['red','blue']# sns.color_palette("Paired")
a = sns.lineplot(
    data = here_data,
    x='omega',
    y='ExpectedReward',
    hue=hue_display_name3,
    legend=True,
    ax=axs[0],
    palette=cmap
)
plt.yticks(np.arange(0, 1, step=0.05))
here_data = here_data[here_data['omega']<-2.5]
here_data = here_data[here_data['omega']>-5]
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


#___________________________________Normalizations comparisons__________________________________
here_data = dataS15A_noBeta[dataS15A_noBeta['reg_proj']=='AddReg']
here_data = here_data[here_data['reward_rescale']=='Trivial']

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

