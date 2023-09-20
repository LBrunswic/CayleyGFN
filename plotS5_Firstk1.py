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
    }
    for column in column_type:
        data[column] = data[column].astype(column_type[column])
    return data

data = load_data(FOLDER)

def pack_reg_fn_alpha(data,step=0.5):
    data['reg_fn_alpha'] = data['reg_fn_alpha'].div(step).floordiv(1).mul(step)
    return data
data = pack_reg_fn_alpha(data)

hue_display_name='display'
data.insert(len(data.columns),hue_display_name,['']*len(data) )
link = np.array(['_']*len(data))
columns = ['optimizer','normalization_fn','normalization_nu_fn','rew_factor']

for column in  columns:
    data[hue_display_name] = data[hue_display_name] + data[column].astype('str') + link

data['EMaxSeenRew'] = data['EMaxSeenRew'].values / data['rew_factor'].values
data['ExpectedReward'] = data['ExpectedReward'].values / data['rew_factor'].values

dataS15 = data[data['graph_size']==15]
dataS15A = dataS15[dataS15['optimizer']=='Adam']
dataS15N = dataS15[dataS15['optimizer']=='Nesterov']

f,axs = plt.subplots(nrows=2,ncols=1,figsize=(32,18))

cmap = sns.color_palette("Paired")
a = sns.lineplot(
    data = dataS15A,
    x='reg_fn_alpha',
    y='EMaxSeenRew',
    hue=hue_display_name,
    legend=False,
    ax=axs[0],
    palette=cmap
)
plt.yticks(np.arange(0, 1, step=0.05))

a = sns.lineplot(
    data = dataS15A,
    x='reg_fn_alpha',
    y='ExpectedReward',
    hue=hue_display_name,
    legend=True,
    ax=axs[1],
    palette=cmap
)
sns.move_legend(
    axs[1], "lower center",
    bbox_to_anchor=(0.5, -.3), ncol=3,  framealpha=1.,fontsize='xx-large'
)
plt.yticks(np.arange(0, 1, step=0.05))
a.get_figure().savefig('ExpectedReward.png')

