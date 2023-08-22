import numpy as np
from utils import extract
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
A = extract()


L = ['graphS5_%s' % i for i in range(0,96)]
A = {x:A[x] for x in A if A[x]['done'] and x in L}
HP = set()
for x in A:
    HP=HP.union(list(A[x]['param']))


data = {
    'HP_'+x:np.array([],dtype=str)
    for x in HP
}
# data['name'] = np.array([])
data['epochs'] = np.array([])
metrics = [
    'M_ErrorRhat',
    'M_ErrorRbar',
    'M_Loss',
    'M_Init',
    'M_totalReward',
    'M_MaxReward',
    'M_Nstate',
    'M_ExpectedLen',
    'M_ExpectedR',
]
TTT = [
    lambda x:x,
    lambda x:x,
    lambda x:np.clip(np.log10(1e-8+x),-8,5),
    lambda x:x,
    lambda x:x,
    lambda x:x,
    lambda x:x,
    lambda x:x,
    lambda x:x,
]
for metric in metrics:
    data[metric] = np.array([])


for x in A:
    length = A[x]['param']['epochs']
    data['epochs'] =np.concatenate([
        data['epochs'],
        np.arange(length)
    ])
    for key in A[x]['param']:
        data['HP_'+key] = np.concatenate([
            data['HP_'+key],
            np.array([A[x]['param'][key]]*length)
        ],dtype=str)

    for i,key in enumerate(metrics):
        data[key] = np.concatenate([
            data[key],
            TTT[i](A[x]['metrics'][i])
        ])

diff = ['loss']


data['diff'] = data['HP_' + diff[0]].copy()
for x in diff[1:]:
    data['diff'] = np.char.add(data['diff'],np.full_like(data['diff'],'_'))
    data['diff'] = np.char.add(data['diff'],data['HP_' + x])
# rais


data = pandas.DataFrame(data)
print(data)

# raise

sns.set_theme(style="darkgrid")
cmap = sns.color_palette("Spectral", as_cmap=True)
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(32,18))
a = sns.lineplot(
    data = data,
     x='epochs',
     y='M_Loss',
     hue="diff",
     ax=axs[0,0],
     legend=False,
     # palette=cmap
)


a = sns.lineplot(
    data = data,
     x='epochs',
     y='M_Init',
     hue="diff",
     ax= axs[0,1],
     legend=False,
     # palette=cmap
)

a = sns.lineplot(
    data = data,
     x='epochs',
     y='M_ExpectedR',
     hue="diff",
     ax= axs[1,1],
     legend=False,
     # palette=cmap
)

a = sns.lineplot(
    data = data,
     x='epochs',
     y='M_ExpectedLen',
     hue="diff",
     ax= axs[1,0],
     legend=True,
     # palette=cmap
)
sns.move_legend(
    axs[1,0], "lower center",
    bbox_to_anchor=(0.5, -.3), ncol=3, title='_'.join(diff), framealpha=1.,fontsize='xx-large'
)
from datetime import datetime

a.get_figure().savefig('results/%s.png' % datetime.now())
