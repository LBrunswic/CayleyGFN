from utils import extract
import seaborn as sns
import numpy as np
import pandas
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
fmri = sns.load_dataset("fmri")
database = extract()
simulation_list_stable = ['graphS5_1205', 'graphS5_1237', 'graphS5_1220', 'graphS5_1225', 'graphS5_1233', 'graphS5_1208', 'graphS5_1246', 'graphS5_1213', 'graphS5_1214', 'graphS5_1254', 'graphS5_1218', 'graphS5_1250', 'graphS5_1201', 'graphS5_1238', 'graphS5_1247', 'graphS5_1251', 'graphS5_1248', 'graphS5_1212', 'graphS5_1217', 'graphS5_1235', 'graphS5_1257', 'graphS5_1232', 'graphS5_1261', 'graphS5_1226', 'graphS5_1262', 'graphS5_1206', 'graphS5_1252', 'graphS5_1209', 'graphS5_1239', 'graphS5_1253', 'graphS5_1229', 'graphS5_1236', 'graphS5_1221', 'graphS5_1256', 'graphS5_1260', 'graphS5_1240', 'graphS5_1216', 'graphS5_1223', 'graphS5_1245', 'graphS5_1255', 'graphS5_1228', 'graphS5_1243', 'graphS5_1244', 'graphS5_1230', 'graphS5_1202', 'graphS5_1219', 'graphS5_1207', 'graphS5_1249', 'graphS5_1203', 'graphS5_1258', 'graphS5_1210', 'graphS5_1204', 'graphS5_1222', 'graphS5_1259', 'graphS5_1241', 'graphS5_1242', 'graphS5_1211', 'graphS5_1227', 'graphS5_1234', 'graphS5_1224', 'graphS5_1231', 'graphS5_1215']

simulation_list_unstable = ['graphS5_1275', 'graphS5_1274', 'graphS5_1273', 'graphS5_1278', 'graphS5_1277', 'graphS5_1279', 'graphS5_1272', 'graphS5_1276']

simulation_list_baseline = ['graphS5_1280']
# simulation_list_stable = ['graphS20_%s' % x for x in range(441,446)]
# simulation_list_unstable = ['graphS20_%s' % x for x in range(441,446)]
MAX_EPOCH = 300



Erew1 = np.concatenate([database[x]['metrics'][-1][:MAX_EPOCH] for x in simulation_list_stable])
Etau1 = np.concatenate([database[x]['metrics'][-2][:MAX_EPOCH] for x in simulation_list_stable])
name1 = np.concatenate([np.array([x]*MAX_EPOCH) for x in simulation_list_stable])
n_series_stable1 = len(simulation_list_stable)
epoch1 = np.concatenate([np.arange(MAX_EPOCH)]*n_series_stable1)
label1 = np.array(['stable'] * MAX_EPOCH)

loss1 = ['$ L_{FM,\Delta},f(x)=\log(1+x^2) g(x)=(1+|x+y|)$']*(MAX_EPOCH*n_series_stable1)

Erew2 = np.concatenate([database[x]['metrics'][-1][:MAX_EPOCH] for x in simulation_list_unstable])
Etau2 = np.concatenate([database[x]['metrics'][-2][:MAX_EPOCH] for x in simulation_list_unstable])
name2 = np.concatenate([np.array([x]*MAX_EPOCH) for x in simulation_list_unstable])
n_series_stable2 = len(simulation_list_unstable)
epoch2 = np.concatenate([np.arange(MAX_EPOCH)]*n_series_stable2)
label2 = np.array(['stable'] * MAX_EPOCH)
loss2 = ['$ L_{FM,g,nu}, g(x)=\log(x)^2 $']*(MAX_EPOCH*n_series_stable2)


Erew3 = np.array([np.mean(database[x]['metrics'][-1]) for x in simulation_list_baseline]*MAX_EPOCH)
Etau3 = np.array([np.mean(database[x]['metrics'][-2]) for x in simulation_list_baseline]*MAX_EPOCH)
name3 = np.concatenate([np.array([x]*MAX_EPOCH) for x in simulation_list_baseline])
n_series_stable3 = len(simulation_list_baseline)
epoch3 = np.concatenate([np.arange(MAX_EPOCH)]*n_series_stable3)
label3 = np.array(['baseline'] * MAX_EPOCH)
loss3 = ['MH no heuristic']*(MAX_EPOCH*n_series_stable3)


data = pandas.DataFrame({
    'epoch': np.concatenate([epoch1,epoch2,epoch3]),
    'E(R)': np.concatenate([Erew1,Erew2,Erew3]),
    'E(tau)': np.concatenate([Etau1,Etau2,Etau3]),
    'simulation':np.concatenate([name1,name2,name3]),
    'loss' : loss1+loss2+loss3
})

print(data)
print(fmri)
# # fig.clf()
# a = sns.lineplot(data = data, x='time',y='E(R)',hue="stability",ax=ax)
# ax.set_ylim(-10, 10)
# a.get_figure().savefig('test.png')



# Load an example dataset with long-form data


# Plot the responses for different events and regions
# a = sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)

fig,axs = plt.subplots(2,1)
a = sns.lineplot(
    data = data,
     x='epoch',
     y='E(R)',
     hue="loss",
     ax=axs[0],
     legend=False
)

# a.get_figure().savefig('ER5.png'


a = sns.lineplot(
    data = data,
     x='epoch',
     y='E(tau)',
     hue="loss",
     ax= axs[1],
     legend=False
)
fig.legend(loc='lower center', title='Team')
a.get_figure().savefig('Etau5.png')
