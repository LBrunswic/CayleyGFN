from utils import extract
import seaborn as sns
import numpy as np
import pandas
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
fmri = sns.load_dataset("fmri")
database = extract()
simulation_list_stable = ['graphS20_474', 'graphS20_469', 'graphS20_489', 'graphS20_494', 'graphS20_479', 'graphS20_484', 'graphS20_499', 'graphS20_495', 'graphS20_475', 'graphS20_500', 'graphS20_480', 'graphS20_470', 'graphS20_485', 'graphS20_490', 'graphS20_486', 'graphS20_496', 'graphS20_501', 'graphS20_491', 'graphS20_476', 'graphS20_471', 'graphS20_481', 'graphS20_487', 'graphS20_497', 'graphS20_502', 'graphS20_477', 'graphS20_492', 'graphS20_482', 'graphS20_472', 'graphS20_493', 'graphS20_498', 'graphS20_478', 'graphS20_483', 'graphS20_503', 'graphS20_473', 'graphS20_488']
simulation_list_stable =['graphS20_534', 'graphS20_483', 'graphS20_535', 'graphS20_540', 'graphS20_543', 'graphS20_515', 'graphS20_539', 'graphS20_508', 'graphS20_561', 'graphS20_503', 'graphS20_527', 'graphS20_521', 'graphS20_557', 'graphS20_510', 'graphS20_531', 'graphS20_517', 'graphS20_512', 'graphS20_542', 'graphS20_526', 'graphS20_473', 'graphS20_538', 'graphS20_533', 'graphS20_529', 'graphS20_537', 'graphS20_551', 'graphS20_530', 'graphS20_524', 'graphS20_498', 'graphS20_525', 'graphS20_520', 'graphS20_541', 'graphS20_516', 'graphS20_544', 'graphS20_550', 'graphS20_560', 'graphS20_518', 'graphS20_493', 'graphS20_513', 'graphS20_553', 'graphS20_546', 'graphS20_511', 'graphS20_558', 'graphS20_554', 'graphS20_552', 'graphS20_478', 'graphS20_514', 'graphS20_556', 'graphS20_545', 'graphS20_488', 'graphS20_559', 'graphS20_532', 'graphS20_536', 'graphS20_548', 'graphS20_523', 'graphS20_563', 'graphS20_562', 'graphS20_528', 'graphS20_522', 'graphS20_549', 'graphS20_555', 'graphS20_509', 'graphS20_519', 'graphS20_547']

# simulation_list_unstable = ['graphS5_1275', 'graphS5_1274', 'graphS5_1273', 'graphS5_1278', 'graphS5_1277', 'graphS5_1279', 'graphS5_1272', 'graphS5_1276']

simulation_list_baseline = ['graphS20_450']
# simulation_list_stable = ['graphS20_%s' % x for x in range(441,446)]
simulation_list_unstable = ['graphS20_%s' % x for x in range(441,446)]
MAX_EPOCH = 100



Erew1 = np.concatenate([database[x]['metrics'][-1][:MAX_EPOCH] for x in simulation_list_stable])
Etau1 = np.concatenate([database[x]['metrics'][-2][:MAX_EPOCH] for x in simulation_list_stable])
name1 = np.concatenate([np.array([x]*MAX_EPOCH) for x in simulation_list_stable])
n_series_stable1 = len(simulation_list_stable)
epoch1 = np.concatenate([np.arange(MAX_EPOCH)]*n_series_stable1)
label1 = np.array(['stable'] * MAX_EPOCH)

loss1 = ['$ L_{FM,\Delta}~~f(x)=\log(1+x^2) ~~g(x)=1+|x+y|^{0.9}$']*(MAX_EPOCH*n_series_stable1)

Erew2 = np.concatenate([database[x]['metrics'][-1][:MAX_EPOCH] for x in simulation_list_unstable])
Etau2 = np.concatenate([database[x]['metrics'][-2][:MAX_EPOCH] for x in simulation_list_unstable])
name2 = np.concatenate([np.array([x]*MAX_EPOCH) for x in simulation_list_unstable])
n_series_stable2 = len(simulation_list_unstable)
epoch2 = np.concatenate([np.arange(MAX_EPOCH)]*n_series_stable2)
label2 = np.array(['stable'] * MAX_EPOCH)
loss2 = ['$ L_{FM,g,nu}~~ g(x)=\log(x)^2 $']*(MAX_EPOCH*n_series_stable2)


Erew3 = np.array([np.mean(database[x]['metrics'][-1]) for x in simulation_list_baseline]*MAX_EPOCH)
Etau3 = np.array([np.mean(database[x]['metrics'][-2]) for x in simulation_list_baseline]*MAX_EPOCH)
name3 = np.concatenate([np.array([x]*MAX_EPOCH) for x in simulation_list_baseline])
n_series_stable3 = len(simulation_list_baseline)
epoch3 = np.concatenate([np.arange(MAX_EPOCH)]*n_series_stable3)
label3 = np.array(['baseline'] * MAX_EPOCH)
loss3 = ['baseline']*(MAX_EPOCH*n_series_stable3)

Etaulab = r"$E(\tau)$"
ERlab = r"$E(R)$"
data = pandas.DataFrame({
    'epoch': np.concatenate([epoch1,epoch2,epoch3]),
    ERlab: np.concatenate([Erew1,Erew2,Erew3]),
    Etaulab: np.concatenate([Etau1,Etau2,Etau3]),
    'simulation':np.concatenate([name1,name2,name3]),
    'loss' : loss1+loss2+loss3
})

print(data)
# print(fmri)
# # fig.clf()
# a = sns.lineplot(data = data, x='time',y='E(R)',hue="stability",ax=ax)
# ax.set_ylim(-10, 10)
# a.get_figure().savefig('test.png')



# Load an example dataset with long-form data


# Plot the responses for different events and regions
# a = sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri)

fig,axs = plt.subplots(2,1,figsize=(15,8))
a = sns.lineplot(
    data = data,
     x='epoch',
     y=ERlab,
     hue="loss",
     style="loss",
     ax=axs[0],
     linewidth=5,
     legend=False
)

# a.get_figure().savefig('ER5.png'


a = sns.lineplot(
    data = data,
     x='epoch',
     y=Etaulab,
     hue="loss",
     style="loss",
     ax= axs[1],
     linewidth=5,
     legend=True
)
sns.move_legend(
    axs[1], "lower center",
    bbox_to_anchor=(0.5, -.3), ncol=3, title=None, framealpha=1.,fontsize='xx-large'
)
a.get_figure().savefig('EtauA20.png')
