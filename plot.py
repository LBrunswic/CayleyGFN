from plotter import plot
import os
from utils import extract
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from time import time
from time import sleep
from datetime import datetime
A = extract()
A = {x:A[x] for x in A if A[x]['done']}
K = list(A.keys())
for x in A:
    if 'reward' not in A[x]['param']:
        A[x]['param']['reward'] = 'R_first_one'
    if 'load' not in A[x]['param']:
        A[x]['param']['load'] = 0
print('There are %s finished experiments' % len(A))
for key in A[K[0]]['param']:
    if key in ['folder','gpu']:
        continue
    choices  = set([A[x]['param'][key] for x in A])
    # if len(choices)==1:
    #     continue
    print(f"{key} : {choices}")
    # for  in SIZES:
    #     n = len({x:A[x] for x in A if A[x]['done'] and A[x]['param']['size']==size})
    #     print(f"{size} : {n}")

def aux(L,name,result_names='graphS{size}{folder}',target_init=1.0,Nepoch=10000,ER='spot',averaging_width=100):
    A = extract()
    to_plot = [result_names.format(**A[x]['param']) for x in L]
    plot(image_name=name+'.png',Nepoch=Nepoch,to_plot=to_plot,target_init=target_init,ER=ER,averaging_width=averaging_width)

def select(key,value):
    A = extract()
    A = {x: A[x] for x in A if A[x]['done']}
    return set([x for x in A if key in A[x]['param'] and A[x]['param'][key]==value])

def select_dict(keys_values,A=None):
    if A is None:
        A = extract()
        A = {x: A[x] for x in A if A[x]['done']}
    A = {x: A[x] for x in A if A[x]['done']}
    res = set(A.keys())
    for key in keys_values:
        if keys_values[key] is not None:
            res = set(x for x in res if key in A[x]['param']  and A[x]['param'][key]==keys_values[key])
        else:
            res = set(x for x in res if key not in A[x]['param'])


    return res
# aux(select('loss','Bengio,0'),name='bengioS30')
# aux(select('loss','pow,1.8'),name='pow1.8S30')

# loss='pow,1.8';aux(select('loss',loss),name=f'{loss}S30'.replace(',',''))
# loss='pow,2';aux(select('loss',loss),name=f'{loss}S30'.replace(',',''))
# loss='pow,0.5';aux(select('loss',loss),name=f'{loss}S30'.replace(',',''))
# loss='pow,0.9';aux(select('loss',loss),name=f'{loss}S30'.replace(',',''))
# loss='pow,1.1';aux(select('loss',loss),name=f'{loss}S30'.replace(',',''))

def score(L,metric=-1,val=5,interval=300,A=None):
    if A is None:
        A = extract()
        A = {x: A[x] for x in A if A[x]['done']}
    res = [x for x in L if np.all(A[x]['metrics'][metric,-interval:]>val)]
    return res


def show(VARIABLE={},threshold=[0,5,10,20,30,40,60],A=None,plot=True,bbox_to_anchor=(0,0),interval=50,ER='spot',Nepoch=20000,averaging_width=100,target_init=1.0):
    if A is None:
        A = extract()
        A = {x: A[x] for x in A if A[x]['done']}
    TODO = []
    for key in VARIABLE:
        if VARIABLE[key] == 'all':
            choices = set([A[x]['param'][key] for x in A if key in A[x]['param']])
            VARIABLE[key] = choices
        if VARIABLE[key] is None:
            VARIABLE[key] = [None]
    var = [list(x) for x in product(*VARIABLE.values())]
    for var_param in var:
        param = {}
        for i in range(len(VARIABLE)):
            param[list(VARIABLE.keys())[i]] = var_param[i]
        TODO.append(param)
    print(TODO)

    def sel(critere, threshold=[0, 5, 10, 20, 30, 40]):
        C = select_dict(critere, A=A)
        # print(len(C))
        # for c in C:
            # print(np.nan_to_num(A[c]['metrics'][-1,-interval:],0))
        Res = np.array([len([c for c in C if np.all(np.nan_to_num(A[c]['metrics'][-1,-interval:],0)>x)]) for x in threshold])
        return Res/len(C)

    plt.clf()
    C=[]
    name=str(time()).split('.')[0]
    name='D'+str(datetime.now()).replace(' ','H').replace('-','_').replace('.','mS').replace(':','_'
                                                                                                 '')
    for DO in TODO:
        if not plot:
            print('/'.join([str(DO[x]) for x in DO.keys()]),sel(DO,threshold=threshold))
        else:
            x = np.linspace(0,max(threshold),100)
            y = sel(DO, threshold=x)
            if not np.isnan(y[0]):
                plt.plot(x,y,label=' '.join([(str(k)+'-'+str(DO[k])).replace('_','') for k in DO]))
                C.extend(select_dict(DO, A=A))
    plt.legend(bbox_to_anchor=bbox_to_anchor)
    plt.savefig(os.path.join('results',name+'_a.png'), bbox_inches="tight")
    plt.clf()
    sleep(1)
    aux(C,name=name+'_b',Nepoch=Nepoch,ER=ER,averaging_width=averaging_width,target_init=target_init)
