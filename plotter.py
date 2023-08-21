#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argcomplete, argparse, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import math
plt.rcParams['text.usetex'] = True



def pad(a,length):
    len_a = a.shape[0]
    return np.interp(np.arange(length),(length-1)/(len_a-1)*np.arange(len_a),a)

def plot(to_plot = [] ,in_folder='tests', dest_folder='results/',file_name='Errors.npy',image_name='Errors.png',Nepoch=10000,target_init=1.0,ER='spot',averaging_width=100):
    ERhat, ERbar, loss, init, totR, MaxR,Nstate,Elen,ERew = [[] for _ in range(9)]
    all = [ERhat, ERbar, loss, init, totR, MaxR, Nstate, Elen, ERew]

    n_graph=len(to_plot)
    ma_width=averaging_width
    for result in to_plot:
        with open(os.path.join(in_folder,result,file_name),'rb') as f:
            A = np.load(f)
            for i,j in zip(*[list(x) for x in np.where(np.isnan(A))]):
                A[i,j]=1
        for i in range(9):
            all[i].append(A[i][:Nepoch])
    for i in range(9):
        all[i] = np.stack(all[i])
        print(all[i].shape)
    all[4] = all[4]/all[-3]/target_init
    all[4] = all[-2]
    all[5] = all[-1]
    print(ER)
    if ER=='max':
        all[5] = np.maximum.accumulate(all[5],axis=1)
    for i in range(7):
        v = np.cumsum(all[i],axis=1)
        all[i] = np.concatenate([all[i][:,:ma_width], (v[:,ma_width:] - v[:,:-ma_width]) / ma_width],axis=1)

        if i == 3:
            all[i] = np.clip(np.abs(all[i]-target_init)/target_init,1e-8,10) #np.abs(all[i]-target_init)/target_init
        if i in [0,1,2,3]:
            all[i] = np.log10(1e-10+all[i])


    x = np.arange(Nepoch)
    fig, axs = plt.subplots(nrows=3,ncols=2,figsize=(32,18))
    y_axes_label = [r'$\log E_{Rhat}$',r'$\log E_{Rbar}$',r'$\mathcal L$',r'$\mathcal E_I$',r'$E(\tau)$',r'$E(R)$']
    colormap = mpl.cm.get_cmap('Set1')
    color='k'
    def c1(x,m=0,M=1):
        return colormap(m+x/n_graph*(M-m))
    axes = []
    for r in axs:
        for c in r:
            axes.append(c)
    axs=axes
    print(axs,len(all),to_plot)
    for i in range(6):
        # print(i)
        axs[i].set_ylabel(y_axes_label[i], position=(0, 0.35), rotation=90, color=color, alpha=1)
        axs[i].set_xlabel('Epochs')
        axs[i].tick_params(axis='y', rotation=45, labelcolor=color)
        for j in range(n_graph):
            axs[i].plot(x, all[i][j], color=c1(j), label=to_plot[j].replace('_','-'))
    axs[0].legend(fancybox=True, shadow=True, ncol=1)
    # fig.tight_layout()
    plt.savefig(os.path.join(dest_folder,image_name),dpi=300)
    plt.clf()
    return ERhat,ERbar,loss,init,totR,MaxR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    results=os.listdir('tests')
    parser.add_argument("--results",default=os.path.join('tests','graphS5_default'))
    parser.add_argument("--target_init",default=0.1)
    parser.add_argument("--file_name",default='Error')
    parser.add_argument("--epochs",default=-1)
    args = parser.parse_args()
    to_plot = args.results.split(',')
    target_init = float(args.target_init)
    name = str(args.file_name)
    Nepoch = int(args.epochs)
    assert(all([x in results for x in to_plot]))
    plot(image_name=name+'.png',Nepoch=Nepoch,to_plot=to_plot,target_init=target_init)
