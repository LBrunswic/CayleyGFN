import utils
import os
import itertools
from time import sleep,time
from datetime import datetime
from multiprocessing import Process,active_children
import subprocess
from utils import extract
import logging
from plot import show
import numpy as np

A = extract()
A = {x:A[x] for x in A if A[x]['done']}

def GET_CONF():
    with open('HP.py','r') as f:
        A = ''.join(f.readlines())
        VARIABLE_LIST = eval(A)
        return VARIABLE_LIST

VARIABLE_LIST = GET_CONF()
show(threshold=np.arange(VARIABLE_LIST[0]['size'][0]*100)/100,VARIABLE={key:VARIABLE_LIST[0][key] for key in VARIABLE_LIST[0] if key not in ['seed']},averaging_width=10,target_init=1.)
