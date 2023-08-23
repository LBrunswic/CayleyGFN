import utils
import os, sys
import itertools
from time import sleep,time
from datetime import datetime
from multiprocessing import Process,active_children
import subprocess
from utils import extract
import logging
import numpy as np

GPUS = [0]
Dt = int(sys.argv[1])

logger = logging.getLogger('launcher')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join('logs','launch%s.log' % datetime.now()),mode='w')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info('START')
A = extract()
A = {x:A[x] for x in A if A[x]['done']}

def GET_CONF():
    with open('HP.py','r') as f:
        A = ''.join(f.readlines())
        VARIABLE_LIST = eval(A)
        return VARIABLE_LIST

# def BUILD_NEXT():
    # VARIABLE = GET_CONF()
    # A = extract()
    # for var in VARIABLE:
    #     param_set = [list(x) for x in itertools.product(*VARIABLE.values())]
    #
    #     { x for x in VARIABLE  if all( for k in set_param)}

def launch_sim(param,EXEC_NUMB):
    param['folder'] = '_%s' % EXEC_NUMB
    folder = os.path.join('tests','graphS%s_%s' % (param['size'],EXEC_NUMB))
    # return
    os.makedirs(folder, exist_ok=True)
    cmd = ['python', 'mainv2.py']
    for key in param:
        value=param[key]
        cmd.append('--%s=%s' % (key,value))
    param['start_date'] = str(datetime.now())
    param['start_timestamp'] = str(time())
    with open(os.path.join(folder,'HP_dict'), "w") as hp_file:
        hp_file.write(str(param))
    print(cmd)
    with open(os.path.join('logs','log%s.log' % EXEC_NUMB), "w") as outfile:
        subprocess.run(cmd, stdout=outfile,stderr=outfile)

def is_done(param):
    B = list(A.keys())
    for key in param:
        B = [x for x in B if key in A[x]['param'] and A[x]['param'][key]==param[key] ]
    return len(B)

def find_EXEC_NUMB(size):
    A = [int(x.split('_')[1]) for x in os.listdir('tests') if ('graphS%s_' % size) in x]
    A.sort()
    for i in range(len(A)):
        if A[i] > i:
            return i
    return len(A)
Processes = []

TODO = []
VARIABLE_LIST = GET_CONF()
for VARIABLE in VARIABLE_LIST:
    var = [list(x) for x in itertools.product(*VARIABLE.values())]
    for var_param in var:
        pass
        param = {}
        for i in range(len(VARIABLE)):
            param[list(VARIABLE.keys())[i]] = var_param[i]
        TODO.append(param)
logger.info(str(TODO))
for param in TODO:
    if is_done(param):
        logger.info('Already done: %s' % param)
        continue
    print(param)
    logger.info('Not done: %s' % param)
    sleep(Dt)
    done = False
    while not done:
        logger.handlers[0].flush()
        gpu_usage = utils.gpu_memory()
        for i in range(len(GPUS)):
            try:
                gpu = GPUS[i]
                used, total = gpu_usage[gpu]
            except:
                break
            print(used,total,2**param['memory_limit']+2**10,end='\r')
            if total-used>2**param['memory_limit']+2**10:
                print()
                param['gpu'] = gpu
                EXEC_NUMB= find_EXEC_NUMB(param['size'])
                p = Process(target=launch_sim, args=(param,EXEC_NUMB))
                p.start()
                Processes.append(p)
                logger.info(param)
                EXEC_NUMB += 1
                GPUS = GPUS[i + 1:] + GPUS[:i + 1]
                done = True
                sleep(2)
                break
        if not done:
            logger.info('no GPU available, we retry later')
