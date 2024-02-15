import numpy as np
import shutil
import os
from datetime import datetime
import numpy as np
from time import time

def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename, 'rb') as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size)).decode(encoding='utf-8')
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


def get_gpu_stat(file_name='gpu_stat'):
    os.system('nvidia-smi>%s' % file_name)
    with open(file_name,'r') as f:
        A = f.readlines()
    return A
def gpu_memory():
    while True:
        A = get_gpu_stat()
        B = [x.replace(' ','').replace('MiB','').split('/') for x in  ''.join(A).split('|') if 'MiB' in x]
        try:
            res = [(int(x),int(y))  for x,y in B[:1]]
            return res
        except:
            print (A)



def extract(TESTS_FOLDER = 'tests',remove=False,load_log = False):
    state={}
    folder_list = os.listdir(TESTS_FOLDER)
    folder_list.sort()
    for test_folder in folder_list:
        state[test_folder] = {}
        state[test_folder]['folder_path'] = os.path.abspath(os.path.join(TESTS_FOLDER,test_folder))
        state[test_folder]['param_path'] = os.path.abspath(os.path.join(TESTS_FOLDER,test_folder,'HP_dict'))
        state[test_folder]['results_path'] = os.path.abspath(os.path.join(TESTS_FOLDER,test_folder,'Errors.npy'))
        state[test_folder]['log_path'] = os.path.abspath(os.path.join(TESTS_FOLDER,test_folder,'training.log'))
        state[test_folder]['PID_path'] = os.path.abspath(os.path.join(TESTS_FOLDER,test_folder,'PID'))

        try:
            with open(state[test_folder]['PID_path'],'r') as f:
                state[test_folder]['PID'] = int(f.readlines()[0])
        except:
            state[test_folder]['PID'] = -1
        try:
            with open(state[test_folder]['param_path'],'r') as f:
                 state[test_folder]['param'] = eval(f.readline())
        except:
            del state[test_folder]
            continue
        try:
            if load_log:
                with open(state[test_folder]['log_path'],'r') as f:
                    state[test_folder]['log'] = f.readlines()
                if 'All done' not in state[test_folder]['log'][-1]:
                    state[test_folder]['done'] = False
                else:
                    state[test_folder]['done'] = True
            else:
                for line in reverse_readline(state[test_folder]['log_path']):
                    if 'All done' not in line:
                        state[test_folder]['done'] = False
                    else:
                        state[test_folder]['done'] = True
                    break
                state[test_folder]['EPOCH'] = None,None
                state[test_folder]['ETA'] = None
                state[test_folder]['last_record'] = None
                for line in reverse_readline(state[test_folder]['log_path']):
                    if 'EPOCH' in line:
                        state[test_folder]['EPOCH'] = line.split('EPOCH : ')[1].split('/')
                        state[test_folder]['last_record'] = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                    if 'ETA' in line:
                        state[test_folder]['ETA'] = datetime.strptime(line.split('ETA : ')[1].split('.')[0], '%Y-%m-%d %H:%M:%S')
                    if state[test_folder]['EPOCH'][0] is not None and state[test_folder]['ETA'] is not None:
                        break
                with open(state[test_folder]['log_path'], 'r') as f:
                    state[test_folder]['first_record'] = datetime.strptime(f.readline()[:19],'%Y-%m-%d %H:%M:%S')


            with open(state[test_folder]['results_path'],'br') as f:
                state[test_folder]['metrics'] = np.load(f)
        except Exception as e:
            state[test_folder]['done'] = False
            # print(e)
        if not state[test_folder]['done']:
            # del state[test_folder]
            if remove:
                shutil.rmtree(state[test_folder]['folder_path'])
    return state

def remove(L,TESTS_FOLDER = 'tests'):
    A = extract(TESTS_FOLDER = 'tests')
    for x in L:
        shutil.rmtree(A[x]['folder_path'])

def concat_dict_of_ndarray(dict_list,check_types=True,check_lengths=True):
    """
    Concatenates list of dicts of lists into one dict of lists
    if check_type is True, checks that for each keys list of every dict have the same type
    if check_lengths is True, checks that for each keys list of every dict have the same length
    :param dict_list:
    :param check_types:
    :return:
    """
    if check_types:
        for key in dict_list[0].keys():
            for d in dict_list:
                assert isinstance(d[key], np.ndarray), 'key %s is not a list'%key
                assert d[key].dtype == dict_list[0][key].dtype, 'key %s has different types'%key
    if check_lengths:
        for d in dict_list:
            i = 0
            for key in dict_list[0].keys():
                if i == 0:
                    L = d[key].shape
                    i =1
                else:
                    assert np.all(L == d[key].shape), 'key %s has different lengths'%key
    res = {}
    for key in dict_list[0].keys():
        res[key] = []
        print(type(d[key]))
        for d in dict_list:
            res[key] += list(d[key])
    return res