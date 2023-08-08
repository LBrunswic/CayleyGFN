import os
import time
import datetime
import sys, select
from itertools import zip_longest
from utils import extract
FOLDER = 'tests'
os.system('clear')
height = 40
while True:
    logs = extract()
    done = len({x for x in logs if logs[x]['done']})
    logs = {x : logs[x] for x in logs if 'last_record' in logs[x] and 'ETA' in logs[x] and 'EPOCH' in logs[x] and (logs[x]['EPOCH'][0] is not None) and (logs[x]['last_record'] is not None)}
    logs = {x : logs[x] for x in logs if datetime.datetime.now()-logs[x]['last_record'] < datetime.timedelta(seconds=60)}
    previous = ''
    OUTPUT=str(done)
    i=0
    PID = []
    for folder in logs.keys():
        PID.append(logs[folder]['PID'])
        if previous != folder.split('_')[0]:
            # print('_________________________________________________')
            previous = folder.split('_')[0]
            # print(previous + '_________________________________________________')
            OUTPUT += previous + '_________________________________________________'+'\n'

        show = folder.split('_')[1]
        if logs[folder]['done']:
            pass
            # OUTPUT += f'{show:10}' + ' : Done!' + '\n'
        else:
            # last_record = logs[folder]['last_record']
            last_record = datetime.datetime.now()-logs[folder]['last_record']
            ETA = logs[folder]['ETA']
            EPOCH_cur,EPOCH_final = logs[folder]['EPOCH']
            OUTPUT += f'{i:3}) {show:6}: {last_record} -> {ETA} -- {EPOCH_cur}/{EPOCH_final}\n'
        i += 1

    os.system('clear')
    OUTPUT = OUTPUT.split('\n')
    for _ in range(OUTPUT.count('')):
        OUTPUT.remove('')
    width =  max([len(line) for line in OUTPUT])
    OUTPUT = [f'{out:99}' for out in OUTPUT]
    OUTPUT = '\n'.join([ x+'\t\t'+y for x,y in zip_longest(OUTPUT[:height],OUTPUT[height:],fillvalue='')])
    # OUTPUT = '\n'.join([x+'\t\t'+y for x,y in zip_longest(OUTPUT[0::2],OUTPUT[1::2],fillvalue='')])
    print(OUTPUT)
    print(datetime.datetime.now(),'Kill one ?',end='')
    i, o, e = select.select([sys.stdin], [], [], 2)
    if (i):
        A = sys.stdin.readline().strip()
        if A == 'all' or A == 'All':
            print('Ok for All !')
            for j in range(len(PID)):
                if PID[j]>=0:
                    os.system('kill -USR1 %s' % PID[j])

        else:
            A = int(A)
            if int(PID[A])>0:
                print('Ok for %s !' % A)
                os.system('kill -USR1 %s' %  PID[A] )
            else:
                print('This process already stopped')
    else:
        print('Ok Kill None !')
    time.sleep(0.1)
