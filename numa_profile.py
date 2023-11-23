import os
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(
                    prog='numaprofile_CayleyGFN',
                    description='Test cpu affinity'
)


parser.add_argument(
    '--N_ATTEMPTS',
    type=int,
    default=32,
    help='Number of experiments Repeat',
)
parser.add_argument(
    '--memory',
    type=int,
    default=-1,
    help='Non positive value activate memory growth. Positive values set the actual GPU memory limit in MB'
)
args = parser.parse_args().__dict__

N_ATTEMPTS = 2
N_CPU = 32
for attempt in range(N_ATTEMPTS):
    for PACK in range(1,9):
        RESULTS_FILE = os.path.join('RESULTS','numa_profile_%s.npy' % time.time())
        results = np.zeros((N_CPU,N_ATTEMPTS))
        for cpu in range(8,N_CPU-PACK):
            cpurange=f'{cpu}-{cpu+PACK-1}'
            T = time.time()
            os.system(f'numactl --physcpubind={cpurange} python3 docker_main.py > cpu_{cpurange}.log')
            results[PACK-1,cpu,attempt] = time.time()-T
            with open(RESULTS_FILE,'wb') as f:
                np.save(f,results)



