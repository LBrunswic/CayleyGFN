import os
import time
import numpy as np
N_ATTEMPTS = 10
N_CPU = 32
RESULTS_FILE = 'numa_profile.npy'
results = np.zeros((N_CPU,N_ATTEMPTS))
for cpu in range(N_CPU):
    for attempt in range(N_ATTEMPTS):
        T = time.time()
        os.system(f'numactl --physcpubind={cpu} python3 docker_main.py profile 2345 > cpu_{cpu}.log')
        results[cpu,attempt] = time.time()-T
        with open(RESULTS_FILE,'wb') as f:
            np.save(f,results)


