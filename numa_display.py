import os
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(
                    prog='numaprofile_display_CayleyGFN',
                    description='display cpu affinity'
)


parser.add_argument(
    '--profile_file',
    type=str,
    default='RESULTS/numa_profile.npy',
    help='Provide de experimental results file. If not provided, defaults to RESULTS/numa_profile.npy',
)
args = parser.parse_args().__dict__

a =  np.load(args['profile_file'])
print(a.shape)
for pack in range(8):
    for attempt in range(2):
        plt.plot(np.arange(32),a[pack,:,attempt], label=f'pack={pack+1} attempt={attempt}')
        plt.legend(loc='right')
plt.show()