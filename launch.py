import os
import argparse
from time import sleep,time
parser = argparse.ArgumentParser(
                    prog='CayleyGFN',
                    description='Launch batch of experiment',
)
parser.add_argument(
    '--hp_gen_file',
    type=str,
    help='Provide the experimental setting file'
)
parser.add_argument(
    '--pool_size',
    type=int,
    default=32,
    help=f'Set the experiment pool size, the effect of results is small ans is simply intended to improve efficiency'
)
parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Set GPU used for the experiment'
)

parser.add_argument(
    '--gpu_memory',
    type=int,
    default=0,
    help=f'Set GPU used for the experiment'
)
parser.add_argument(
    '--test',
    type=int,
    default=0,
    help=f'Set test mode'
)
parser.add_argument(
    '--prescript',
    type=str,
    default='',
    help=f'Set prescript mode'
)

args = parser.parse_args().__dict__
FOLDER = 'HP'
if args['prescript'] != '':
    print(os.path.abspath('.'),'dewde')
    print(os.listdir('.'))
    # sleep(1)
    os.system('bash '+args['prescript'])


with open(args['hp_gen_file'],'r') as f:
    exec(f.read())


for hp_file_name in os.listdir(FOLDER):
    hp_file_path = os.path.join(FOLDER,hp_file_name)
    hash = os.path.split(hp_file_path)[-1].split('.')[0]
    data_save = os.path.join('RESULTS',f'{hash}_{args["pool_size"]}.csv')
    if not os.path.exists(data_save):
        print(f'Launching {hp_file_path} -> {data_save}...')
        if args['test'] == 0:
            T = time()
            os.system(f"python3 docker_main.py --gpu={args['gpu']} --pool_size={args['pool_size']} --hp_file={hp_file_path}")
            print(f'Done in {time()-T}')
        print('done!')
    else:
        print(f'{hp_file_path} already done! The file {data_save} exists!')


