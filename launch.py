import os
import argparse
parser = argparse.ArgumentParser(
                    prog='CayleyGFN',
                    description='Launch batch of experiment',
)
FOLDER = os.path.join('Hyperparameters','experimental_settings')
parser.add_argument(
    '--HP_FOLDER',
    type=str,
    default=FOLDER,
    help=f'Provide the experimental setting file, If not provided, defaults to {FOLDER}'
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


args = parser.parse_args().__dict__


for hp_file_name in os.listdir(FOLDER):
    hp_file_path = os.path.join(FOLDER,hp_file_name)
    hash = os.path.split(hp_file_path)[-1].split('.')[0]
    data_save = os.path.join('RESULTS',hash + '.csv')
    if not os.path.exists(data_save):
        print(f'Launching {hp_file_path}')
        os.system(f"python3 docker_main.py --gpu={args['gpu']} --pool_size={args['pool_size']} --hp_file={hp_file_path}")

