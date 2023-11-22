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

args = parser.parse_args().__dict__
FOLDER = args['HP_FOLDER']

for hp_file_name in os.listdir(FOLDER):
    hp_file_path = os.path.join(FOLDER,hp_file_name)
    hash = os.path.split(hp_file_path)[-1].split('.')[0]
    data_save = os.path.join('RESULTS',hash + '.csv')
    if not os.path.exists(data_save):
        print(f'Launching {hp_file_path}')
        os.system(f'python3 docker_main.py --pool_size=48 --hp_file={hp_file_path}')

