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



for hp_file_name in os.listdir(FOLDER):
    hp_file_path = os.path.join(FOLDER,hp_file_name)
    os.system(f'python3 docker_main.py --pool_size=48 --hp_file={hp_file_path}')

