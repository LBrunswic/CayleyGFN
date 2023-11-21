import os
FOLDER = os.path.join('Hyperparameters','experimental_settings')
for hp_file_name in os.listdir(FOLDER):
    hp_file_path = os.path.join(FOLDER,hp_file_name)
    os.system(f'docker_main.py --pool_size=48 --hp_file={hp_file_path}')

