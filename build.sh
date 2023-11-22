docker build --no-cache --tag ploptest - < Dockerfile
docker run --rm numa_profile0 -it --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
docker run --rm numa_profile1 -it --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
#docker run --rm devtest1 -it --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --HP_FOLDER='Hyperparameters/experimental_settings'
#docker run --rm devtest1 -it --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --HP_FOLDER='Hyperparameters/experimental_settings_low'



