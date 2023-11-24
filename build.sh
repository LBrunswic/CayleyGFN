docker build --no-cache --tag ploptest - < Dockerfile
#docker run --rm --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
#docker run --rm --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
docker run --rm --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --numactl=0-3 --HP_FOLDER='Hyperparameters/experimental_settings' --pool_size='48' & disown
docker run --rm --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --numactl=4-7 --HP_FOLDER='Hyperparameters/experimental_settings_low' --pool_size='16' & disown



