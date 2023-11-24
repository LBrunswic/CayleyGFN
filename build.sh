docker build --no-cache --tag ploptest - < Dockerfile
#docker run --rm --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
#docker run --rm --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
docker run --rm --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  numactl --physcpubind=0-3 "python3 launch.py --HP_FOLDER='Hyperparameters/experimental_settings' --pool_size='48'" & disown
docker run --rm --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  numactl --physcpubind=4-7 "python3 launch.py --HP_FOLDER='Hyperparameters/experimental_settings_low' --pool_size='16'"" & disown



