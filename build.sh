docker build --no-cache --tag ploptest - < Dockerfile
#docker run --rm --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
#docker run --rm --gpus '"device=1"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 numa_profile.py
#docker run --rm --cpuset-cpus=0-3 --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --gpu=0 --HP_FOLDER='Hyperparameters/experimental_settings' --pool_size='32' &> gpu0.log & disown
docker run --rm --cpuset-cpus=0-3 --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --gpu=0 --HP_FOLDER='Hyperparameters/experimental_settings_S15G3W1' --pool_size='8' &> gpu0.log & disown
#docker run --rm --cpuset-cpus=1-4 --gpus '"device=0"' -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --HP_FOLDER='Hyperparameters/experimental_settings_low' --pool_size='16' &> gpu1.log & disown
#docker run --rm --cpuset-cpus=5-24 -v /home/maxbrain/DATA/TaskForce/logs:/TASK/LOGS/ -v /home/maxbrain/DATA/TaskForce/Results:/TASK/RESULTS/ -v /home/maxbrain/DATA/TaskForce/Models:/TASK/MODELS/ ploptest  python3 launch.py --HP_FOLDER='Hyperparameters/experimental_settings_low_cpu' --pool_size='8' &> cpu.log & disown



