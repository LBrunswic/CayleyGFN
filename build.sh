#!/bin/bash

while getopts "f:c:g:b:m:l:p:t:" option; do
   case "$option" in
       f) hpfile=${OPTARG};;
       c) cpu=${OPTARG};;
       g) gpu=${OPTARG};;
       b) build=${OPTARG};;
       m) memory=${OPTARG};;
       l) laptop=${OPTARG};;
       p) pool=${OPTARG};;
       t) test=${OPTARG};;
   esac
done

echo "Options chosen:"
echo 'build: '$build
echo 'cpu: '$cpu
echo 'gpu: '$gpu
echo 'hpfile: '$hpfile
echo 'memory: '$memory
echo 'laptop: '$laptop
echo 'pool: '$pool
echo 'test: '$test


RESULTS=${hpfile::-3}
mkdir -v -p /home/maxbrain/DATA/TaskForce/Results/CayleyGFN/$RESULTS
mkdir -v -p /home/maxbrain/DATA/TaskForce/logs/CayleyGFN/$RESULTS
mkdir -v -p /home/maxbrain/DATA/TaskForce/Models/CayleyGFN/$RESULTS


if [[ $build -eq 1 ]]; then
  if [[ $laptop -eq 0 ]]; then
    docker build --no-cache --tag cayleygfn - < Dockerfile
  else
    docker build --no-cache --tag cayleygfn - < Dockerfile_laptop
  fi
fi


if  [[ $gpu -eq -1 ]]; then
  gpu=""
fi
echo $gpucommand

if [[ $laptop -eq 0 ]]; then
  echo 'Laptop is 0'
  docker run --rm --cpuset-cpus=$cpu --gpus \"device=$gpu\"  \
  -v /home/maxbrain/DATA/TaskForce/logs/CayleyGFN/$RESULTS:/TASK/LOGS/ \
  -v /home/maxbrain/DATA/TaskForce/Results/CayleyGFN/$RESULTS:/TASK/RESULTS/ \
  -v /home/maxbrain/DATA/TaskForce/Models/CayleyGFN/$RESULTS:/TASK/MODELS/ \
  -v `realpath $hpfile`:/TASK/hp_gen.py \
  cayleygfn python3 launch.py --hp_gen_file=hp_gen.py --gpu_memory=$memory --pool_size=$pool --test=$test  &> gpu$gpu'_'$cpu.log & disown
else
  echo 'Laptop is 1'
  docker run --rm --cpuset-cpus=$cpu --gpus \"device=$gpu\"  \
  -v /home/maxbrain/DATA/IA/Experiments/CayleyGFN/:/TASK/ \
  -v /home/maxbrain/DATA/TaskForce/logs/CayleyGFN/$RESULTS:/TASK/LOGS/ \
  -v /home/maxbrain/DATA/TaskForce/Results/CayleyGFN/$RESULTS:/TASK/RESULTS/ \
  -v /home/maxbrain/DATA/TaskForce/Models/CayleyGFN/$RESULTS:/TASK/MODELS/ \
  -v `realpath $hpfile`:/TASK/hp_gen.py cayleygfn \
  python3 launch.py --hp_gen_file=hp_gen.py --gpu_memory=$memory --pool_size=$pool --test=$test &> gpu$gpu'_'$cpu.log \
  & disown
fi

