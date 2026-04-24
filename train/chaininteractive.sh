#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ((i=1; i<=20; i++)); do
  echo "Starting interactive epoch run $i"

  success=0
  while [ $success -eq 0 ]; do
    salloc -J DESIMAE \
         --nodes=4 \
         --qos=interactive \
         --time=01:30:00 \
         --gpus-per-node=4 \
         --constraint=gpu \
         --account=desi_g \
    bash -lc '
      source ~/.bashrc
      conda activate /global/cfs/cdirs/desi/users/pzehao/envs/peng
      cd "'"$SCRIPT_DIR"'"
      srun --ntasks-per-node=4 --cpus-per-task=32 python MaeTrainOneEpoch.py
      echo "One epoch finished; releasing interactive allocation."
      exit 0
    '

    if [ $? -eq 0 ]; then
      success=1
    else
      echo "salloc failed, retrying in 60s..."
      sleep 60
    fi
  done

  echo "Finished interactive epoch run $i"
done