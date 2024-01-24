#!/usr/bin/zsh

#SBATCH --job-name=GND_NET_MODEL_TRAINING
#SBATCH --gres=gpu:1
#SBATCH --output=output.%J.txt
#SBATCH --time=10:00:00

export CONDA_ROOT=$HOME/miniconda3
export PATH="$CONDA_ROOT/bin:$PATH"

source activate gndnet

module load CUDA
echo; export; echo; nvidia-smi; echo

python3 main.py -s --config config/config_kittiSem2.yaml