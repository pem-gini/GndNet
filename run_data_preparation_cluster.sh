#!/usr/bin/zsh

#SBATCH --cpus-per-task=11
#SBATCH --mem-per-cpu=500M
#SBATCH --job-name=GND_NET_DATA_PREP
#SBATCH --output=output.%J.txt
#SBATCH --time=00:45:00

export CONDA_ROOT=$HOME/miniconda3
export PATH="$CONDA_ROOT/bin:$PATH"

source activate gndnet

cd dataset_utils/gnd_data_generator
python3 semKitti_morph_data_camera.py
