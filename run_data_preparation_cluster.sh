#!/usr/bin/zsh

#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=2048M
#SBATCH --job-name=GND_NET_DATA_PREP
#SBATCH --output=output.%J.txt
#SBATCH --time=00:30:00

cd dataset_utils/gnd_data_generator
python3 semKitti_morph_data.py