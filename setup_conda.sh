export CONDA_ROOT=$HOME/miniconda3
#$CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda create --name gndnet python==3.9

source activate gndnet

./setup.sh

