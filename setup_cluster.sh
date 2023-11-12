module load GCCcore/.10.3.0
module load Python/3.9.5
module load CUDA/12.0.0

export CONDA_ROOT=$HOME/miniconda3
$CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gndnet