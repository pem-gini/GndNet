# Init cluster modules
./setup_cluster.sh

# Install python packages
pip3 install --upgrade pip
pip3 install --user numpy scipy PyYAML
pip3 install --user numba
pip3 install --user torch torchvision torchaudio