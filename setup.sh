# Init cluster modules
./setup_cluster.sh

# Install python packages
pip3 install --upgrade pip
pip3 install --user numpy scipy PyYAML ros2_numpy
pip3 install --user numba
pip3 install --user torch torchvision torchaudio
pip3 install --user shapely

# Install ROS packages
sudo apt install ros-humble-tf-transformations
pip3 install --user transforms3d
