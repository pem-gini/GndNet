# GndNet: Fast Ground plane Estimation and Point Cloud Segmentation for Autonomous Vehicles.
This is the Readme of our adjusted version of GndNet. All general information about the original project can be found [here](https://github.com/anshulpaigwar/GndNet).

Authors adjustments: Finn Gegenmantel    
Authors original papers: Anshul Paigwar, Ozgur Erkent, David Sierra Gonzalez, Christian Laugier

<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/GndNet_Teaser.png" alt="drawing" width="400"/>

## Introduction

This repository is based on the code release for the GndNet paper accepted in International conference on Robotic Systems, IROS 2020. [Link](https://hal.inria.fr/hal-02927350/document)

## Abstract

Ground plane estimation and ground point seg-mentation is a crucial precursor for many applications in robotics and intelligent vehicles like navigable space detection and occupancy grid generation, 3D object detection, point cloud matching for localization and registration for mapping. In this paper, we present GndNet, a novel end-to-end approach that estimates the ground plane elevation information in a grid-based representation and segments the ground points simultaneously in real-time. GndNet uses PointNet and Pillar Feature Encoding network to extract features and regresses ground height for each cell of the grid. We augment the SemanticKITTI dataset to train our network. We demonstrate qualitative and quantitative evaluation of our results for ground elevation estimation and semantic segmentation of point cloud. GndNet establishes a new state-of-the-art, achieves a run-time of 55Hz for ground plane estimation and ground point segmentation. 
<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/GndNet_architecture_final.png" alt="drawing" width="800"/>

## Installation

### Dependencies
`ros_numpy` is added as a submodule because of a persiting bug in the currently published version. Therefore you need to clone the submodules first:
```
git submodule update --init --recursive
```

```
Python 3.6
CUDA (tested on 10.1)
PyTorch (tested on 1.4)
scipy
ipdb
argparse
numba
ros2_numpy (if using ros)
```

All dependecies are also part of the `setup.sh` script. It can simply be run install all required packages, if you are not currently on the RWTH Cluster, comment out the first line to ignore the cluster setup script.

It is recommended using an anaconda environment to avoid version conflicts, see the `setup_conda.sh` script for details.

### Visualization
For visualisation of the ground estimation, semantic segmentation of pointcloud, and easy integration with our real system we use Robot Operating System 2 (ROS2):
```
ROS
ros_numpy
```

Because of the bug in the ros2 version of the ros_numpy package, a fork is included as a git submodule into this project. Make sure it is cloned properly. 

## Data Preperation
There is a [detailed description](dataset_utils/gnd_data_generator/README.md) of the new ground plane computation algorithm and parsing of the original dataset in the ground preparation folder.

# Training Algorithm
The training algorithm and the model itself has been untouched from the original paper, therefore I will only reference it [here](https://github.com/anshulpaigwar/GndNet). The general concepts are:
- Grouping the whole point cloud into voxels
- Extracting features for each voxel (distance to cluster center, distance to pillar center, and some random fixed number of points)
- Passing it into the network
- Having a cost function that compares to the previously computed ground truth and also gives credit for smoothness of the plane

# Training Process
To train the model, the [`main.py`](main.py) file needs to be executed using the proper config file. This requires all packages listed in the [`setup.sh`](setup.sh) script to be installed and an active Cuda environment with an available GPU. 

If training on a local machine, you're all set and can simply start the process:
(In the first lines of the file you can set if using ROS, or some visulizations)

```
python3 training.py -s --config=config/config_camera.yaml
```

Make sure the config file has all the required attributes. (The -s is for saving the checkpoints while training)

### Training on the RWTH HPC Cluster
#### Connecting to the Cluster
To connect to the HPC cluster, use one of the login nodes (login18-1.hpc.itc.rwth-aachen.de). Keep in mind that if you want to interactivaly test your script and not just submit a job, you will need to be on a login node with a GPU installed (e.g. login18-g-1.hpc.itc.rwth-aachen.de). For a full and up-to-date list, see [here](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/0a23d513f31b4cf1849986aaed475789/).

Once you decided on a login node, just ssh onto it (provided you [already set up](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/598d0f7f78cb4ab8b81af1b3f68ba831/) your HPC account and are in the RWTH network either live or via VPN):

```bash
ssh -l ab123456 login18-1.hpc.itc.rwth-aachen.de
```
(replace ab123456 with your RWTH user name)

The server will ask for the HPC password set in the RegApp (Not your RWTH Password!) and you will need your 2FA, which should also be set up within the [RegApp](https://regapp.itc.rwth-aachen.de/) (Under Index/My Tokens).

**If your are using VS Code to connect to the cluster, activate the `  remote.SSH.showLoginTerminal
` setting, otherwise you will not be prompted with the 2FA and cannot log in!**

After the login was successfull, follow the rest of the Readme.

#### The Slurm Script
The RWTH High Performance Cluster is using a schedular called `slurm`. It needs a shell script as an entry point, which includes both the configurations and the commands to execute. 

To run the data preparation, there is already a [script](run_data_preparation_cluster.sh) that will execute the [preparation](#data-preperation-custom-version) as described above. 

To run the actual training, there is another script. The [`run_model_training_cluster.sh`](run_model_training_cluster.sh), which will configure the HPC and start the training:

```bash
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
```

The `#SBATCH` lines, are the configurations for the `slurm` scheduler. We are configuring the name of the job, the number of GPUs (1), our output file (%J will use the job id), the max runtime, the amount of memory we need and the number of CPUs.

After the slurm parameters, the actual script is starting. Because the default HPC environment has the wrong python version and some conflicts we don't want, the training will happen within a conda environment. This environment needs to be created first, if not the case yet. Simply execute the [`setup_conda.sh`](setup_conda.sh) script.

Once we are in the correct environment, the `main.py` training process can be started. 

#### Starting the Job
To start the training job, use the slurm command line tool:

```shell
sbatch run_model_training_cluster.sh
```

To check if the job is waiting or running, use:

```shell
squeue -u $USER
```

To cancel:

```shell
scancel JOBID
```

To check your current HPC quota (will only update after midnight):
```shell
r_wlm_usage -q
```

And to have a live dashboard of the current (and past) processes:

[https://perfmon.hpc.itc.rwth-aachen.de/](https://perfmon.hpc.itc.rwth-aachen.de/)

For more slurm commands, see the [official RWTH documentation](https://help.itc.rwth-aachen.de/en/service/rhr4fjjutttf/article/3d20a87835db4569ad9094d91874e2b4/).

## Evaluating Results
Once you have a trained model or are simply using the pre-trained once provided in this repository, you can use it to run predictions on any point clouds you like. Simply execute the `predict_ground.py` script with the correct arguments:

```
python3 predict_ground.py -v -gnd --resume=trained_models/checkpoint_2024_04_20.pth.tar --pcl=data/000000.npy --config=config/config_camera.yaml 
```

Make sure the config file is the same one as used for the training process. Especially the number of input features and the grid dimensions have to match, otherwise the model loading path will fail or the results to not make sense. 

If using one of the two pre-trained models:
- `trained_models.pth.tar` [trained by GndNet original project]: Uses 4-features input features (x,y,z,intesity) -> `config/config_kittiSem.yaml`
- `trained_models_20204_04_20.pth.tar` [trained by us]: Uses 3-features (x,y,z) and a camera extract -> `config/config_camera.yaml`

Make sure rviz is running and listeing to the correct topics:
```
ros2 run rviz2 rviz2 -d config/rviz_predict_ground.rviz
```