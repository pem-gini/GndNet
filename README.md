# GndNet: Fast Ground plane Estimation and Point Cloud Segmentation for Autonomous Vehicles.
This is the original GndNet Readme. For all current informations see the part [below](#gndnet-gini-version).


Authors: Anshul Paigwar, Ozgur Erkent, David Sierra Gonzalez, Christian Laugier

<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/GndNet_Teaser.png" alt="drawing" width="400"/>

## Introduction

This repository is code release for our GndNet paper accepted in International conference on Robotic Systems, IROS 2020. [Link](https://hal.inria.fr/hal-02927350/document)

## Abstract

Ground plane estimation and ground point seg-mentation is a crucial precursor for many applications in robotics and intelligent vehicles like navigable space detection and occupancy grid generation, 3D object detection, point cloud matching for localization and registration for mapping. In this paper, we present GndNet, a novel end-to-end approach that estimates the ground plane elevation information in a grid-based representation and segments the ground points simultaneously in real-time. GndNet uses PointNet and Pillar Feature Encoding network to extract features and regresses ground height for each cell of the grid. We augment the SemanticKITTI dataset to train our network. We demonstrate qualitative and quantitative evaluation of our results for ground elevation estimation and semantic segmentation of point cloud. GndNet establishes a new state-of-the-art, achieves a run-time of 55Hz for ground plane estimation and ground point segmentation. 
<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/GndNet_architecture_final.png" alt="drawing" width="800"/>

## Installation

We have tested the algorithm on the system with Ubuntu 18.04, 12 GB RAM and NVIDIA GTX-1080.

### Dependencies
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
### Visualization
For visualisation of the ground estimation, semantic segmentation of pointcloud, and easy integration with our real system we use Robot Operating System (ROS):
```
ROS
ros_numpy
```
## Data Preparation

We train our model using the augmented [SematicKITTI](http://www.semantic-kitti.org/) dataset. A sample data is provided in this repository, while the full dataset can be downloaded from [link](https://archive.org/details/semantickitti-gndnet-data). We use the following procedure to generate our dataset:
* We first crop the point cloud within the range of (x, y) = [(-50, -50), (50, 50)] and apply incremental rotation [-10, 10] degrees about the X and Y axis to generate data with varying slopes and uphills. (SemanticKITTI dataset is recorded with mostly flat terrain)
* Augmented point cloud is stored as a NumPy file in the folder *reduced_velo*.
* To generate ground elevation labels we then use the CRF-based surface fitting method as described in [1].
* We subdivide object classes in SematicKITTI dataset into two categories 
	1. Ground (road, sidewalk, parking, other-ground, vegetation, terrain)
	2. Non-ground (all other)
* We filter out non-ground points from *reduced_velo* and use CRF-method [1] only with the ground points to generate an elevation map.
* Our ground elevation is represented as a 2D grid with cell resolution 1m x 1m and of size (x, y) = [(-50, -50), (50, 50)], where values of each cell represent the local ground elevation.
* Ground elevation map is stored as NumPy file in gnd_labels folder.
* Finally, GndNet uses gnd_labels and reduced_velo (consisting of both ground and non-ground points) for training.

If you find the dataset useful consider citing our work and for queries regarding the dataset please contact the authors. 

## Training

To train the model update the data directory path in the config file: config_kittiSem.yaml
```
python main.py -s
```
 It takes around 6 hours for the network to converge and model parameters would be stored in checkpoint.pth.tar file. A pre-trained model is provided in the trained_models folder it can be used to evaluate a sequence in the SemanticKITTI dataset.

```
python evaluate_SemanticKITTI.py --resume checkpoint.pth.tar --data_dir /home/.../kitti_semantic/dataset/sequences/07/
```

## Using pre-trained model
Download the SemanticKITTI dataset from their website [link](http://www.semantic-kitti.org/). To visualize the output we use ROS and rviz. The predicted class (ground or non-ground) of the points in the point cloud is substituted in the intensity field of sensor_msgs.pointcloud. In the rviz use intensity as a color transformer to visualize segmented pointcloud. For the visualization of ground elevation, we use the ROS line marker. 

```
roscore
rviz
python evaluate_SemanticKITTI.py --resume trained_models/checkpoint.pth.tar -v -gnd --data_dir /home/.../SemanticKITTI/dataset/sequences/00/
```
Note: The current version of the code for visualization is written in python which can be very slow specifically the generation of ROS marker.
To only visualize segmentation output without ground elevation remove the `-gnd` flag.

## Results

Semantic segmentation of point cloud ground (green) and non-ground (purple):

<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/segmntation_results.png" alt="drawing" width="800"/>

Ground elevation estimation:

<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/ground_estimation.png" alt="drawing" width="800"/>

**YouTube video (Segmentation):**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/W_jXU-ewJR0/0.jpg)](https://www.youtube.com/watch?v=W_jXU-ewJR0&t) 

**YouTube video (Ground Estimation):**

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/kjZ-n_aIJAg/0.jpg)](https://www.youtube.com/watch?v=kjZ-n_aIJAg)

## TODO
* Current dataloader loads the entire dataset into RAM first, this reduces training time but it can be hog systems with low RAM.
* Speed up visualization of ground elevation. Write C++ code for ROS marker.
* Create generalized ground elevation dataset to be with correspondence to SemanticKitti to be made public.


## Citation

If you find this project useful in your research, please consider citing our work:
```
@inproceedings{paigwar2020gndnet,
  title={GndNet: Fast Ground Plane Estimation and Point Cloud Segmentation for Autonomous Vehicles},
  author={Paigwar, Anshul and Erkent, {\"O}zg{\"u}r and Gonz{\'a}lez, David Sierra and Laugier, Christian},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```

## Contribution

We welcome you for contributing to this repo, and feel free to contact us for any potential bugs and issues.


## References

[1] L. Rummelhard, A. Paigwar, A. Nègre and C. Laugier, "Ground estimation and point cloud segmentation using SpatioTemporal Conditional Random Field," 2017 IEEE Intelligent Vehicles Symposium (IV), Los Angeles, CA, 2017, pp. 1105-1110, doi: 10.1109/IVS.2017.7995861.

[2] Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., & Gall, J. (2019). SemanticKITTI: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE International Conference on Computer Vision (pp. 9297-9307).

# GndNET Gini Version
For the gini we adapted the original GndNet slightly to meet all criteria and have higher accuracies using a stereo camera instead of Lidar data.

## Custom training
### Data preperation (Custom Version)
For the training of the network we use a new ground truth computation compared to the original version. This is mainly to get rid of the falsly classified objects close to the ground, that existed mainly because of too high tolerances. In the original version the ground was elevated about half a meter over the actual ground and also smoothing was too big for an accurate prediction on everything beneath 0.5 to 1 m.

The new algorithm uses: TODO...

# Training Algorithm
While the model itself was untouched so far, the training algorithm changed slightly the in sense of data loading. We added a ring buffer to dynamically load big datasets on the go without the requirement to have everything in memory at once. This significantly reduced hardware requirements while reducing performance only about 1%. 

In the current configuration the training data will use the ring buffer with a buffer size of about 4 GB, while the validation data is loaded completely into memory (~1GB). Buffer sizes can be changed in the main file at the data loading parameters, and switching out the async to sync or the other way around can easliy be done in the [`dataset_utils/dataset_provider.py`](dataset_utils/dataset_provider.py#L239) file.

For there to be little to no reduction in performance, the training script will create a seperate process that only manages the ring buffer and offering everthing in a shared memory for the main process to work with.

# Training Process
To train the model, the [`main.py`](main.py) file needs to be executed using the proper config file. This requres all packages listed in the [`setup.sh`](setup.sh) script to be installed and an active Cuda environment with an available GPU. 

If training on a local machine, you're all set and can follow the general training procedure.

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