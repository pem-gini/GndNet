#!/usr/bin/env python

import sys
sys.path.insert(1, '..') # This way all modules can import a package abolute with gnd_net.SUBMODULE.SUBSUBMODULE

from dataset_utils.gnd_data_generator.dataset_augmentation import AugmentationConfig, DataAugmentation

import argparse
import os
import shutil
import yaml
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# from modules import gnd_est_Loss
from gnd_net.model import GroundEstimatorNet
from gnd_net.utils.utils import segment_cloud
from gnd_net.utils.point_cloud_ops import points_to_voxel
import gnd_net.dataset_utils.gnd_data_generator.dataset_generator_utils as gnd_utils

from numba import jit



use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used


#############################################         DEMO PARAMETERS         #######################################

interval = 2 # Interval in which to regenerate a random augmentation
cut_camera_fov = True

load_from_config = True # If you want to see the augmentation used of a specific config file

# Otherwise you can set custom parameters here
if not load_from_config:
    rotate_cloud = True
    rotate_range = [5, 5, 180] # Max Front slope, Max Side tilt, Max rotation (in degrees)

    shift_height = True
    height_range = .5 # +/- augmentation height in meters

    noise_augmentation = True
    noise_coefficient_top = (0,0)
    noise_coefficient_bottom = (0,0.6)
    noise_min_distance = (0, 5)
    noise_density_top = (.1,50)
    noise_density_bottom=(.1,50)

#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config/config_camera.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
parser.add_argument('--gnd_truth', default='data/training/sequences/00/labels/000000.label', type=str, metavar='PATH', help='visualize ground truth elevation')
parser.add_argument('--pcl', default="data/training/sequences/00/velodyne/000000.bin", 
                        type=str, metavar='PATH', help='path to config file (default: none)')
args = parser.parse_args()


# Ros Includes
import sys
import rclpy
from utils.ros_utils import np2ros_pub_2, gnd_marker_pub
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

if os.path.isfile(args.config):
    print("using config file:", args.config)
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use

    if load_from_config:
        rotate_cloud = cfg.num_rotations > 0
        rotate_range = [cfg.maxFrontSlope, cfg.maxSideTild, cfg.maxRotation] # Max Front slope, Max Side tilt, Max rotation (in degrees)

        shift_height = cfg.num_height_var > 0
        height_range = cfg.maxHeight # +/- augmentation height in meters

        noise_augmentation = cfg.num_noise_var > 0
        noise_coefficient_top = cfg.noise_coefficient_top
        noise_coefficient_bottom = cfg.noise_coefficient_bottom
        noise_min_distance = cfg.noise_min_distance
        noise_density_top = cfg.noise_density_top
        noise_density_bottom=cfg.noise_density_bottom

else:
    print("=> no config file found at '{}'".format(args.config))
    exit(-1)

rclpy.init(args=sys.argv)
node = rclpy.create_node('gnd_data_provider')
pcl_pub = node.create_publisher(PointCloud2, "/kitti/velo/pointcloud", 10)
marker_pub_gnd_truth = node.create_publisher(Marker, "/kitti/gnd_marker_gnd_truth", 10)



def predict_ground(pcl_file: str, labels_file: str):
    # Load data point cloud
    if pcl_file.endswith('npy'):
        points = np.load(pcl_file)[:,:3]
    else:
        points = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)[:,:3]
    points[:,2] += cfg.lidar_height

    # Load data point cloud
    if labels_file.endswith('.label'):
        labels = np.fromfile(labels_file, dtype=np.uint32).reshape((-1,1))
        if labels.shape[0] == points.shape[0]:
            labels = labels & 0xFFFF  # semantic label in lower half
            points = np.concatenate((points,labels), axis = 1)
        else:
            raise Exception(f"The number of points and labels do not match! {labels.shape[0]} vs {points.shape[0]}")
    else:
        raise Exception("Labbels must be a .label file!")
        
    aug_config = AugmentationConfig(
        grid=cfg.grid_range,
        voxel_size=cfg.voxel_size[0],
        keep_original=False,
        num_rotations=1 if rotate_cloud else 0,
        num_height_var=1 if shift_height else 0,
        num_noise_aug=1 if noise_augmentation else 0,
        maxFrontSlope=rotate_range[0],
        maxSideTild=rotate_range[1],
        maxRotation=rotate_range[2],
        maxHeight=height_range,
        noise_coefficient_top = noise_coefficient_top, 
        noise_coefficient_bottom = noise_coefficient_bottom, 
        noise_min_distance = noise_min_distance, 
        noise_density_top = noise_density_top, 
        noise_density_bottom=noise_density_bottom
    )

    augmentation = DataAugmentation(config=aug_config)
    
    while True:
        augmented_points = augmentation.getAugmentedData(points.copy().reshape((1,)+points.shape))
        
        # start = time.time()
        #augmented_points, _ = augmentation.getCameraFOV(augmented_points, np.zeros(augmented_points.shape[0]))
        # print(time.time()-start)

        _, gnd_plane = gnd_utils.compute_ground_plane(
            cloud=augmented_points[0], 
            grid_size=aug_config.grid, 
            voxel_size=aug_config.voxel_size,
            lidar_height=0, # We already moved the entire cloud to zero
        )

        augmented_points = augmentation.addNoise(augmented_points[0], gnd_plane)

        augmented_points = gnd_utils.extract_pc_in_box2d(augmented_points, cfg.grid_range)

        # random sample point cloud to specified number of points
        augmented_points = gnd_utils.random_sample_numpy(augmented_points, N = 10000)


        # pred_GndSeg = segment_cloud(points_.copy(),np.asarray(cfg.grid_range), cfg.voxel_size[0], elevation_map = ground_truth_.T, threshold = 0.08)
        
        np2ros_pub_2(node, augmented_points, pcl_pub, None, np.ones(augmented_points.shape[0]))
        gnd_marker_pub(node, gnd_plane.T, marker_pub_gnd_truth, cfg, color = (175,175,175))

        time.sleep(interval)


def main():
    print('Make sure to start rviz with the rviz_augmentation_demo.rviz config!')
    global args
    predict_ground(args.pcl, args.gnd_truth)


if __name__ == '__main__':
    main()
