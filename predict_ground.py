#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""


# If this is the main scripts, we need gnd_net within our path as a module
# This is to make all imports compatible with both ros and ros free environments
if __name__ == '__main__':
    import sys
    sys.path.insert(1, "..") # Put the parent right behind the current directory (puts it before all ROS packages)

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
from gnd_net.dataset_utils.gnd_data_generator.dataset_augmentation import DataAugmentation, AugmentationConfig
from gnd_net.utils.utils import segment_cloud
from gnd_net.utils.point_cloud_ops import points_to_voxel

from numba import jit



use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used





#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()

parser.add_argument('--resume', default='trained_models/2024_02_18_model_best.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', default='config/config_camera.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true', help='visualize model on validation set')
parser.add_argument('-gnd', '--visualize_gnd', dest='visualize_gnd', action='store_true', help='visualize ground elevation')
parser.add_argument('--gnd_truth', default='', type=str, metavar='PATH', help='visualize ground truth elevation')
parser.add_argument('-l', '--labels', default='', dest='visualize_labels', type=str, metavar='PATH', help='visualize labels')
parser.add_argument('-a', '--augment', dest='augment', action='store_true', help='Augment point cloud')
parser.add_argument('--pcl', default="/home/anshul/es3cap/semkitti_gndnet/kitti_semantic/dataset/sequences/07/", 
                        type=str, metavar='PATH', help='path to config file (default: none)')
args = parser.parse_args()


if os.path.isfile(args.config):
    print("using config file:", args.config)
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use

else:
    print("=> no config file found at '{}'".format(args.config))

print("setting batch_size to 1")
cfg.batch_size = 1


if args.visualize:

    # Ros Includes
    import sys
    import rclpy
    from utils.ros_utils import np2ros_pub_2, gnd_marker_pub
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import Marker

    rclpy.init(args=sys.argv)
    node = rclpy.create_node('gnd_data_provider')
    pcl_pub = node.create_publisher(PointCloud2, "/kitti/velo/pointcloud", 10)
    labels_pub = node.create_publisher(PointCloud2, "/kitti/velo/labels", 10)
    marker_pub_2 = node.create_publisher(Marker, "/kitti/gnd_marker_pred", 10)
    marker_pub_gnd_truth = node.create_publisher(Marker, "/kitti/gnd_marker_gnd_truth", 10)

model = GroundEstimatorNet(cfg).cpu()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)



@jit(nopython=True)
def _shift_cloud(cloud, height):
    cloud += np.array([0,0,height,0], dtype=np.float32)
    return cloud


def InferGround(cloud):

    cloud = _shift_cloud(cloud[:,:4], cfg.lidar_height)

    voxels, coors, num_points = points_to_voxel(cloud, cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
    voxels = torch.from_numpy(voxels).float().cpu()
    coors = torch.from_numpy(coors)
    coors = F.pad(coors, (1,0), 'constant', 0).float().cpu()
    num_points = torch.from_numpy(num_points).float().cpu()
    with torch.no_grad():
            output = model(voxels, coors, num_points)
    return output


def predict_ground(pcl_file: str):
    #point_cloud = open3d.io.read_point_cloud(pcl_file)
    if pcl_file.endswith('npy'):
        points = np.load(pcl_file)[:,:cfg.input_features]
        points[:,2] -= cfg.lidar_height
        if points.shape[1] < cfg.input_features:
            print(f'Add fake data: {(points.shape[0],cfg.input_features-points.shape[1])}')
            points = np.concatenate((points, np.ones((points.shape[0],cfg.input_features-points.shape[1]))), axis=1)
        print(points.shape)
        #points[:,2] += .1
    else:
        points = np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)[:,:cfg.input_features]
    #points = np.c_[points[:,2]*-1, points[:,0], (points[:,1]-5)/3, np.zeros(points.shape[0])]

    # points = points[points[:,2] < 3.5]

    print(np.amin(points, axis=0))
    print(np.amax(points, axis=0))

    if args.gnd_truth != '':
        ground_truth = np.load(args.gnd_truth)
    else:
        ground_truth = np.zeros((50,50))
    if args.augment:
        aug_config = AugmentationConfig(
            grid=cfg.grid_range,
            keep_original=cfg.keep_original,
            num_rotations=cfg.num_rotations,
            num_height_var=cfg.num_height_var,
            maxFrontSlope=cfg.maxFrontSlope,
            maxSideTild=cfg.maxSideTild,
            maxRotation=cfg.maxRotation,
            maxHeight=cfg.maxHeight,
        )

        augmentation = DataAugmentation(config=aug_config)
        
        points_, ground_truth_ = augmentation.getAugmentedData(points.reshape((1,)+points.shape), ground_truth.reshape((1,)+ground_truth.shape))
        points = points_[0]
        ground_truth = ground_truth_[0]

    pred_gnd = InferGround(points)
    pred_gnd = pred_gnd.cpu().numpy()
    # TODO: Remove the points which are very below the ground
    pred_GndSeg = segment_cloud(points.copy(),np.asarray(cfg.grid_range), cfg.voxel_size[0], elevation_map = pred_gnd.T, threshold = 0.08)
    
    if args.visualize:
        np2ros_pub_2(node, points, pcl_pub, None, pred_GndSeg)
        if args.visualize_gnd:
            histogram = np.histogram(pred_gnd, bins=10)
            print(histogram)
            gnd_marker_pub(node, pred_gnd, marker_pub_2, cfg, color = "red")
        if args.gnd_truth != '':
            #assert(os.path.exists(args.visualize_gnd_truth), "The ground truth file does not exist!")
            #ground_truth = np.load(args.gnd_truth)
            gnd_marker_pub(node, ground_truth, marker_pub_gnd_truth, cfg, color = "green")
        if args.visualize_labels != '':
            #assert(os.path.exists(args.visualize_gnd_truth), "The ground truth file does not exist!")
            labels = np.load(args.visualize_gnd_truth, dtype=np.int16).reshape(-1, 2)[:, 0]
            np2ros_pub_2(node, points, labels_pub, None, labels / np.max(labels))
        # pdb.set_trace()


def main():
    global args
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(-1)
    else:
        raise Exception('please specify checkpoint to load')

    predict_ground(args.pcl)


if __name__ == '__main__':
    main()
