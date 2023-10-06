#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""

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
from model import GroundEstimatorNet
from modules.loss_func import MaskedHuberLoss
from dataset_utils.dataset_provider import get_train_loader, get_valid_loader
from utils.utils import lidar_to_img, lidar_to_heightmap, segment_cloud
from utils.point_cloud_ops import points_to_voxel
import ipdb as pdb
import matplotlib.pyplot as plt

import numba
from numba import jit,types
import open3d



use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used





#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', default='config/config_kittiSem.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true', help='visualize model on validation set')
parser.add_argument('-gnd', '--visualize_gnd', dest='visualize_gnd', action='store_true', help='visualize ground elevation')
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
    marker_pub_2 = node.create_publisher(Marker, "/kitti/gnd_marker_pred", 10)

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


def predict_ground(pcl_file):
    point_cloud = open3d.io.read_point_cloud(pcl_file)
    points = np.asarray(point_cloud.points) # np.fromfile(pcl_file, dtype=np.float32).reshape(-1, 4)
    points = np.c_[points[:,2]*-1, points[:,0], (points[:,1]-5)/3, np.zeros(points.shape[0])]

    # points = points[points[:,2] < 3.5]

    print(np.amin(points, axis=0))
    print(np.amax(points, axis=0))

    pred_gnd = InferGround(points)
    pred_gnd = pred_gnd.cpu().numpy()
    # TODO: Remove the points which are very below the ground
    pred_GndSeg = segment_cloud(points.copy(),np.asarray(cfg.grid_range), cfg.voxel_size[0], elevation_map = pred_gnd.T, threshold = 0.0)
    
    if args.visualize:
        np2ros_pub_2(node, points, pcl_pub, None, pred_GndSeg)
        if args.visualize_gnd:
            gnd_marker_pub(node, pred_gnd, marker_pub_2, cfg, color = "red")
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
    else:
        raise Exception('please specify checkpoint to load')

    predict_ground(args.pcl)



if __name__ == '__main__':
    main()
