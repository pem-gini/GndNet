#!/usr/bin/env python

import os
import subprocess
import time

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

# from modules import gnd_est_Loss
from gnd_net.model import GroundEstimatorNet
from gnd_net.utils.point_cloud_ops import points_to_voxel
from gnd_net.utils.utils import cloud_msg_to_numpy, segment_cloud, split_segmented_cloud
from gnd_net.utils.ros_utils import np2ros_pub_2, gnd_marker_pub, np2ros_pub_2_no_intensity
# import ipdb as pdb

# Ros Includes
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
import tf2_ros
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory

from types import SimpleNamespace
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

import yaml

def get_git_root():
    try:
        git_root = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf-8')
        return git_root
    except subprocess.CalledProcessError:
        return None

def resolveEnv(v):
    if isinstance(v, str):
        if "$HOME" in v:
            v = v.replace("$HOME", os.environ.get('HOME'))
        if "~" in v:
            v = v.replace("~", os.environ.get('HOME'))
        if "$GITDIR" in v:
            v = v.replace("$GITDIR", get_git_root())
    return v

class GndNetNode(Node):
    def __init__(self):
        super().__init__('gnd_net')
        self.log("Initializing GndNet Node")

        self.declare_parameter("debug", True)
        self.declare_parameter("model_path", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("model_config", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("target_frame", 'map', ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_point_cloud", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_ground_plane", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_segmented_point_cloud", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_pcl_no_ground", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))

        # self.declare_parameter("num_points", 100000, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        # self.declare_parameter("grid_range", [-10, -10, 10, 10] , ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        # self.declare_parameter("pc_range", [-10,-10,-4,10,10,4] , ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))# cmcdot grid origin is at base_link not the velodyne so have to shift cropping points
        # self.declare_parameter("voxel_size", [.4, .4, 8.0], ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        # self.declare_parameter("max_points_voxel", 100, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        # self.declare_parameter("max_voxels", 2500, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        # self.declare_parameter("input_features", 3, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        # self.declare_parameter("use_norm", False, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        # self.declare_parameter("vfe_filters", [64] , ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY))# only one filter for now
        # self.declare_parameter("with_distance", False, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        # self.cfg = {
        #     "shift_cloud": self.get_parameter("shift_cloud").value,
        #     "camera_height": self.get_parameter("camera_height").value,
        #     "num_points": self.get_parameter("num_points").value,
        #     "grid_range": self.get_parameter("grid_range").value,
        #     "pc_range": self.get_parameter("pc_range").value,
        #     "voxel_size": self.get_parameter("voxel_size").value,
        #     "max_points_voxel": self.get_parameter("max_points_voxel").value,
        #     "max_voxels": self.get_parameter("max_voxels").value,
        #     "input_features": self.get_parameter("input_features").value,
        #     "use_norm": self.get_parameter("use_norm").value,
        #     "vfe_filters": self.get_parameter("vfe_filters").value,
        #     "with_distance": self.get_parameter("with_distance").value,
        # }
        # self.cfg = SimpleNamespace(**self.cfg) # Make it compatible with the GndNet config format (support dot notation)

        # Read in all parameters
        self.debug: bool = self.get_parameter("debug").value
        self.targetFrame: str = self.get_parameter("target_frame").value
        self.modelPath: str = self.get_parameter("model_path").value
        self.modelConfig: str = self.get_parameter("model_config").value
        self.topicPointCloud: str = self.get_parameter("topic_point_cloud").value
        self.topicGroundPlane: str = self.get_parameter("topic_ground_plane").value
        self.topicSegmentedPointcloud: str = self.get_parameter("topic_segmented_point_cloud").value
        self.topicPclNoGround: str = self.get_parameter("topic_pcl_no_ground").value

        # Gnd Net Model Parameter
        package_share_directory = get_package_share_directory('gnd_net')
        #package_share_directory = pkg_resources.get_distribution('gnd_net').get_metadata('RECORD').split()[0]
        model_config_path = os.path.join(package_share_directory, self.modelConfig)
        with open(model_config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

            for k, v in config_dict.items():
                ### env var replacement
                v = resolveEnv(v)
                ### set attribute
                config_dict[k] = v

        class ConfigClass:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        
        self.cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use
        self.cfg.batch_size = 1 # Always set it to one

        self.cudaEnabled = torch.cuda.is_available()

        if not self.cudaEnabled:
            self.warning('CUDA not available. Using CPU for GndNet prediction!!!')

        # Buffer current coordinate frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create publishers
        self.pubGroundPlane = self.create_publisher(Marker, self.topicGroundPlane, 1)
        self.pubSegmentedPointcloud = self.create_publisher(PointCloud2, self.topicSegmentedPointcloud, 1)
        self.pubPclNoGround = self.create_publisher(PointCloud2, self.topicPclNoGround, 1)

        # Create subscriber
        self.create_subscription(PointCloud2, self.topicPointCloud, self.callback, 1)

        # Load model into memory/GPU
        self.model = self.loadModel()

    def log(self, msg):
        self.get_logger().info(msg)
    def warning(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)

    def loadModel(self):
        if self.cudaEnabled:
            torch.device('cuda')
            try:
                model = GroundEstimatorNet(self.cfg).cuda()
            except:
                self.error('Cannot load model parameters onto GPU using Cuda. Check if Cuda is available!')
        else:
            model = GroundEstimatorNet(self.cfg).cpu()

        package_share_directory = get_package_share_directory('gnd_net')
        #package_share_directory = pkg_resources.get_distribution('gnd_net').get_metadata('RECORD').split()[0]
        checkpoint_path = os.path.join(package_share_directory, self.modelPath)

        if os.path.isfile(checkpoint_path):
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu')) #TODO: Remove cpu location
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))

        return model.eval() # Switch to evaluation mode

    def callback(self, cloud_msg):
        self.log('Got new frame')
        # start_time = time.time()
        # cloud = process_cloud(cloud_msg, cfg, shift_cloud = True, sample_cloud = False)
        cloud = cloud_msg_to_numpy(cloud_msg, 0, shift_cloud = False)

        if self.targetFrame != cloud_msg.header.frame_id:
            try:
                transform = self.tf_buffer.lookup_transform(self.targetFrame, cloud_msg.header.frame_id, self.get_clock().now().to_msg())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.error('Error looking up transform')
                return
        
            # Transform the cloud. Code from tf2_sensor_msgs package: https://github.com/ros2/geometry2/blob/rolling/tf2_sensor_msgs/tf2_sensor_msgs/tf2_sensor_msgs.py
            rotation = R.from_quat(np.array([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]))
            rotation_matrix = R.as_matrix(rotation)
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ])

            cloud = np.einsum('ij, pj -> pi', rotation_matrix, cloud) + translation
        # np_conversion = time.time()
        # print("np_conversion_time: ", np_conversion- start_time)

        # Drop all points with invalid values (e.g. nan)
        #cloud_raw = np.copy(cloud).reshape((180,-1))
        cloud = cloud[~np.isnan(cloud).any(axis=1)]
        if len(cloud) == 0:
            self.warning('Received empty point cloud, ignore')
            return

        self.log('Voxilize point cloud')
        voxels, coors, num_points = points_to_voxel(cloud, self.cfg.voxel_size, self.cfg.pc_range, self.cfg.max_points_voxel, True, self.cfg.max_voxels)
        if self.cudaEnabled:
            voxels = torch.from_numpy(voxels).float().cuda()
            coors = torch.from_numpy(coors)
            coors = F.pad(coors, (1,0), 'constant', 0).float().cuda()
            num_points = torch.from_numpy(num_points).float().cuda()
        else:
            voxels = torch.from_numpy(voxels).float().cpu() 
            coors = torch.from_numpy(coors)
            coors = F.pad(coors, (1,0), 'constant', 0).float().cpu()
            num_points = torch.from_numpy(num_points).float().cpu()

        # cloud_process = time.time()
        # print("cloud_process: ", cloud_process - np_conversion)
        self.log('Run prediction')
        with torch.no_grad():
                output = self.model(voxels, coors, num_points)
                # model_time = time.time()
                # print("model_time: ", model_time - cloud_process)

        self.log('Segment cloud')
        cloud_obs, cloud_gnd, pred_GndSeg = split_segmented_cloud(cloud.copy(),np.asarray(self.cfg.grid_range), self.cfg.voxel_size[0], elevation_map = output.cpu().numpy().T, threshold = 0.16)

        # seg_time = time.time()
        # print("seg_time: ", seg_time - model_time )
        # print("total_time: ", seg_time - np_conversion)
        # print()
        # pdb.set_trace()
        self.log('Publish results')
        gnd_marker_pub(self, output.cpu().numpy(), self.pubGroundPlane, self.cfg, color = "red", frame_id=self.targetFrame)
        np2ros_pub_2(self, cloud, self.pubSegmentedPointcloud, None, pred_GndSeg, self.targetFrame)
        np2ros_pub_2_no_intensity(self, cloud_obs, self.pubPclNoGround, self.targetFrame)
        # vis_time = time.time()
        # print("vis_time: ", vis_time - model_time)


def main(args=None):
    rclpy.init(args=args)
    
    node = GndNetNode()

    #node.create_subscription(PointCloud2, "/kitti/velo/pointcloud", callback, 1)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

