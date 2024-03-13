#!/usr/bin/env python

import os

import torch
import torch.nn.functional as F
import numpy as np

# from modules import gnd_est_Loss
from .model import GroundEstimatorNet
from .utils.point_cloud_ops import points_to_voxel
from .utils.utils import cloud_msg_to_numpy, segment_cloud
from .utils.ros_utils import np2ros_pub_2, gnd_marker_pub, np2ros_pub_2_no_intesity
# import ipdb as pdb

# Ros Includes
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterType, ParameterDescriptor
from ament_index_python.packages import get_package_share_directory

from types import SimpleNamespace
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

class GndNetNode(Node):
    def __init__(self):
        super().__init__('gnd_net')
        self.log("Initializing GndNet Node")

        self.declare_parameter("debug", True)
        self.declare_parameter("cuda_enabled", True)
        self.declare_parameter("model_path", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("target_frame", 'map', ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_point_cloud", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_ground_plane", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_segmented_point_cloud", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("topic_pcl_no_ground", None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING))
        self.declare_parameter("camera_height", 0.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("shift_cloud", True)

        # Gnd Net Model Parameter
        self.declare_parameter("num_points", 100000, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter("grid_range", [-10, -10, 10, 10] , ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter("pc_range", [-10,-10,-4,10,10,4] , ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))# cmcdot grid origin is at base_link not the velodyne so have to shift cropping points
        self.declare_parameter("voxel_size", [.4, .4, 8.0], ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        self.declare_parameter("max_points_voxel", 100, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("max_voxels", 2500, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("input_features", 3, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter("use_norm", False, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("vfe_filters", [64] , ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER_ARRAY))# only one filter for now
        self.declare_parameter("with_distance", False, ParameterDescriptor(type=ParameterType.PARAMETER_BOOL))
        self.cfg = {
            "shift_cloud": self.get_parameter("shift_cloud").value,
            "camera_height": self.get_parameter("camera_height").value,
            "num_points": self.get_parameter("num_points").value,
            "grid_range": self.get_parameter("grid_range").value,
            "pc_range": self.get_parameter("pc_range").value,
            "voxel_size": self.get_parameter("voxel_size").value,
            "max_points_voxel": self.get_parameter("max_points_voxel").value,
            "max_voxels": self.get_parameter("max_voxels").value,
            "input_features": self.get_parameter("input_features").value,
            "use_norm": self.get_parameter("use_norm").value,
            "vfe_filters": self.get_parameter("vfe_filters").value,
            "with_distance": self.get_parameter("with_distance").value,
        }
        self.cfg = SimpleNamespace(**self.cfg) # Make it compatible with the GndNet config format (support dot notation)

        # Read in all parameters
        self.debug: bool = self.get_parameter("debug").value
        self.cudaEnabled: bool = self.get_parameter("cuda_enabled").value
        self.targetFrame: str = self.get_parameter("target_frame").value
        self.modelPath: str = self.get_parameter("model_path").value
        self.topicPointCloud: str = self.get_parameter("topic_point_cloud").value
        self.topicGroundPlane: str = self.get_parameter("topic_ground_plane").value
        self.topicSegmentedPointcloud: str = self.get_parameter("topic_segmented_point_cloud").value
        self.topicPclNoGround: str = self.get_parameter("topic_pcl_no_ground").value
        self.shiftCloud: bool = self.get_parameter("shift_cloud").value
        self.cameraHeight: float = self.get_parameter("camera_height").value

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
        # start_time = time.time()
        # cloud = process_cloud(cloud_msg, cfg, shift_cloud = True, sample_cloud = False)
        cloud = cloud_msg_to_numpy(cloud_msg, self.cameraHeight, shift_cloud = self.shiftCloud)

        # np_conversion = time.time()
        # print("np_conversion_time: ", np_conversion- start_time)

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

        with torch.no_grad():
                output = self.model(voxels, coors, num_points)
                # model_time = time.time()
                # print("model_time: ", model_time - cloud_process)

        pred_GndSeg = segment_cloud(cloud.copy(),np.asarray(self.cfg.grid_range), self.cfg.voxel_size[0], elevation_map = output.cpu().numpy().T, threshold = 0.08)
        # seg_time = time.time()
        # print("seg_time: ", seg_time - model_time )
        # print("total_time: ", seg_time - np_conversion)
        # print()
        # pdb.set_trace()
        gnd_marker_pub(self, output.cpu().numpy(), self.pubGroundPlane, self.cfg, color = "red", frame_id=self.targetFrame)
        np2ros_pub_2(self, cloud, self.pubSegmentedPointcloud, None, pred_GndSeg, self.targetFrame)
        np2ros_pub_2_no_intesity(self, cloud, self.pubPclNoGround, self.targetFrame)
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
