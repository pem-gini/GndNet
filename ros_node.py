#!/usr/bin/env python

import os
import subprocess
import time
import threading
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# from modules import gnd_est_Loss
from gnd_net.model import GroundEstimatorNet
from gnd_net.utils.point_cloud_ops import points_to_voxel
from gnd_net.utils.utils import cloud_msg_to_numpy, segment_cloud, segment_cloud_noground
from gnd_net.utils.ros_utils import np2ros_pub_2, gnd_marker_pub, np2ros_pub_2_no_intensity, array_to_pointcloud2
from gnd_net.utils import transform
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
import std_msgs.msg as stdMsgs
from visualization_msgs.msg import Marker
#############################################################################################
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
#############################################################################################
class InferenceThread(threading.Thread):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.hasModel = model != None
        self.cfg = cfg
        self.daemon = True
        self.output = None
        self.mutex = threading.Lock()
        self.running = True
        self.clearInputs()

    def transformCuda(self, voxels, coors, num_points):
        voxels = torch.from_numpy(voxels).float().cuda()
        coors = torch.from_numpy(coors)
        coors = F.pad(coors, (1,0), 'constant', 0).float().cuda()
        num_points = torch.from_numpy(num_points).float().cuda()
        return voxels, coors, num_points
    def transformCpu(self, voxels, coors, num_points):
        voxels = torch.from_numpy(voxels).float().cpu() 
        coors = torch.from_numpy(coors)
        coors = F.pad(coors, (1,0), 'constant', 0).float().cpu()
        num_points = torch.from_numpy(num_points).float().cpu()
        return voxels, coors, num_points
        
    def dryrun(self, tensorTransformFunc):
        # Define the grid size
        # cloud = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

        # Create a static plane with the height zero
        x_values = np.arange(-10, 10, 0.35)
        y_values = np.arange(-10, 10, 0.35)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = np.zeros_like(x_flat)
        cloud = np.stack((x_flat, y_flat, z_flat), axis=-1).astype(np.float32)

        print(f'inference-thread: start dry run')
        ### do inference stuff
        start = time.time()
        # self.log('Voxilize point cloud')
        voxels, coors, num_points = points_to_voxel(cloud, self.cfg.voxel_size, self.cfg.pc_range, self.cfg.max_points_voxel, True, self.cfg.max_voxels)
        voxels, coors, num_points = tensorTransformFunc(voxels, coors, num_points)
        dt1 = time.time() - start
        if self.hasModel != None:
            try:
                with torch.no_grad():
                    self.output = self.model(voxels, coors, num_points)
            except Exception as e:
                print(e)
            dt2 = time.time() - start
            # print("dt: %s" % (dt))
            #print(f'end dry run: {dt1} | {dt2-dt1}')
            return
        
        #print(f'end dry run without model: {dt1}')

        # Wait until the model is available and rerun the dryrun to also compile all functions within the model
        while not self.hasModel:
            time.sleep(0.05)
        self.dryrun()

    def run(self):

        cudaEnabled = torch.cuda.is_available()
        tensorTransformFunc = lambda v,c,n: self.transformCuda(v,c,n)
        if not cudaEnabled: 
            tensorTransformFunc = lambda v,c,n: self.transformCpu(v,c,n)

        self.dryrun(tensorTransformFunc)
        print('inference-thread: Finished dryrun, start listening for inferences')
        while self.running:
            with self.mutex:
                cloud = self.input_image_buf_1
            if cloud.any():
                #print(f'start inference {cloud.shape}')
                ### do inference stuff
                start = time.time()
                # self.log('Voxilize point cloud')
                voxels, coors, num_points = points_to_voxel(cloud, self.cfg.voxel_size, self.cfg.pc_range, self.cfg.max_points_voxel, True, self.cfg.max_voxels)
                #dt3 = time.time() - start
                voxels, coors, num_points = tensorTransformFunc(voxels, coors, num_points)
                output = None
                #dt1 = time.time() - start
                try:
                    with torch.no_grad():
                        self.output = self.model(voxels, coors, num_points)
                except Exception as e:
                    print(e)
                #dt2 = time.time() - start
                # print("dt: %s" % (dt))
                #print(f'end inference: {dt3} {dt1-dt3} | {dt2-dt1}')
            else:
                ### safety delay for thread to not run amok
                time.sleep(0.001)
    def stop(self):
        self.running = False
    def setModel(self, model):
        print('Set inference model')
        self.hasModel = True
        self.model = model
    def setInputs(self, x):
        with self.mutex:
            self.input_image_buf_1 = x.copy()
    def getOutput(self):
        return self.output
    def clearInputs(self):
        self.input_image_buf_1 = np.empty((0,0))
#############################################################################################
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

        #################################################
        ### inference thread
        self.inference = InferenceThread(None, self.cfg)
        self.inference.start()
        #################################################
        # Load model into memory/GPU
        model = self.loadModel(self.cfg)
        self.inference.setModel(model)
        # Buffer current coordinate frame transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Trigger first compilation of the transformation
        self.transform_dryrun()
        # Create publishers
        self.pubGroundPlane = self.create_publisher(Marker, self.topicGroundPlane, 1)
        self.pubSegmentedPointcloud = self.create_publisher(PointCloud2, self.topicSegmentedPointcloud, 1)
        self.pubPclNoGround = self.create_publisher(PointCloud2, self.topicPclNoGround, 1)
        # Create subscriber
        self.create_subscription(PointCloud2, self.topicPointCloud, self.callback, 1)

    def destroy_node(self):
        self.inference.stop()
        super().destroy_node()    
    def log(self, msg):
        self.get_logger().info(msg)
    def warning(self, msg):
        self.get_logger().warn(msg)
    def error(self, msg):
        self.get_logger().error(msg)
    def loadModel(self, cfg):
        if torch.cuda.is_available():
            self.warning('CUDA available. Using GPU for GndNet prediction!!!')
            torch.device('cuda')
            try:
                model = GroundEstimatorNet(cfg).cuda()
            except:
                self.error('Cannot load model parameters onto GPU using Cuda. Check if Cuda is available!')
        else:
            self.warning('CUDA not available. Using CPU for GndNet prediction!!!')
            model = GroundEstimatorNet(cfg).cpu()
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
    
    def transform_dryrun(self):
        print('Start transformation dryrun')
        start = time.time()
        try:
            # Define a static cloud of a couple of points#
            cloud = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

            # A static translation + rotation quaternion
            ts = tf2_ros.TransformStamped()
            ts.transform.translation.x = 1.0
            ts.transform.translation.y = 1.0
            ts.transform.translation.z = 1.0
            ts.transform.rotation.x = 0.25
            ts.transform.rotation.y = 0.25
            ts.transform.rotation.z = 0.25
            ts.transform.rotation.w = 0.25
            trafo = transform.transformStampedToTransformationMatrix(ts)
            cloud = transform.transformCloud(cloud, trafo)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.error('Error during transform dryrun')
            return
        
        print(f'Finish transformation dryrun {time.time()-start}')

    def callback(self, cloud_msg):
        ### convert cloud to numpy
        cloud = cloud_msg_to_numpy(cloud_msg, 0, shift_cloud = False)["xyz"]
        ### transform cloud to target frame
        if self.targetFrame != cloud_msg.header.frame_id:
            try:
                ts = self.tf_buffer.lookup_transform(self.targetFrame, cloud_msg.header.frame_id, cloud_msg.header.stamp)
                trafo = transform.transformStampedToTransformationMatrix(ts)
                cloud = transform.transformCloud(cloud, trafo)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.error('Error looking up transform')
                return
        # Drop all points with invalid values (e.g. nan)
        cloud = cloud[~np.isnan(cloud).any(axis=1)]
        if len(cloud) == 0:
            self.warning('Received empty point cloud, ignore')
            return
        ### set new inference input
        self.inference.setInputs(cloud)
        ### grab last available output from model (could be old)
        predictedGroundPlaneOutput = self.inference.getOutput()
        if predictedGroundPlaneOutput == None:
            return
        ### segment pointcloud with detected gnd plane
        pred_GndSeg, cloud_obs = segment_cloud_noground(cloud.copy(), cloud.copy(), np.asarray(self.cfg.grid_range), self.cfg.voxel_size[0], elevation_map = predictedGroundPlaneOutput.cpu().numpy().T, threshold = 0.16)
        ### prepare result & publish
        if self.pubGroundPlane.get_subscription_count() > 0:
            gnd_marker_pub(self, predictedGroundPlaneOutput.cpu().numpy(), self.pubGroundPlane, self.cfg, color = (175,175,175), frame_id=self.targetFrame)
        if self.pubSegmentedPointcloud.get_subscription_count() > 0:
            np2ros_pub_2(self, cloud, self.pubSegmentedPointcloud, None, pred_GndSeg, self.targetFrame)
        if self.pubPclNoGround.get_subscription_count() > 0:
            # np2ros_pub_2_no_intensity(self, cloud_obs, self.pubPclNoGround, self.targetFrame)
            header = stdMsgs.Header(stamp=cloud_msg.header.stamp, frame_id=self.targetFrame)
            nogroundcloud = array_to_pointcloud2(cloud_obs, header)
            self.pubPclNoGround.publish(nogroundcloud)

def main(args=None):
    rclpy.init(args=args)
    node = GndNetNode()
    #node.create_subscription(PointCloud2, "/kitti/velo/pointcloud", callback, 1)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

