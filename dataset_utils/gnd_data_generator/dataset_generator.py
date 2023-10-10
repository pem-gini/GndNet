import glob
import os
import sys
sys.path.append("../..") # Adds higher directory to python modules path.

import yaml
import ipdb as pdb
import time
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from message_filters import TimeSynchronizer, Subscriber,ApproximateTimeSynchronizer
import message_filters
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import ros2_numpy.ros2_numpy as ros2_numpy
import numba


import torch
from torch.utils.data.sampler import SubsetRandomSampler
from dataset_generator_utils import random_sample_numpy, extract_pc_in_box2d, shift_cloud_func



plt.ion()

# with open('config/config.yaml') as f:
with open('../../config/config_kittiSem.yaml') as f:
	config_dict = yaml.load(f, Loader=yaml.FullLoader)

class ConfigClass:
	def __init__(self, **entries):
		self.__dict__.update(entries)

cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use



# resolution of ground estimator grid is 1m x 1m 
visualize = False


pc_range = cfg.pc_range[:2] + cfg.pc_range[3:5] # select minmax xy
grid_size = cfg.grid_range
length = int(grid_size[2] - grid_size[0]) # x direction
width = int(grid_size[3] - grid_size[1])	# y direction
out_dir = '/home/finn/pem/ducktrain/GndNet/data/training/000'

# @jit(nopython=True)
def process_cloud(cloud_msg):
	# Convert Ros pointcloud2 msg to numpy array
	pc = ros2_numpy.numpify(cloud_msg)
	points=np.zeros((pc.shape[0],4))
	points[:,0]=pc['x']
	points[:,1]=pc['y']
	points[:,2]=pc['z']
	points[:,3]=pc['intensity']
	cloud = np.array(points, dtype=np.float32)
	# cloud = cloud[:, :3]  # exclude luminance
	# pdb.set_trace()

	cloud = extract_pc_in_box2d(cloud, pc_range)

	# random sample point cloud to specified number of points
	cloud = random_sample_numpy(cloud, N = cfg.num_points)

	if cfg.shift_cloud:
		cloud  = shift_cloud_func(cloud, cfg.lidar_height)

	# print(cloud.shape)
	return cloud


def process_label(ground_msg):
	label = np.zeros((width, length))
	for pt in ground_msg.points:
		# print(pt.y, pt.x)
		# print(int(pt.y - grid_size[1]), int(pt.x - grid_size[0]))
		label[ int(pt.y - grid_size[1]), int(pt.x - grid_size[0])] = pt.z
	return label


# @jit(nopython=True)
def recorder(cloud, gnd_label,num):
	velo_path = out_dir + "/reduced_velo/" + "%06d" % num
	label_path = out_dir + "/gnd_labels/" + "%06d" % num
	np.save(velo_path,cloud)
	np.save(label_path,gnd_label)




class Listener(Node):
	def __init__(self):
		super().__init__('ground_estimation_dataset')

		if not os.path.isdir(out_dir + "/reduced_velo/"):
			os.mkdir(out_dir + "/reduced_velo/")
		if not os.path.isdir(out_dir + "/gnd_labels/"):
			os.mkdir(out_dir + "/gnd_labels/")

		# self.point_cloud_sub = message_filters.Subscriber("/kitti/classified_cloud", PointCloud2)
		self.point_cloud_sub = message_filters.Subscriber(self, PointCloud2, "/kitti/raw/pointcloud")
		self.ground_marker_sub = message_filters.Subscriber(self, Marker, '/kitti/ground_marker')

		ts = ApproximateTimeSynchronizer([self.point_cloud_sub, self.ground_marker_sub],10, 0.1, allow_headerless=True)
		ts.registerCallback(self.callback)
		self.hf = plt.figure()
		self.count = 0

	def callback(self,cloud_msg, ground_msg):
		start_time = time.time()
		gnd_label = process_label(ground_msg)
		# label_time = time.time()

		#@imp: We are subscribing classified cloud which is wrt base_link so no need to shift the point cloud.in z direction
		cloud = process_cloud(cloud_msg)
		# cloud_time = time.time()

		# print("label_process: ", label_time- start_time)
		# print("cloud_process: ", cloud_time- label_time)

		recorder(cloud, gnd_label, self.count)
		end_time = time.time()
		print("total_process: ", end_time- start_time, self.count)
		self.count += 1


		if visualize:

			self.hf.clear()
			cs = plt.imshow(gnd_label.T, interpolation='nearest')
			cbar = self.hf.colorbar(cs)


			self.ha = self.hf.add_subplot(111, projection='3d')
			self.ha.set_xlabel('$X$', fontsize=20)
			self.ha.set_ylabel('$Y$')
			X = np.arange(0, length, 1)
			Y = np.arange(0, width, 1)
			X, Y = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
			# R = np.sqrt(X**2 + Y**2)
			self.ha.plot_surface(Y, X, gnd_label)
			self.ha.set_zlim(-10, 10)
			plt.draw()
			plt.pause(0.01)
			self.hf.clf()


if __name__ == '__main__':
	rclpy.init(args=sys.argv)
	obj = Listener()
	rclpy.spin(obj)
	rclpy.shutdown()
	# plt.show(block=True)



	# cloud += np.array([0,0,lidar_height], dtype=np.float32) # shift the pointcloud as you want it wrt base_link
	# print(cloud.shape)
	# cloud = np.array([p for p in cloud if p[0] > abs(p[1])])
	# cloud = np.array([p for p in cloud if p[0] > 0])

	# # Use PCL to clip the pointcloud
	# rregion = (grid_size[0],grid_size[1],grid_size[2],0,grid_size[3],grid_size[4],grid_size[5],0)
	# #(xmin, ymin, zmin, smin, xmax, ymax, zmax, smax)
	# pcl_cloud = pcl.PointCloud()
	# pcl_cloud.from_array(cloud)
	# clipper = pcl_cloud.make_cropbox()
	# clipper.set_MinMax(*region)
	# out_cloud = clipper.filter()

	# # if(out_cloud.size > 15000):
	# leaf_size = 0.05
	# vox = out_cloud.make_voxel_grid_filter()
	# vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
	# out_cloud = vox.filter()

