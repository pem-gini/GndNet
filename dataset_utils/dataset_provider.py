from __future__ import print_function, division
import os
import sys
import math
sys.path.append("..") # Adds higher directory to python modules path.

import time
import multiprocessing
from multiprocessing import shared_memory
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import yaml
import os
import collections
import logging
import random
# import ipdb as pdb

Msg = collections.namedtuple('Msg', ['event', 'args'])

class AsyncDataLoader(multiprocessing.Process):
	def __init__(self, data_dir: str, dir_name: str, max_memory=4e6, num_input_features = 3, skip_frames = 1, parent_logger=logging.root, *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.data_dir = data_dir
		self.dir_name = dir_name
		self.skip_frames = skip_frames
		self.num_input_features = num_input_features
		self.currentId = -1
		self.logger = parent_logger.getChild('async_data_loader')

		# Index entire directory (without loading anything yet)
		self.data_index, self.label_index = ([], [])
		self.indexFrames()

		# Compute the maximum amount of frames that fit into storage
		frame_size, label_size, data_shape, label_shape, data_type, label_type = self.getFrameSize()
		max_frame_cnt = math.floor(max_memory / (frame_size+label_size))
		memory_frames = frame_size * max_frame_cnt
		memory_labels = label_size * max_frame_cnt

		self.data_types = ((max_frame_cnt,)+data_shape, (max_frame_cnt,)+label_shape, data_type, label_type)
		self.frameCnt = max_frame_cnt

		# Initialize shared memory for async data loading
		self.logger.info(f'Setting up shared memory for {max_frame_cnt} frames: {memory_frames} + {memory_labels} bytes')
		self.shm_data = shared_memory.SharedMemory(create=True, size=memory_frames)
		self.shm_label = shared_memory.SharedMemory(create=True, size=memory_labels)
		self.memory_data_map = np.ndarray((max_frame_cnt,)+data_shape, dtype=data_type, buffer=self.shm_data.buf)
		self.memory_labels_map = np.ndarray((max_frame_cnt,)+label_shape, dtype=label_type, buffer=self.shm_data.buf)

		# Initialize data queues
		self.frames_loaded = multiprocessing.Queue(maxsize=max_frame_cnt)
		self.frames_free = multiprocessing.Queue(maxsize=max_frame_cnt)

		# Mark all frames as free
		for i in range(max_frame_cnt):
			self.frames_free.put(i)


	def indexFrames(self):
		self.logger.info(f'indexing {self.dir_name} data')
		seq_folders = os.listdir(os.path.join(self.data_dir, self.dir_name))
		for seq in seq_folders:
			seq_path = os.path.join(self.data_dir, self.dir_name, seq)
			files_in_seq = os.listdir(os.path.join(seq_path, 'reduced_velo'))

			for data_num in range(0, len(files_in_seq),self.skip_frames): # too much of dataset we skipping files
				# Check if process is still supposed to run
				
				self.data_index.append(os.path.join(seq_path, 'reduced_velo', "%06d.npy" % data_num))
				self.label_index.append(os.path.join(seq_path, 'gnd_labels', "%06d.npy" % data_num))
		
		self.logger.info(f'Data indexing completed. Found {len(self.data_index)} frames')

	def getFrameSize(self):
		point_set1 = np.load(self.data_index[0])[:,:self.num_input_features] #(N,3) point set
		point_set2 = np.load(self.label_index[0]) #(N,3) point set
		return (point_set1.nbytes, point_set2.nbytes, point_set1.shape, point_set2.shape, point_set1.dtype, point_set2.dtype)
	
	def getOverallFrameCount(self):
		return len(self.data_index)

	def getSharedMemoryHandles(self):
		return (self.shm_data.name, self.shm_label.name)

#20128, 4000128
	def getNextFrameID(self) -> int:
		# Release the previously loaded data (it won't be needed anymore)
		if self.currentId != -1:
			self.frames_free.put_nowait(self.currentId)
			# self.logger.debug(f'Released last frame: {self.currentId}')

		self.currentId: int = self.frames_loaded.get()
		# self.logger.debug(f'Got next frame: {self.currentId}')
		return self.currentId

	def __del__(self):
		self.logger.info('Exiting async data loader')
		self.frames_loaded.full()
		self.frames_free.empty()
		self.shm_data.unlink()
		self.shm_label.unlink()
		

	def run(self):
		self.logger.info(f'loading {self.dir_name} data ')
		while(self.is_alive()):
			random_frames = list(zip(self.data_index, self.label_index))
			random.shuffle(random_frames)
			for data_path, label_path in random_frames:
					# Wait for a free block of memory
					freeIndex = self.frames_free.get()
					# self.logger.debug(f'Load next frame into memory: {freeIndex} - {data_path.split("/")[-1]}')

					# Load frame data and ground labels into memory
					self.memory_data_map[freeIndex] = np.load(data_path)[:,:self.num_input_features] #(N,3) point set
					self.memory_labels_map[freeIndex] = np.load(label_path) # (W x L)
					self.frames_loaded.put(freeIndex)
		
		logging.info('Data loading completed!')

		

class kitti_gnd_async(Dataset):
	def __init__(self, data_dir, train = True, skip_frames = 1, num_input_features = 3, max_memory=4e6, logger: logging.Logger=logging.root):
		self.train = train
		self.logger = logger

		self.skip_frames = skip_frames
		self.num_input_features = num_input_features
		self.current_memory_size = 0

		self.dir_name = 'training' if self.train else 'validation'

		self.data_loader = AsyncDataLoader(data_dir, self.dir_name, max_memory=max_memory, num_input_features=num_input_features, skip_frames=skip_frames, parent_logger=logger)
		
		# Mount shared memory data
		data_handle, label_handle = self.data_loader.getSharedMemoryHandles()
		data_shape, label_shape, data_type, label_type = self.data_loader.data_types
		self.shm_data = shared_memory.SharedMemory(name=data_handle)
		self.shm_label = shared_memory.SharedMemory(name=label_handle)
		self.loaded_data = np.ndarray(data_shape, dtype=data_type, buffer=self.shm_data.buf)
		self.loaded_labels = np.ndarray(label_shape, dtype=label_type, buffer=self.shm_label.buf)

		self.last_index = -1

		self.data_loader.start()

	def __getitem__(self, index):
		if index != (self.last_index + 1):
			logging.warn(f'Requested index is not in order: {self.last_index} -> {index}')
		self.last_index = index
		data_pos = self.data_loader.getNextFrameID()
		# self.logger.debug(f'Access: {index} ({data_pos})')
		return self.loaded_data[data_pos], self.loaded_labels[data_pos]

	def __len__(self):
		return self.data_loader.getOverallFrameCount()

	def __del__(self):
		if self.shm_data != None:
			self.shm_data.close()
			self.shm_label.close()


class kitti_gnd_sync(Dataset):
	def __init__(self, data_dir, train = True, skip_frames = 1, num_input_features=3, max_memory=1e9, logger: logging.Logger=logging.root):
		self.train = train
		self.logger = logger
		self.max_memory = max_memory
		self.num_input_features = num_input_features

		self.dir_name = 'training' if train else 'validation'
		self.data_path = os.path.join(data_dir, self.dir_name)

		self.current_memory_data = 0
		self.current_memory_labels = 0
		self.last_memory_size_frame = 0
		self.loaded_data = []
		self.loaded_labels = []

		self.logger.info(f'loading {self.dir_name} data')
		seq_folders = os.listdir(self.data_path)
		for seq_num in seq_folders:
			seq_path = os.path.join(self.data_path, seq_num)
			files_in_seq = os.listdir(os.path.join(seq_path, 'reduced_velo'))

			for data_num in range(0, len(files_in_seq),skip_frames): # too much of dataset we skipping files
				file_name = files_in_seq[data_num]
				if self.current_memory_data + self.current_memory_labels + self.last_memory_size_frame > self.max_memory:
					self.logger.warn(f'Stop loading data at frame {file_name} in seq {seq_num}!\n\tReached {self.current_memory_data} Bytes of data + {self.current_memory_labels} Bytes of labels (={self.current_memory_data+self.current_memory_labels} Bytes). Would go over the limit of {self.max_memory} Bytes')
					return
				data_file_path = os.path.join(seq_path, 'reduced_velo', file_name)
				point_set = np.load(data_file_path)[:,:self.num_input_features] #(N,3) point set
				self.loaded_data.append(point_set)

				self.current_memory_data += point_set.nbytes

				label_path = os.path.join(seq_path, 'gnd_labels', file_name)
				label = np.load(label_path) # (W x L)
				self.loaded_labels.append(label)

				self.current_memory_labels += label.nbytes
		
		self.logger.info(f'Loaded {self.current_memory_data} Bytes of data + {self.current_memory_labels} Bytes of labels (={self.current_memory_data+self.current_memory_labels} Bytes)')


	def __getitem__(self, index):
		return self.loaded_data[index], self.loaded_labels[index]


	def __len__(self):
		return len(self.loaded_data)




def get_valid_loader(data_dir, batch = 4, skip = 1, num_input_features = 3, max_memory=4e6, parent_logger=logging.root):

	parent_logger = parent_logger.getChild('dataset_provider.valid')

	use_cuda = torch.cuda.is_available()
	if use_cuda:
		parent_logger.info("using cuda")
		num_workers = 1
		pin_memory = True
	else:
		num_workers = 4
		pin_memory = True


	valid_loader = DataLoader(kitti_gnd_sync(data_dir,train = False, skip_frames = skip, num_input_features=num_input_features, max_memory=max_memory, logger=parent_logger),
					batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,drop_last=True)

	parent_logger.info("Valid Data size %d",len(valid_loader)*batch)

	return valid_loader





def get_train_loader(data_dir, batch = 4, skip = 1, num_input_features = 3, max_memory=4e6, parent_logger=logging.root):

	parent_logger = parent_logger.getChild('dataset_provider.train')

	use_cuda = torch.cuda.is_available()
	if use_cuda:
		parent_logger.info("using cuda")
		num_workers = 1
		pin_memory = True
	else:
		num_workers = 4
		pin_memory = True

	train_loader = DataLoader(kitti_gnd_sync(data_dir,train = True, skip_frames = skip, num_input_features=num_input_features, max_memory=max_memory, logger=parent_logger),
					batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,drop_last=True)

	parent_logger.info("Train Data size %d",len(train_loader)*batch)

	return train_loader




if __name__ == '__main__':

	with open('config/config_kittiSem.yaml') as f:
		config_dict = yaml.load(f, Loader=yaml.FullLoader)

	class ConfigClass:
		def __init__(self, **entries):
			self.__dict__.update(entries)

	cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use
	
	# IO Includes
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D

	# Ros Includes
	import rospy
	from sensor_msgs.msg import PointCloud2
	import std_msgs.msg
	from visualization_msgs.msg import Marker
	import ros_numpy
	from utils.ros_utils import np2ros_pub, gnd_marker_pub
	
	rospy.init_node('gnd_data_provider', anonymous=True)
	pcl_pub = rospy.Publisher("/kitti/reduced_velo", PointCloud2, queue_size=10)
	marker_pub = rospy.Publisher("/kitti/gnd_marker", Marker, queue_size=10)
	fig = plt.figure()
	data_dir = '/home/anshul/es3cap/my_codes/GndNet/data/'
	train_loader, valid_loader =  get_data_loaders(data_dir)
	
	for batch_idx, (data, labels) in enumerate(valid_loader):
		B = data.shape[0] # Batch size
		N = data.shape[1] # Num of points in PointCloud
		print(N)
		data = data.float()
		labels = labels.float()

		for i in range(B):
			pdb.set_trace()
			np2ros_pub(data[i].numpy(),pcl_pub)
			gnd_marker_pub(labels[i].numpy(),marker_pub, cfg, color = "red")
			# # visualize_gnd_3D(gnd_label, fig)
			# visualize_2D(labels[i],data[i],fig)

