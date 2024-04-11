import numpy as np
from numba import njit, prange
from scipy.spatial.transform import Rotation as R
from gnd_net.dataset_utils.gnd_data_generator.frustrum_culling import filter_points_by_frustum, extract_pc_in_box2d
import time


class AugmentationConfig():
	def __init__(self, grid, 
			  keep_original=False, 
			  num_rotations = 0, 
			  num_height_var = 0,
			  maxFrontSlope = 5, 
			  maxSideTild = 0, 
			  maxRotation = 0, 
			  maxHeight = 0) -> None:
		
		self.grid = grid
		self.keep_original = keep_original
		self.num_rotations = num_rotations
		self.num_height_var = num_height_var
		self.maxFrontSlope = maxFrontSlope
		self.maxSideTild = maxSideTild
		self.maxRotation = maxRotation
		self.maxHeight = maxHeight

class DataAugmentation():
	def __init__(self, config: AugmentationConfig) -> None:
		"""keep_original=True will temporarly make a copy of the entire dataset. Make sure you have enough memory!"""

		self.config = config

	def getAugmentedData(self, velodyne_data):
		"""Augments a whole batch of data at once (roations and height variations)
		 Only works with point clouds that have the same number of points"""

		data = velodyne_data if not self.config.keep_original else np.copy(velodyne_data)

		if self.config.num_rotations > 0:
			# Duplicate the frames if necessary
			if self.config.num_rotations > 1:
				data = np.repeat(data, self.config.num_rotations, axis=0)
			self.augmentRotation(data, self.config.maxFrontSlope, self.config.maxSideTild, self.config.maxRotation)

		if self.config.num_height_var > 0:
			# Duplicate the frames if necessary
			if self.config.num_height_var > 1:
				data = np.repeat(data, self.config.num_height_var, axis=0)
			self.augmentHeight(data, self.config.maxHeight)
		
		if self.config.keep_original:
			data = np.concatenate((velodyne_data, data))
		
		return data
	
	def augmentRotation(self, data: 'np.ndarray', maxFrontSlope = 5, maxSideTild = 0, maxRotation = 0):
		# Define max positive rotation
		theta = np.asarray([maxRotation, maxSideTild, maxFrontSlope])

		# Get all random rotations for every frame
		rotations = theta * (2 * np.random.rand(data.shape[0], 3) - 1)

		# Convert the euler angles to rotation matrices
		r = R.from_euler('zyx', rotations, degrees=True).as_matrix()

		# Rotate all points within each frame
		for i in range(data.shape[0]):
			# Rotate all points
			data[i,:,:3] = np.dot(data[i,:,:3], r[i].T)

	def augmentHeight(self, data: 'np.ndarray', maxHeight = 5):
		random_height = maxHeight * (2 * np.random.rand(data.shape[0]) - 1)
		data[:,:,2] += random_height[:,np.newaxis] # Each frame gets its random height variation added

	def getAugmentedDataWithGroundTruth(self, velodyne_data, ground_labels):
		"""Augments a whole batch of data at once (roations and height variations)
		 Only works with point clouds that have the same number of points"""

		data, labels = (velodyne_data, ground_labels) if not self.config.keep_original else (np.copy(velodyne_data), np.copy(ground_labels))
		# if self.config.num_camera_fov > 0:
		# 	# Duplicate the frames if necessary
		# 	if self.config.num_rotations > 1:
		# 		data = np.repeat(data, self.config.num_rotations, axis=0)
		# 		labels = np.repeat(labels, self.config.num_rotations, axis=0)
		# 	self.cameraFOV(data)

		if self.config.num_rotations > 0:
			# Duplicate the frames if necessary
			if self.config.num_rotations > 1:
				data = np.repeat(data, self.config.num_rotations, axis=0)
				labels = np.repeat(labels, self.config.num_rotations, axis=0)
			self.augmentRotationWithGroundTruth(data, labels, self.config.grid, self.config.maxFrontSlope, self.config.maxSideTild, self.config.maxRotation)

		if self.config.num_height_var > 0:
			# Duplicate the frames if necessary
			if self.config.num_height_var > 1:
				data = np.repeat(data, self.config.num_height_var, axis=0)
				labels = np.repeat(labels, self.config.num_height_var, axis=0)
			self.augmentHeightWithGroundTruth(data, labels, self.config.maxHeight)
		
		if self.config.keep_original:
			data = np.concatenate((velodyne_data, data))
			labels = np.concatenate((ground_labels, labels))
		
		return (data, labels)

	def augmentRotationWithGroundTruth(self, data: 'np.ndarray', labels: 'np.ndarray', grid: list[int], maxFrontSlope = 5, maxSideTild = 0, maxRotation = 0):
		# Define max positive rotation
		theta = np.asarray([maxRotation, maxSideTild, maxFrontSlope])

		# Get all random rotations for every frame
		rotations = theta * (2 * np.random.rand(data.shape[0], 3) - 1)

		# Convert the euler angles to rotation matrices
		r = R.from_euler('zyx', rotations, degrees=True).as_matrix()

		# Get conversion from grid to coordinates
		grid = np.asarray(grid)
		shape = np.asarray(labels[0].shape)
		offset = grid[0:2]
		scale = (grid[2:4] - grid[0:2]) / shape
		indices = np.indices(labels[0].shape).T.reshape(-1, 2) # Assume all label shapes are the same (wich sould always be the case)
		labels_coordinates = indices * scale + offset # These are the coordinates for each grid component of the labels

		# Rotate all points within each frame
		for i in range(data.shape[0]):
			# Rotate all points
			data[i,:,:3] = np.dot(data[i,:,:3], r[i].T)

			# Rotate ground plane
			grid_as_list = np.concatenate((labels_coordinates, labels[i].reshape((1,-1)).T), axis=1) # [x, y, gnd_height]
			grid_transformed = np.dot(grid_as_list, r[i].T) # Transform the ground plane with the rotation matrix
			labels[i] = grid_transformed[:,2].reshape(labels[i].shape) # Convert back into a grid
	
	def augmentHeightWithGroundTruth(self, data: 'np.ndarray', maxHeight = 5):
		random_height = maxHeight * (2 * np.random.rand(data.shape[0]) - 1)
		data[:,:,2] += random_height[:,np.newaxis] # Each frame gets its random height variation added
		labels[:] += random_height[:,np.newaxis, np.newaxis] # For each frame the entire ground map gets the height variation added

	def getCameraFOV(self, data: 'np.ndarray', labels: 'np.ndarray'):
		observer_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		observer_direction = np.array([1, 0, 0]).astype(np.float32)

		fov = 115  # Field of view in degrees
		aspect_ratio = 16/9  # Width/height ratio of the viewport
		near = 0.1
		far = 10.0

		#for i in range(data.shape[0]):
		print(data.shape)
		data = extract_pc_in_box2d(data, self.config.grid)
		print(data.shape)
		data = filter_points_by_frustum(data, observer_position, observer_direction, fov, aspect_ratio, near, far)
		print(data.shape)
		return (data, labels)