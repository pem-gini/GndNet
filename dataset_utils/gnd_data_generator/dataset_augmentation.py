import numpy as np
from numba import njit, prange
from scipy.spatial.transform import Rotation as R
from gnd_net.dataset_utils.gnd_data_generator.frustrum_culling import filter_points_by_frustum, extract_pc_in_box2d
import time


class AugmentationConfig():
	def __init__(self, grid, voxel_size,
			  keep_original=False, 
			  num_rotations = 0, 
			  num_height_var = 0,
			  num_noise_aug = 0,
			  maxFrontSlope = 5, 
			  maxSideTild = 0, 
			  maxRotation = 0, 
			  maxHeight = 0,
			  noise_coefficient_top = (0,0), 
			  noise_coefficient_bottom = (0.4,0.6), 
			  noise_min_distance = (1.2, 4), 
			  noise_density_top = (1,50), 
			  noise_density_bottom=(1,50)
			  ) -> None:
		
		self.grid = np.array(grid)
		self.voxel_size = voxel_size
		self.keep_original = keep_original
		self.num_rotations = num_rotations
		self.num_height_var = num_height_var
		self.num_noise_aug = num_noise_aug
		self.maxFrontSlope = maxFrontSlope
		self.maxSideTild = maxSideTild
		self.maxRotation = maxRotation
		self.maxHeight = maxHeight
		self.noise_coefficient_top = noise_coefficient_top
		self.noise_coefficient_bottom = noise_coefficient_bottom
		self.noise_min_distance = noise_min_distance
		self.noise_density_top = noise_density_top
		self.noise_density_bottom = noise_density_bottom

		self.num_augmentations = num_rotations + num_height_var + keep_original

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
	
	def addNoise(self, data: 'np.ndarray', gnd_planes: 'np.ndarray'):
		return self._addNoise(data, gnd_planes, self.config.noise_coefficient_top, self.config.noise_coefficient_bottom, self.config.noise_min_distance, self.config.noise_density_top, self.config.noise_density_bottom)
	
	def _addNoise(self, data: 'np.ndarray', gnd_plane: 'np.ndarray', noise_coefficient_top = (0,0), noise_coefficient_bottom = (0.4,0.6), min_distance = (1.2, 4), density_top = (1,50), density_bottom=(1,50)):
		"""Adds noise to the point cloud and labels it as category 260. The noise will be shaped as a triangle if you look at it from the side.
		The density is in noisy points per cube meter, everything else is in meter"""
		
		# Find the range of the current points
		grid_range = self.config.grid
		pc_range = np.empty((3,3))
		pc_range[:,0] = data[:,:3].min(axis=0) # Get the min x,y,z of all points and store it in the first column
		pc_range[:2,1] = grid_range[:2]	# Load the min grid range into the second column
		pc_range[:2,0] = pc_range[:2,:2].max(axis=1)  # Take the greater x,y values of the minimum grid range and point cloud to make sure we are not out of index

		pc_range[:,1] = data[:,:3].max(axis=0) # Get the max x,y,z of all points and store it in the second column
		pc_range[:2,1] = grid_range[2:] # Load the max grid range into the third column
		#pc_range[:,:2,1] = pc_range[:,:2,1:].min(axis=2) # Take the greater x,y values of the max grid range and point cloud to make sure we are not out of index

		# Make sure that the minimum noise distance is either the first measured point or the min distance
		min_distance = np.random.random() * (min_distance[1]-min_distance[0]) + min_distance[0]
		pc_range[0,0] = np.maximum(pc_range[0,0], min_distance)

		min_range = pc_range #np.empty((3,2))
		# min_range[0] = pc_range[:,0]
		# min_range[1] = pc_range[:,1]

		noise_cnt = np.zeros(2, dtype=np.uint64)

		# Find the target volume for the noise
		for i in range(2): # Run everthing for the top (0) and the bottom (1)
			coefficients = np.array(noise_coefficient_top) if i == 0 else np.array(noise_coefficient_bottom)
			densities = np.array(density_top) if i == 0 else np.array(density_bottom)
			# If the lower and upper limits are zero, do not add noise to this side
			if coefficients[0] == 0 and coefficients[1] == 0:
				continue

			coefficient = np.random.random() * (coefficients[1]-coefficients[0]) + coefficients[0]
			density = np.random.random() * (densities[1]-densities[0]) + densities[0]
			max_x = min_range[0][1]

			# print(f'{i}: Coefficient: {coefficient}, Density: {density}, Min distance: {min_distance}')

			# If the noise only starts behind the data, simply do not add noise
			if min_distance >= max_x:
				return data
			
			area_xz_plane = 0.5 * (max_x - min_distance) * max_x * coefficient
			volume = area_xz_plane * (min_range[0][1] - min_range[0][0])
			noise_cnt[i] = int(volume * density)
		
		cut_off_back = 10 - np.random.random() * 3
		padding_coefficient = np.abs(np.random.normal(0, 1)) / pc_range[0,1]
		valid_data = data[data[:,0] <= cut_off_back]
		new_data = np.empty((valid_data.shape[0]+int(noise_cnt.sum()), data.shape[1]))
		new_data[:valid_data.shape[0]] = valid_data # Copy the old data into the new tensor

		for i, factor in enumerate([1, -1]): # Run everthing for the top (1) and the bottom (-1)
			if noise_cnt[i] == 0: continue

			# Add new random points
			new_data[valid_data.shape[0]:,:2] = np.random.random((noise_cnt[i], 2)) * (pc_range[:2,1]-pc_range[:2,0]) + pc_range[:2,0]
			# Find the corresponding grid squares to each random point in the xy plane
			grid_indices = np.floor((new_data[valid_data.shape[0]:,:2] - np.array(grid_range)[:2]) / self.config.voxel_size).astype(np.int32)
			# Now add the noise below and above the ground plane within each corresponding voxel
			new_data[valid_data.shape[0]:,2] = gnd_plane[grid_indices[:,0], grid_indices[:,1]] - padding_coefficient * new_data[valid_data.shape[0]:,0] + np.abs(np.random.normal(0, (new_data[valid_data.shape[0]:,0]-pc_range[0,0])*coefficient)) * factor # Make sure the noise only lies in the specified half (bottom or top)
			new_data[valid_data.shape[0]:,3] = 260

		return new_data
 
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