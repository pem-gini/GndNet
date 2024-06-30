from numba import jit,types
from functools import reduce
from scipy.spatial import Delaunay
import torch
import numpy as np
from numpy.linalg import inv
import logging

from scipy import signal
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# @jit(nopython=True)
def in_hull(p, hull):
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds


# @jit(nopython=True)
def extract_pc_in_box2d(pc, box2d):
    ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
    box2d_corners = np.zeros((4,2))
    box2d_corners[0,:] = [box2d[0],box2d[1]] 
    box2d_corners[1,:] = [box2d[2],box2d[1]] 
    box2d_corners[2,:] = [box2d[2],box2d[3]] 
    box2d_corners[3,:] = [box2d[0],box2d[3]] 
    box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
    return pc[box2d_roi_inds,:]


# @jit(nopython=True)
def random_sample_torch(cloud, N):
    if(cloud.size > 0):
        cloud = torch.from_numpy(np.asarray(cloud)).float().cuda()

        points_count = cloud.shape[0]
        # pdb.set_trace()
        # print("indices", len(ind))
        if(points_count > 1):
            prob = torch.randperm(points_count) # sampling without replacement
            if(points_count > N):
                idx = prob[:N]
                sampled_cloud = cloud[idx]
                # print(len(crop))
            else:
                r = int(N/points_count)
                cloud = cloud.repeat(r+1,1)
                sampled_cloud = cloud[:N]

        else:
            sampled_cloud = torch.ones(N,3).cuda()
    else:
        sampled_cloud = torch.ones(N,3).cuda()
    return sampled_cloud.cpu().numpy()


# @jit(nopython=True)
def random_sample_numpy(cloud, N):
    if(cloud.size > 0):
        points_count = cloud.shape[0]
        if(points_count > 1):
            idx = np.random.choice(points_count,N) # sample with replacement
            sampled_cloud = cloud[idx]
        else:
            sampled_cloud = np.ones((N,3))
    else:
        sampled_cloud = np.ones((N,3))
    return sampled_cloud

@jit(nopython=True)
def shift_cloud_func(cloud, height):
    cloud += np.array([0,0,height,0], dtype=np.float32)
    return cloud



def parse_calibration(filename):
  """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib


def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses


# @jit(nopython=True)
def segment_cloud(cloud, gnd_labels):
    index = np.isin(cloud[:,3], gnd_labels)
    gnd = cloud[index]
    obs = cloud[np.invert(index)]
    return gnd, obs


@jit(nopython=True)
def lidar_to_img(points, grid_size, voxel_size, fill, lidar_height):
    # pdb.set_trace()
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] + lidar_height
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data /voxel_size # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    # lidar_data = np.reshape(lidar_data, (-1, 2))
    voxelmap_shape = (grid_size[2:]-grid_size[:2])/voxel_size
    lidar_img = np.zeros((int(voxelmap_shape[0]),int(voxelmap_shape[1])))
    N = lidar_data.shape[0]
    for i in range(N):
        if(height_data[i] < 10):
            if (0 < lidar_data[i,0] < lidar_img.shape[0]) and (0 < lidar_data[i,1] < lidar_img.shape[1]):
                lidar_img[lidar_data[i,0],lidar_data[i,1]] = fill
    return lidar_img


@jit(nopython=True)
def lidar_to_heightmap(points, grid_size, voxel_size, max_points, lidar_height):
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] + lidar_height
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data / voxel_size # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    # lidar_data = np.reshape(lidar_data, (-1, 2))
    heightmap_shape = (grid_size[2:]-grid_size[:2])/voxel_size
    heightmap = np.zeros((int(heightmap_shape[0]),int(heightmap_shape[1]), max_points))
    num_points = np.zeros((int(heightmap_shape[0]),int(heightmap_shape[1])), dtype = np.int32) # num of points in each cell # np.ones just to avoid division by zero
    N = lidar_data.shape[0] # Total number of points
    for i in range(N):
        x = lidar_data[i,0]
        y = lidar_data[i,1]
        z = height_data[i]
        if(z < 10):
            if (0 <= x < heightmap.shape[0]) and (0 <= y < heightmap.shape[1]):
                k = num_points[x,y] # current num of points in a cell
                if k < max_points:
                    heightmap[x,y,k] = z
                    num_points[x,y] += 1
    
    return heightmap.sum(axis = 2), heightmap, num_points

#     return heightmap

# def lidar_to_heightmap(points, grid_size, voxel_size, max_points):
#     return _lidar_to_heightmap(points, grid_size, voxel_size, max_points).max(axis=2)


def rotate_cloud(cloud, theta):
    # xyz = cloud[:,:3]
    # xyz = np.concatenate((xyz,np.array([1]), axis = 1))
    r = R.from_euler('zyx', theta, degrees=True)
    r = r.as_matrix()
    cloud[:,:3] = np.dot(cloud[:,:3], r.T)
    return cloud


@jit(nopython=True)
def semantically_segment_cloud(points, grid_size, voxel_size, elevation_map, lidar_height, threshold = 0.08):
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] + lidar_height
    rgb = np.zeros((points.shape[0],3))
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data /voxel_size # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    N = lidar_data.shape[0] # Total number of points
    for i in range(N):
        x = lidar_data[i,0]
        y = lidar_data[i,1]
        z = height_data[i]
        if (0 < x < elevation_map.shape[0]) and (0 < y < elevation_map.shape[1]):
            if z > elevation_map[x,y] + threshold:
                rgb[i,0] = 1
                # rgb[i,1] = 1
            else:
                rgb[i,0] = 0
        else:
            rgb[i,0] = -1
    return rgb

def compute_ground_plane(cloud, grid_size, voxel_size, lidar_height, export_intermediate_step=False, visualizer=None, logger=logging.root):
    # start = time.time()
    # remove all non ground points; gnd labels = [40, 44, 48, 49]
    # gnd, obs = segment_cloud(cloud,[40, 44, 48, 49])
    # A complete list of all labels: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    gnd, obs = segment_cloud(cloud,[40, 44, 48, 49,60,72]) # ['road', 'parking', 'sidewalk', 'other-ground', 'lane-marking', 'terrain'] 

    grid_size_np = np.asarray(grid_size)
    gnd_heightmap, heightmap, num_points = lidar_to_heightmap(np.copy(gnd), grid_size_np, voxel_size, max_points = 100, lidar_height=lidar_height)

    filled_voxels = num_points!=0
    gnd_heightmap = np.divide(gnd_heightmap, num_points, where=filled_voxels)

    if export_intermediate_step:
        np.save('heightmap', gnd_heightmap)
        np.save('heightmap_raw', heightmap)
        np.save('num_points', num_points)

    # Interpolate missing spots
    for i in range(10): # Max 3 runs
        y,x = np.where(filled_voxels)
        interpL = LinearNDInterpolator(list(zip(y,x)), gnd_heightmap[y,x])
        xx = np.arange(gnd_heightmap.shape[0])
        yy = np.arange(gnd_heightmap.shape[1])
        X, Y = np.meshgrid(xx, yy, indexing='ij')
        interpolated_linear = interpL(X,Y)
    
        empty_voxels = np.ma.masked_invalid(interpolated_linear).mask
        y,x = np.where(np.logical_not(empty_voxels))
        interpQ = NearestNDInterpolator(list(zip(y,x)), interpolated_linear[y,x])

        y,x=np.where(empty_voxels)
        image_result = np.copy(interpolated_linear)
        image_result[y,x] = np.nan_to_num(interpQ(y,x))

        # Make sure there are no outliers
        average_ground = signal.convolve2d(image_result, np.ones((5,5))/25, mode='same' , boundary='symm') # Average in 5x5 squares = 1m²
        diff_to_average = np.abs(image_result - average_ground) # Should be max less than 0.1m <= 10% of elevation
        outliers = diff_to_average > 0.1
        
        # gradient = np.zeros(image_result.shape)
        # gradient[:-1,:-1] = np.maximum((image_result[:-1,:] - image_result[1:,:])[:,:-1], (image_result[:,:-1] - image_result[:,1:])[:-1,:])
        # gradient[:,-1] = gradient[:,-2]
        # gradient[-1,:] = gradient[-2,:]
        # gradient[-1,-1] = gradient[-2,-2]
        # outliers = gradient > 0.1

        # If there is a visualizer function given call it now
        if visualizer != None:
            visualizer(i, filled_voxels, diff_to_average, outliers, gnd_heightmap, interpolated_linear, image_result, cloud)

        # If there are no outliers, we are done here
        if not outliers.any():
            break
        
        # Otherwise, remove the outliers in the original data and run the interpolation again
        filled_voxels[outliers] = False # Simply mark the squares of the outliers as free
        logger.debug(f'Remove outliers, rerun ({i})')

    return gnd, image_result
