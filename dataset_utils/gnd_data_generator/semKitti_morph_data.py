use_ros = True
visualize = False
export = False

import os
import yaml
import ipdb as pdb
import numpy as np
from numpy.linalg import inv

if use_ros:
    from semiKitti_ros_utils import ros_init
else:
    from dataset_generator_utils import random_sample_numpy, extract_pc_in_box2d

if visualize:
    import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation as R
import cv2

from numba import jit
from scipy import signal, ndimage
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator, NearestNDInterpolator
from skimage.morphology import reconstruction
from skimage.restoration import inpaint
from skimage.morphology import erosion, dilation
from torch.utils.data.sampler import SubsetRandomSampler


if visualize:
    plt.ion()

# with open('config/config.yaml') as f:
with open('../../config/config_kittiSem2.yaml') as f:
	config_dict = yaml.load(f, Loader=yaml.FullLoader)

class ConfigClass:
	def __init__(self, **entries):
		self.__dict__.update(entries)

cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use


# # resolution of ground estimator grid is 1m x 1m 
# visualize = False
# pc_range = [0.6, -30, 60.6, 30]
# grid_size = [0, -30, 60, 30]
pc_range = cfg.pc_range[:2] + cfg.pc_range[3:5] # select minmax xy
grid_size = cfg.grid_range
length = int(grid_size[2] - grid_size[0]) # x direction
width = int(grid_size[3] - grid_size[1])    # y direction
lidar_height = cfg.lidar_height

voxel_dimensions = cfg.voxel_size
if voxel_dimensions[0] != voxel_dimensions[1]:
    print('Non-square voxels are not yet supported!')
    exit(-1)
voxel_size = voxel_dimensions[0]

print(grid_size)
print(type(grid_size))
print(voxel_size)
print(type(voxel_size))

data_dir = '/home/finn/pem/ducktrain/GndNet/data/prediction/seq_000'
out_dir = '/home/finn/pem/ducktrain/GndNet/data/training/000'


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
    index = np.isin(cloud[:,4], gnd_labels)
    gnd = cloud[index]
    obs = cloud[np.invert(index)]
    return gnd, obs


@jit(nopython=True)
def lidar_to_img(points, grid_size, voxel_size, fill):
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
def lidar_to_heightmap(points, grid_size, voxel_size, max_points):
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] + lidar_height
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data /voxel_size # multiplying by the resolution
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
def semantically_segment_cloud(points, grid_size, voxel_size, elevation_map, threshold = 0.08):
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

if visualize:
    fig = plt.figure()
else:
    fig = None

def process_cloud(cloud):
    # start = time.time()
    # remove all non ground points; gnd labels = [40, 44, 48, 49]
    # gnd, obs = segment_cloud(cloud,[40, 44, 48, 49])
    gnd, obs = segment_cloud(cloud,[40, 44, 48, 49,60,72])
    global visualize, grid_size, voxel_size

    grid_size_np = np.asarray(grid_size)
    gnd_img = lidar_to_img(np.copy(gnd), grid_size_np, voxel_size, fill = 1)
    gnd_heightmap, heightmap, num_points = lidar_to_heightmap(np.copy(gnd), grid_size_np, voxel_size, max_points = 100)

    filled_voxels = num_points!=0
    gnd_heightmap = np.divide(gnd_heightmap, num_points, where=filled_voxels)

    if export:
        np.save('heightmap', gnd_heightmap)
        np.save('heightmap_raw', heightmap)
        np.save('num_points', num_points)


    # Interpolate missing spots
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

    # kernel = np.ones((5,5),np.uint8)
    # gnd_img_dil = cv2.dilate(gnd_img,kernel,iterations = 2)
    # mask = gnd_img_dil - gnd_img
    # inpaint_result = inpaint.inpaint_biharmonic(gnd_heightmap, mask)
    #conv_kernel = np.ones((3,3))
    #image_result = signal.convolve2d(image_result_2, conv_kernel, boundary='wrap', mode='valid')/conv_kernel.sum()
    seg = semantically_segment_cloud(cloud.copy(), grid_size_np, voxel_size, image_result)

    if visualize:
        fig.clear()

        fig.add_subplot(2, 3, 1)
        plt.imshow(gnd_img, interpolation='nearest')

        # fig.add_subplot(2, 3, 2)
        # plt.imshow(gnd_img_dil, interpolation='nearest')

        # fig.add_subplot(2, 3, 3)
        # plt.imshow(mask, interpolation='nearest')

        fig.add_subplot(2, 3, 4)
        cs = plt.imshow(gnd_heightmap, interpolation='nearest')
        fig.colorbar(cs)
        
        fig.add_subplot(2, 3, 5)
        cs = plt.imshow(interpolated_linear, interpolation='nearest')
        fig.colorbar(cs)

        # image_result = inpaint.inpaint_biharmonic(image_result, mask)
        # image_result = cv2.dilate(image_result,kernel,iterations = 1)

        # kernel = np.array([[0,1,0],
        #                    [1,0,1],
        #                    [0,1,0]])
        # kernel = np.ones((7,7),np.uint8)
        # kernel[3,3] = 0
        # ind = mask == 1

        # for i in range(10):
        #     conv_out = signal.convolve2d(gnd_heightmap, kernel, boundary='wrap', mode='same')/kernel.sum()
        #     gnd_heightmap[ind] = conv_out[ind]

        fig.add_subplot(2, 3, 6)
        cs = plt.imshow(image_result, interpolation='nearest')
        cbar = fig.colorbar(cs)
        plt.show(block=False)
        fig.canvas.flush_events()
        # cbar.remove()

    return gnd, image_result.T, seg



class KittiSemanticDataGenerator():

    def __init__(self, data_dir, logger=None) -> None:

        self.data_dir = data_dir
        self.velodyne_dir = os.path.join(self.data_dir, "velodyne/")
        self.label_dir = os.path.join(self.data_dir, 'labels/')

        self.frames_cnt = len(os.listdir(self.velodyne_dir))
        self.calibration = parse_calibration(os.path.join(self.data_dir, "calib.txt"))
        self.poses = parse_poses(os.path.join(self.data_dir, "poses.txt"), self.calibration)

        self.logger = logger

        self.angle = 0
        self.increase = True
        self.current_frame = 0

        self.cloud = None
        self.points = None
        self.gnd_label = None
        self.seg = None

    def kitti_semantic_data_generate(self):
        current_frame = self.current_frame

        # Increase the frame cnt and cancle timer if we reached the end
        self.current_frame += 1
        if self.current_frame > self.frames_cnt:
            return False

        points_path = os.path.join(self.velodyne_dir, "%06d.bin" % current_frame)
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)

        label_path = os.path.join(self.label_dir, "%06d.label" % current_frame)
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        if label.shape[0] == points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
        label = np.expand_dims(label, axis = 1)
        points = np.concatenate((points,label), axis = 1)

        if self.angle > 5.0:
            self.increase = False
        elif self.angle < -5.0:
            self.increase = True


        if self.increase:
            self.angle +=0.1
        else:
            self.angle -=0.1

        if self.logger != None:
            self.logger(f'Frame {current_frame} with angle {self.angle}')

        self.points = rotate_cloud(points, theta = [0,5,self.angle]) #zyx

        self.cloud, self.gnd_label, self.seg = process_cloud(points)
        # cloud = process_cloud(points)

        # points += np.array([0,0,lidar_height,0,0], dtype=np.float32)
        # points[0,2] += lidar_height

        global cfg, lidar_height
        if cfg.shift_cloud:
            self.points[:,2] += lidar_height
            self.cloud[:,2] += lidar_height

        return True


def main(data_dir: str):
    # When using ROS, the data will be parsed in a periodic interval and published for other nodes to process and visualize
    if use_ros:
        global fig
        ros_init(data_generator=KittiSemanticDataGenerator(data_dir=data_dir), fig=fig, cfg=cfg)
    
    # Otherwise the data will be processed in a loop and saved to file
    else:
        global out_dir
        if not os.path.isdir(os.path.join(out_dir,"reduced_velo/")):
            os.mkdir(os.path.join(out_dir, "reduced_velo/"))
        if not os.path.isdir(os.path.join(out_dir, "gnd_labels/")):
            os.mkdir(os.path.join(out_dir, "gnd_labels/"))

        data_generator = KittiSemanticDataGenerator(data_dir=data_dir, logger=print)
        count = 0
        while data_generator.kitti_semantic_data_generate():
            cloud = extract_pc_in_box2d(data_generator.points, pc_range)

            # random sample point cloud to specified number of points
            cloud = random_sample_numpy(cloud, N = cfg.num_points)

            # Save data to file
            velo_path = os.path.join(out_dir, "reduced_velo/", "%06d" % count)
            label_path = os.path.join(out_dir, "gnd_labels/", "%06d" % count)
            np.save(velo_path,cloud)
            np.save(label_path, data_generator.gnd_label)

            count += 1


if __name__ == '__main__':
    main(data_dir)


# # @jit(nopython=True)
# def in_hull(p, hull):
#     if not isinstance(hull,Delaunay):
#         hull = Delaunay(hull)
#     return hull.find_simplex(p)>=0

# def extract_pc_in_box3d(pc, box3d):
#     ''' pc: (N,3), box3d: (8,3) '''
#     box3d_roi_inds = in_hull(pc[:,0:3], box3d)
#     return pc[box3d_roi_inds,:], box3d_roi_inds


# # @jit(nopython=True)
# def extract_pc_in_box2d(pc, box2d):
#     ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
#     box2d_corners = np.zeros((4,2))
#     box2d_corners[0,:] = [box2d[0],box2d[1]] 
#     box2d_corners[1,:] = [box2d[2],box2d[1]] 
#     box2d_corners[2,:] = [box2d[2],box2d[3]] 
#     box2d_corners[3,:] = [box2d[0],box2d[3]] 
#     box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
#     return pc[box2d_roi_inds,:]

