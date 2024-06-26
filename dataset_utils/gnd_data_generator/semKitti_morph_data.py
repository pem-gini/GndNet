use_ros = False
visualize = False
export = False

import math
import multiprocessing
import time
import os
import yaml
import logging
import numpy as np
from numpy.linalg import inv

if use_ros:
    from semiKitti_ros_utils import ros_init

import dataset_generator_utils as gnd_utils

if visualize:
    import matplotlib.pyplot as plt


from scipy.spatial.transform import Rotation as R
# import cv2

from numba import jit
from scipy import signal, ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
#from torch.utils.data.sampler import SubsetRandomSampler


if visualize:
    plt.ion()

# with open('config/config.yaml') as f:
with open('../../config/config_kittiSem2.yaml') as f:
	config_dict = yaml.load(f, Loader=yaml.FullLoader)

class ConfigClass:
	def __init__(self, **entries):
		self.__dict__.update(entries)

cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use

# Init logging
logging.basicConfig()
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger_main = logging.getLogger().getChild('main')
logger_main.setLevel(logging.DEBUG)

fh = logging.FileHandler('dataprep.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_main.addHandler(fh)

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
    logger_main.error('Non-square voxels are not yet supported!')
    exit(-1)
voxel_size = voxel_dimensions[0]

logger_main.debug(grid_size)
logger_main.debug(type(grid_size))
logger_main.debug(voxel_size)
logger_main.debug(type(voxel_size))

data_dir = '/work/yw410445/dataset/sequences/'
out_dir = '/work/yw410445/training_data/sequences/'


if visualize:
    fig = plt.figure()
else:
    fig = None

def process_cloud(cloud):
    # start = time.time()
    # remove all non ground points; gnd labels = [40, 44, 48, 49]
    # gnd, obs = segment_cloud(cloud,[40, 44, 48, 49])
    # A complete list of all labels: https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml
    gnd, obs = gnd_utils.segment_cloud(cloud,[40, 44, 48, 49,60,72]) # ['road', 'parking', 'sidewalk', 'other-ground', 'lane-marking', 'terrain'] 
    global visualize, grid_size, voxel_size

    grid_size_np = np.asarray(grid_size)
    gnd_img = gnd_utils.lidar_to_img(np.copy(gnd), grid_size_np, voxel_size, fill = 1)
    gnd_heightmap, heightmap, num_points = gnd_utils.lidar_to_heightmap(np.copy(gnd), grid_size_np, voxel_size, max_points = 100)

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
    seg = gnd_utils.semantically_segment_cloud(cloud.copy(), grid_size_np, voxel_size, image_result)

    if visualize:
        fig.clear()

        fig.add_subplot(2, 3, 1)
        plt.imshow(gnd_img, interpolation='nearest')

        fig.add_subplot(2, 3, 4)
        cs = plt.imshow(gnd_heightmap, interpolation='nearest')
        fig.colorbar(cs)
        
        fig.add_subplot(2, 3, 5)
        cs = plt.imshow(interpolated_linear, interpolation='nearest')
        fig.colorbar(cs)

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
        self.calibration = gnd_utils.parse_calibration(os.path.join(self.data_dir, "calib.txt"))
        self.poses = gnd_utils.parse_poses(os.path.join(self.data_dir, "poses.txt"), self.calibration)

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
            self.logger.info(f'Frame {current_frame} with angle {self.angle}')

        self.points = gnd_utils.rotate_cloud(points, theta = [0,5,self.angle]) #zyx

        self.cloud, self.gnd_label, self.seg = process_cloud(points)
        # cloud = process_cloud(points)

        # points += np.array([0,0,lidar_height,0,0], dtype=np.float32)
        # points[0,2] += lidar_height

        global cfg, lidar_height
        if cfg.shift_cloud:
            self.points[:,2] += lidar_height
            self.cloud[:,2] += lidar_height

        return True


def main(logger: logging.Logger, data_dir: str, sequence_start=0, sequence_end=100, max_frames=None):
    global out_dir
    sequences = sorted(os.listdir(data_dir))

    framesCnt = 0
    sequenceCnt = 0

    for sequence in sequences:
        # Only compute sequences within the given range
        if not (sequence_start <= int(sequence) <= sequence_end):
            continue

        sequence_dir = os.path.join(data_dir, sequence)
        sequence_dir_out = os.path.join(out_dir, sequence)
        if not os.path.isdir(sequence_dir):
            continue

        # When using ROS, the data will be parsed in a periodic interval and published for other nodes to process and visualize
        if use_ros:
            global fig
            ros_init(data_generator=KittiSemanticDataGenerator(data_dir=sequence_dir), fig=fig, cfg=cfg)
        
        # Otherwise the data will be processed in a loop and saved to file
        else:
            os.makedirs(os.path.join(sequence_dir_out,"reduced_velo/"), exist_ok=True)
            os.makedirs(os.path.join(sequence_dir_out,"gnd_labels/"), exist_ok=True)

            data_generator = KittiSemanticDataGenerator(data_dir=sequence_dir, logger=logger.getChild(f'seq{sequence}'))
            count = 0
            while data_generator.kitti_semantic_data_generate():
                if count == 1:
                    start = time.time()

                cloud = gnd_utils.extract_pc_in_box2d(data_generator.points, pc_range)

                # random sample point cloud to specified number of points
                cloud = gnd_utils.random_sample_numpy(cloud, N = cfg.num_points)

                # Save data to file
                velo_path = os.path.join(sequence_dir_out, "reduced_velo/", "%06d" % count)
                label_path = os.path.join(sequence_dir_out, "gnd_labels/", "%06d" % count)
                np.save(velo_path,cloud)
                np.save(label_path, data_generator.gnd_label)

                count += 1

                # For debugging purposes it can help to limit the number of frames
                framesCnt += 1
                if max_frames != None and framesCnt >= max_frames:
                    logger.debug(time.time()-start)
                    return

if __name__ == '__main__':
    processes = []
    numCores = 4
    numSequences = 11

    lowerBound = 0
    seqPerProcess = math.ceil(numSequences / numCores)
    for i in range(numCores):
        end = min(numSequences, lowerBound+seqPerProcess)
        logger_thread = logger_main.getChild(f'process{i}')
        p = multiprocessing.Process(target=main, args=(logger_thread, data_dir, lowerBound, end-1, None))
        processes.append(p)

        logger_main.info(f'Start process {i} of {numCores}: {lowerBound}-{end}')
        p.start()

        lowerBound = end

    for process in processes:
        process.join()


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

