use_ros = False
show_ros_intermediate_steps = False
visualize = False
export = False

import sys
sys.path.insert(1, '../../..')

from queue import Queue
import math
import concurrent.futures
import time
import os
import yaml
import logging
import numpy as np


if use_ros:
    from semiKitti_ros_utils import ros_init

import gnd_net.dataset_utils.gnd_data_generator.dataset_generator_utils as gnd_utils

if visualize:
    import matplotlib.pyplot as plt

from gnd_net.dataset_utils.gnd_data_generator.dataset_augmentation import AugmentationConfig, DataAugmentation

# import cv2

from numba import jit
from scipy import signal, ndimage
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
#from torch.utils.data.sampler import SubsetRandomSampler


if visualize:
    plt.ion()

# with open('config/config.yaml') as f:
with open('../../config/config_camera.yaml') as f:
	config_dict = yaml.load(f, Loader=yaml.FullLoader)

class ConfigClass:
	def __init__(self, **entries):
		self.__dict__.update(entries)

cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use
data_generator_ref = None

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
grid_size = np.array(cfg.grid_range)
length = int(grid_size[2] - grid_size[0]) # x direction
width = int(grid_size[3] - grid_size[1])    # y direction
lidar_height = cfg.lidar_height

voxel_dimensions = cfg.voxel_size
if voxel_dimensions[0] != voxel_dimensions[1]:
    logger_main.error('Non-square voxels are not yet supported!')
    exit(-1)
voxel_size = voxel_dimensions[0]

data_dir = cfg.data_dir
out_dir = cfg.out_dir

if visualize:
    fig = plt.figure()
else:
    fig = None

def gnd_plane_step_visualizer(i, filled_voxels, diff_to_average, outliers, gnd_heightmap, interpolated_linear, image_result, cloud):
    if i > 0 and show_ros_intermediate_steps:
        time.sleep(10)

        fig.clear()

        fig.add_subplot(2, 3, 1)
        cs = plt.imshow(filled_voxels, interpolation='nearest')
        fig.colorbar(cs)

        fig.add_subplot(2, 3, 2)
        cs = plt.imshow(diff_to_average, interpolation='nearest')
        fig.colorbar(cs)

        fig.add_subplot(2, 3, 3)
        cs = plt.imshow(outliers, interpolation='nearest')
        fig.colorbar(cs)

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
    
    # This is definitely not a nice way to handle this, but it works for debugging purposes
    if show_ros_intermediate_steps and data_generator_ref != None:
        cloud2 = cloud.copy()
        seg = gnd_utils.semantically_segment_cloud(cloud2.copy(), grid_size, voxel_size, image_result, lidar_height)
        cloud2[:,2] += lidar_height
        data_generator_ref.node.show_step(cloud2, seg, image_result.T)


class KittiSemanticDataGenerator():

    def __init__(self, data_dir, first_frame=0, last_frame=1e6, step=1, logger=None) -> None:

        self.data_dir = data_dir
        self.step = step
        self.last_frame = last_frame
        self.velodyne_dir = os.path.join(self.data_dir, "velodyne/")
        self.label_dir = os.path.join(self.data_dir, 'labels/')

        self.calibration = gnd_utils.parse_calibration(os.path.join(self.data_dir, "calib.txt"))
        self.poses = gnd_utils.parse_poses(os.path.join(self.data_dir, "poses.txt"), self.calibration)

        self.logger = logger

        self.angle = 0
        self.increase = True
        self.current_frame = first_frame

        self.cloud = None
        self.points = None
        self.gnd_label = None
        self.seg = None

        self.augmentationConfig = AugmentationConfig(
            grid=grid_size,
            voxel_size=voxel_size,
            keep_original=cfg.keep_original,
            num_rotations=cfg.num_rotations,
            num_height_var = cfg.num_height_var,
            num_noise_aug= cfg.num_noise_var,
            maxFrontSlope = cfg.maxFrontSlope, 
            maxSideTild = cfg.maxSideTild, 
            maxRotation = cfg.maxRotation, 
            maxHeight = cfg.maxHeight,
            noise_coefficient_top = cfg.noise_coefficient_top,
            noise_coefficient_bottom = cfg.noise_coefficient_bottom,
            noise_min_distance = cfg.noise_min_distance,
            noise_density_top = cfg.noise_density_top,
            noise_density_bottom = cfg.noise_density_bottom,
        )
        self.augmentation = DataAugmentation(self.augmentationConfig)

        # All frames that are already loaded into memory
        self.loaded_frames = Queue()
        self.complete = False
        self.doneLoading = False

    def get_next_frame(self):
        # If there are no new frames loaded, load the next ones
        while self.loaded_frames.empty():
            if not self.kitti_semantic_data_generate(): # If we already reached the last one, return None
                self.complete = True
                return (None, None, None)
           
        return self.loaded_frames.get()
    
    def load_future_frame(self, max_buffer = 16):
        if self.loaded_frames.qsize() < max_buffer and not self.doneLoading:
            self.kitti_semantic_data_generate()
    
    def kitti_semantic_data_generate(self):
        global cfg, lidar_height
        current_frame = self.current_frame

        # Increase the frame cnt and cancle timer if we reached the end
        self.current_frame += self.step
        if self.current_frame > self.last_frame:
            self.doneLoading = True
            return False

        points_path = os.path.join(self.velodyne_dir, "%06d.bin" % current_frame)
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)[:,:3] # Only load the x,y,z values and disregard the reflectiveness

        label_path = os.path.join(self.label_dir, "%06d.label" % current_frame)
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        if label.shape[0] == points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
        if label.shape[0] != points.shape[0]:
            self.logger.warning(f'Number of points and labels to not match! ({label.shape[0]} vs {points.shape[0]})')#
            return True # Just skip this after printing the warning
        label = np.expand_dims(label, axis = 1)
        points = np.concatenate((points,label), axis = 1)

        # Makes all necessary augmentations, and will return the frame as multiple differently augmented frames
        augmentations = self.augmentation.getAugmentedData(points.reshape((1,)+points.shape))
        num_augmentations = augmentations.shape[0]
        for i in range(num_augmentations):

            if self.logger != None:
                self.logger.info(f'Frame {current_frame} with augmentation {i+1}/{num_augmentations}')

            # Get all points that are labels as ground and compute the ground plane 
            visualizer = gnd_plane_step_visualizer if visualize else None
            gnd_pcl, gnd_plane = gnd_utils.compute_ground_plane(augmentations[i], grid_size, voxel_size, lidar_height, export_intermediate_step=export, logger=self.logger, visualizer=visualizer)

            gnd_plane_t = gnd_plane.T
            # Add noise to the pointcloud 
            augmentation = self.augmentation.addNoise(augmentations[i].reshape((1,)+augmentations[i].shape), gnd_plane_t.reshape((1,)+gnd_plane_t.shape))[0]

            # Segment the entire cloud into ground, obstacle and out of bound
            seg = gnd_utils.semantically_segment_cloud(augmentation.copy(), grid_size, voxel_size, gnd_plane, lidar_height)

            # points += np.array([0,0,lidar_height,0,0], dtype=np.float32)
            # points[0,2] += lidar_height

            if cfg.shift_cloud:
                augmentation[:,2] += lidar_height
                #self.cloud[:,2] += lidar_height
            
            self.loaded_frames.put((augmentation, gnd_plane_t, seg))

        return True

def compute_extract(logger: logging.Logger, data_dir: str, sequence=0, first_frame=0, last_frame=0, block_id=0, step=1):
    logger.info(f'Start to compute block {block_id}: {sequence}: {first_frame}-{last_frame}')
    start = time.time()

    sequence_dir = os.path.join(data_dir, sequence)
    sequence_dir_out = os.path.join(out_dir, sequence)
    if not os.path.isdir(sequence_dir):
        logger.error(f'Sequence directory does not exist: {sequence_dir}')
        logger.info(f'Failed to compute block {block_id}')
        return (False, time.time()-start)

    # When using ROS, the data will be parsed in a periodic interval and published for other nodes to process and visualize
    if use_ros:
        global fig, data_generator_ref
        data_generator_ref = KittiSemanticDataGenerator(data_dir=sequence_dir)
        ros_init(data_generator=data_generator_ref, fig=fig, cfg=cfg)
    
    # Otherwise the data will be processed in a loop and saved to file
    else:
        os.makedirs(os.path.join(sequence_dir_out,"reduced_velo/"), exist_ok=True)
        os.makedirs(os.path.join(sequence_dir_out,"gnd_labels/"), exist_ok=True)

        start = time.time()
        data_generator = KittiSemanticDataGenerator(sequence_dir, first_frame, last_frame, step, logger=logger.getChild(f'seq{sequence}-{block_id}'))
        count = math.ceil(first_frame / step) * data_generator.augmentationConfig.num_augmentations
        while not data_generator.complete:
            points, gnd_net, seg = data_generator.get_next_frame()
                
            if data_generator.complete:
                break

            cloud = gnd_utils.extract_pc_in_box2d(points, pc_range)

            # random sample point cloud to specified number of points
            cloud = gnd_utils.random_sample_numpy(cloud, N = cfg.num_points)

            # Save data to file
            velo_path = os.path.join(sequence_dir_out, "reduced_velo/", "%06d" % count)
            label_path = os.path.join(sequence_dir_out, "gnd_labels/", "%06d" % count)

            try:
                np.save(velo_path, cloud)
                np.save(label_path, gnd_net)
            except Exception as e:
                logger.error(e)
                logger.info(f'Failed to compute block {block_id}')
                return (False, time.time()-start)

            count += 1
    
    duration = time.time()-start
    logger.info(f'Completed next block: {block_id} ({duration}s)')
    return (True, duration)

def main(logger: logging.Logger, data_dir: str, step=1):
    global out_dir, cfg
    sequences = sorted(os.listdir(data_dir))[:11]
    num_workers = cfg.num_workers

    frames_per_block = cfg.frames_per_block * step

    data_blocks: list[tuple[str]] = []

    # Do not create process pool when using ros
    if use_ros:
        for sequence in sequences:
            frames_cnt = len(os.listdir(os.path.join(data_dir, sequence, "velodyne/")))
            compute_extract(logger, data_dir, sequence, 0, frames_cnt-1, 0, step)
        return

    for sequence in sequences:
        frames_cnt = len(os.listdir(os.path.join(data_dir, sequence, "velodyne/")))
        cnt = 0
        num_blocks = 0
        while cnt < frames_cnt-1:
            is_last = (cnt + 1.5 * frames_per_block) >= frames_cnt
            num_blocks += 1
            if is_last:
                data_blocks.append((sequence, cnt, frames_cnt-1))
                cnt = frames_cnt-1
                break
            else:
                data_blocks.append((sequence, cnt, cnt+frames_per_block-1))
                cnt = cnt+frames_per_block
        
        logger.info(f'Split sequence {sequence} into {num_blocks} blocks')
    logger.info(f'Created {len(data_blocks)} blocks for {len(sequences)} sequence')

    num_blocks = len(data_blocks)
    total_time = 0
    completed = 0 # Including failures
    failed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_extract, logger, data_dir, block[0], block[1], block[2], i, step) for i, block in enumerate(data_blocks)]

        start = time.time()
        for future in concurrent.futures.as_completed(futures):
            status, duration = future.result()

            if completed == 0:
                start = time.time() # Ignore the first finished package, it takes way longer because of compilation times

            total_time += duration
            completed += 1
            if not status:
                failed += 1

            avg_time = (time.time()-start)/completed
            time_left = avg_time * (num_blocks - completed)
            logger.info(f'Status: {completed/num_blocks:.0%} ({completed}/{num_blocks}): {time_left/60:.2f}min remaining ({total_time/completed:.2f}s per block)')

if __name__ == '__main__':
    main(logger_main, data_dir, step=cfg.frame_step)


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

