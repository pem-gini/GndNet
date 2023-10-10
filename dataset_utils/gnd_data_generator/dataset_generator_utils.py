from numba import jit,types
from functools import reduce
from scipy.spatial import Delaunay
import torch
import numpy as np

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

