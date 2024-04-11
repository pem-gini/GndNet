import numpy as np

from scipy.spatial import Delaunay
from numba import njit, prange

# @jit(nopython=True)
def in_hull(p, hull):
	if not isinstance(hull,Delaunay):
		hull = Delaunay(hull)
	return hull.find_simplex(p)>=0

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

@njit
def compute_frustum(observer_position, observer_direction, fov_degrees, near_clip, far_clip, aspect_ratio=1.0):
    # Convert field of view from degrees to radians
    fov_radians = np.radians(fov_degrees)
    # Compute frustum basis vectors
    forward = observer_direction / np.linalg.norm(observer_direction)
    right = np.cross([0, 0, -1], forward)
    up = -np.cross(forward, right)
    # Compute frustum near and far plane centers
    near_center = observer_position + forward * near_clip
    far_center = observer_position + forward * far_clip
    # Compute frustum half extents based on field of view
    near_height = np.tan(fov_radians / 2) * near_clip
    near_width = near_height * aspect_ratio
    far_height = np.tan(fov_radians / 2) * far_clip
    far_width = far_height * aspect_ratio
    # Compute frustum right, left, top, and bottom planes
    near_frustum_right = right * near_width
    near_frustum_left = -near_frustum_right
    near_frustum_top = up * near_height
    near_frustum_bottom = -near_frustum_top
    far_frustum_right = right * far_width
    far_frustum_left = -far_frustum_right
    far_frustum_top = up * far_height
    far_frustum_bottom = -far_frustum_top
    # Compute frustum points for near and far planes
    near_top_left = near_center + near_frustum_top + near_frustum_left
    near_top_right = near_center + near_frustum_top + near_frustum_right
    near_bottom_left = near_center + near_frustum_bottom + near_frustum_left
    near_bottom_right = near_center + near_frustum_bottom + near_frustum_right
    far_top_left = far_center + far_frustum_top + far_frustum_left
    far_top_right = far_center + far_frustum_top + far_frustum_right
    far_bottom_left = far_center + far_frustum_bottom + far_frustum_left
    far_bottom_right = far_center + far_frustum_bottom + far_frustum_right
    # Represent frustum planes as tuples of three points
    near_plane = (near_top_left, near_top_right, near_bottom_left, near_bottom_right)
    far_plane = (far_top_right, far_top_left, far_bottom_right, far_bottom_left)

    top_plane = (near_top_left, far_top_left, far_top_right, near_top_right)
    right_plane = (near_top_right, far_top_right, far_bottom_right, near_bottom_right)
    bottom_plane = (near_bottom_right, far_bottom_right, far_bottom_left, near_bottom_left)
    left_plane = (near_bottom_left, far_bottom_left, far_top_left, near_top_left)
    
    ### frustrum normals
    return (near_plane, far_plane, top_plane, right_plane, bottom_plane, left_plane)

@njit
def compute_normal_vector(plane_points):
        line1 = plane_points[1] - plane_points[0]
        line2 = plane_points[2] - plane_points[0]
        normal = np.cross(line1, line2)
        ### normalize normal
        normal /= np.linalg.norm(normal)
        return normal

@njit
def is_point_in_frustum(point, frustum_planes):
    
    for plane_points in frustum_planes:
        normal_vector = compute_normal_vector(plane_points)
        # Choose any point from the plane to calculate the distance
        reference_point = plane_points[0]
        vector_to_point = point - reference_point
        # Check against the frustum plane
        dotP = np.dot(vector_to_point, normal_vector)
        if dotP < 0:
            return False
    return True

@njit # TODO: @njit(parallel=True), throwing some errors for now, but could potentially improve performance
def filter_points_by_frustum(points_3d, observer_position, observer_direction, fov, aspect_ratio, near, far):
    frustum_planes = compute_frustum(observer_position, observer_direction, fov, near, far, aspect_ratio)

    # Check if the point is inside the updated frustum
    num_points = len(points_3d)
    result = np.zeros(num_points, dtype=np.bool_)
    for i in prange(num_points):
        result[i] = is_point_in_frustum(points_3d[i], frustum_planes)

    return points_3d[result == 1]
