import numpy as np
import math
import copy

import numba as nb


###numby precompile
from numba.pycc import CC
cc = CC("transform_module")
cc.verbose = True

def euler_from_quaternion(q):
    return euler_from_qxqyqzqw(q.x, q.y, q.z, q.w)

@nb.njit
def euler_from_qxqyqzqw(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

@nb.njit
def quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return np.array([qx, qy, qz, qw])

@nb.njit
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3] ### qw
    q1 = Q[0] ### qx
    q2 = Q[1] ### qy
    q3 = Q[2] ### qz
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])                            
    return rot_matrix

@nb.njit
def translation_vector(x,y,z):
    return np.array([x,y,z])

def transformation_matrix(T, R):
    return np.array([
        [R[0][0],R[0][1],R[0][2], T[0]],
        [R[1][0],R[1][1],R[1][2], T[1]],
        [R[2][0],R[2][1],R[2][2], T[2]],
        [0,      0,      0,       1]
    ])

def transformStampedToTransformationMatrix(ts):
    T = np.array([ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z])
    Q = np.array([ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w])
    R = quaternion_rotation_matrix(Q)
    return transformation_matrix(T, R)

def transformPoseStamped(p, transform):
    v = [p.pose.position.x, p.pose.position.y, p.pose.position.z, 1.0]
    res = np.dot(transform, np.transpose(v))
    result = copy.deepcopy(p)
    result.pose.position.x = res[0]
    result.pose.position.y = res[1]
    result.pose.position.z = res[2]
    return result
def transformPose(p, transform):
    v = [p.position.x, p.position.y, p.position.z, 1.0]
    res = np.dot(transform, np.transpose(v))
    result = copy.deepcopy(p)
    result.position.x = res[0]
    result.position.y = res[1]
    result.position.z = res[2]
    return result

    
def transformPointStamped(p, transform):
    v = [p.point.x, p.point.y, p.point.z, 1.0]
    res = np.dot(transform, np.transpose(v))
    result = copy.deepcopy(p)
    result.point.x = res[0]
    result.point.y = res[1]
    result.point.z = res[2]
    return result

def transformPointList(points, x, y, z, rol,pit,yaw):
    Q = quaternion_from_euler(rol, pit, yaw)
    R = quaternion_rotation_matrix(Q)
    T = translation_vector(x, y, z)
    trafo = transformation_matrix(T,R)
    return [p.transform(trafo) for p in points]

### numba capable numpy impl without ros types
@nb.njit
def fastTransform3d(a, transform):
    v = np.append(a, 1.0)
    res = np.dot(transform, np.transpose(v))
    return res

@nb.njit
def transformCloud(points, trafo):
    for i,xyz in enumerate(points):
        # x,y,z = xyz
        # ps = geometryMsg.PointStamped()
        # ps.point.x = float(x)
        # ps.point.y = float(y)
        # ps.point.z = float(z)
        # transformed = transform.transformPointStamped(ps, trafo)
        transformed = fastTransform3d(xyz, trafo)
        points[i] = [transformed[0], transformed[1], transformed[2]]
    return points
