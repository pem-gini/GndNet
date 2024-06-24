
"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



from __future__ import print_function
from __future__ import division

import argparse
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import shutil
import time
import yaml
import torch
from torchvision import datasets, transforms
# from utils.point_cloud_ops_test import points_to_voxel

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point

import gnd_net.ros2_numpy.ros2_numpy as ros2_numpy
import shapely.geometry
from scipy.spatial import Delaunay

import numba
from numba import jit,types


def gnd_marker_pub(ros_node, gnd_label, marker_pub, cfg, color = "red", frame_id = 'map'):
    length = int(cfg.grid_range[2] - cfg.grid_range[0]) # x direction
    width = int(cfg.grid_range[3] - cfg.grid_range[1])    # y direction

    gnd_marker = Marker()
    gnd_marker.header.frame_id = frame_id
    gnd_marker.header.stamp = ros_node.get_clock().now().to_msg()
    gnd_marker.type = gnd_marker.LINE_LIST
    gnd_marker.action = gnd_marker.ADD
    gnd_marker.scale.x = 0.05
    gnd_marker.scale.y = 0.05
    gnd_marker.scale.z = 0.05
    if isinstance(color, tuple):
        gnd_marker.color.a = 0.5
        r,g,b = color
        gnd_marker.color.r = float(r)
        gnd_marker.color.g = float(g)
        gnd_marker.color.b = float(b)
    else:
        if(color == "red"):
            gnd_marker.color.a = 0.5
            gnd_marker.color.r = 1.0
            gnd_marker.color.g = 0.0
            gnd_marker.color.b = 0.0
        if(color == "green"):
            gnd_marker.color.a = 0.5
            gnd_marker.color.r = 0.0
            gnd_marker.color.g = 1.0
            gnd_marker.color.b = 0.0
    gnd_marker.points = []

    # gnd_labels are arranged in reverse order
    x_step = length / gnd_label.shape[0]
    y_step = width / gnd_label.shape[1]
    for j in range(gnd_label.shape[0]):
        for i in range(gnd_label.shape[1]):
            pt1 = Point()
            pt1.x = float(i*x_step + cfg.grid_range[0])
            pt1.y = float(j*y_step + cfg.grid_range[1])
            pt1.z = float(gnd_label[j,i])

            if j>0 :
                pt2 = Point()
                pt2.x = float(i*x_step + cfg.grid_range[0])
                pt2.y = float((j-1)*y_step +cfg.grid_range[1])
                pt2.z = float(gnd_label[j-1, i])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if i>0 :
                pt2 = Point()
                pt2.x = float((i-1)*x_step + cfg.grid_range[0])
                pt2.y = float(j*y_step + cfg.grid_range[1])
                pt2.z = float(gnd_label[j, i-1])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if j < width-1 :
                pt2 = Point()
                pt2.x = float(i*x_step + cfg.grid_range[0])
                pt2.y = float((j+1)*y_step + cfg.grid_range[1])
                pt2.z = float(gnd_label[j+1, i])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if i < length-1 :
                pt2 = Point()
                pt2.x = float((i+1)*x_step + cfg.grid_range[0])
                pt2.y = float(j*y_step + cfg.grid_range[1])
                pt2.z = float(gnd_label[j, i+1])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

    marker_pub.publish(gnd_marker)

def np2ros_pub(ros_node, points, pcl_pub, timestamp = None):
    npoints = points.shape[0] # Num of points in PointCloud
    points_arr = np.zeros((npoints,), dtype=[
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('r', np.uint8),
                                        ('g', np.uint8),
                                        ('b', np.uint8)])
    points = np.transpose(points)
    points_arr['x'] = points[0]
    points_arr['y'] = points[1]
    points_arr['z'] = points[2]
    points_arr['r'] = 255
    points_arr['g'] = 255
    points_arr['b'] = 255

    if timestamp == None:
        timestamp = ros_node.get_clock().now().to_msg()
    cloud_msg = ros2_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "map")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)
    # rospy.sleep(0.1)


def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
    
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
    
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb


def np2ros_pub_2(ros_node, points, pcl_pub, timestamp, color, frame_id = "map"):
    npoints = points.shape[0] # Num of points in PointCloud
    points_arr = np.zeros((npoints,), dtype=[
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('intensity', np.float32)])
    points = np.transpose(points)
    points_arr['x'] = points[0]
    points_arr['y'] = points[1]
    points_arr['z'] = points[2]

    # float_rgb = rgb_to_float(color)
    points_arr['intensity'] = color
    # points_arr['g'] = 255
    # points_arr['b'] = 255

    if timestamp == None:
        timestamp = ros_node.get_clock().now().to_msg()
    cloud_msg = ros2_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id=frame_id)
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)

def np2ros_pub_2_no_intensity(ros_node, points, pcl_pub, frame_id = "map"):
    npoints = points.shape[0] # Num of points in PointCloud
    points_arr = np.zeros((npoints,), dtype=[
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32)])
    points = np.transpose(points)
    points_arr['x'] = points[0]
    points_arr['y'] = points[1]
    points_arr['z'] = points[2]

    timestamp = ros_node.get_clock().now().to_msg()
    cloud_msg = ros2_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id=frame_id)
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)