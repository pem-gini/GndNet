import sys
sys.path.append("../..") # Adds higher directory to python modules path.

import numpy as np
import rclpy
from tf_transformations import quaternion_from_matrix
import tf2_ros as tf2
from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, TransformStamped
import ros2_numpy.ros2_numpy as ros2_numpy

grid_size = [0,0,0,0]
width = 0
length = 0

def np2ros_pub(points, pcl_pub, timestamp):
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
    points_arr['intensity'] = points[3]
    # points_arr['g'] = 255
    # points_arr['b'] = 255

    cloud_msg = ros2_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "map")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)


def np2ros_pub_2(points, pcl_pub, timestamp, color):
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
    points_arr['intensity'] = color[0]
    # points_arr['g'] = 255
    # points_arr['b'] = 255

    cloud_msg = ros2_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "map")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)


def gnd_marker_pub(gnd_label, marker_pub,timestamp):
    global grid_size, length, width

    gnd_marker = Marker()
    gnd_marker.header.frame_id = "map"
    gnd_marker.header.stamp = timestamp
    gnd_marker.type = gnd_marker.LINE_LIST
    gnd_marker.action = gnd_marker.ADD
    gnd_marker.scale.x = 0.05
    gnd_marker.scale.y = 0.05
    gnd_marker.scale.z = 0.05
    gnd_marker.color.a = 0.1
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
            pt1.x = float(i*x_step + grid_size[0])
            pt1.y = float(j*y_step + grid_size[1])
            pt1.z = float(gnd_label[j,i])

            if j>0 :
                pt2 = Point()
                pt2.x = float(i*x_step + grid_size[0])
                pt2.y = float((j-1)*y_step +grid_size[1])
                pt2.z = float(gnd_label[j-1, i])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if i>0 :
                pt2 = Point()
                pt2.x = float((i-1)*x_step + grid_size[0])
                pt2.y = float(j*y_step + grid_size[1])
                pt2.z = float(gnd_label[j, i-1])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if j < width-1 :
                pt2 = Point()
                pt2.x = float(i*x_step + grid_size[0])
                pt2.y = float((j+1)*y_step + grid_size[1])
                pt2.z = float(gnd_label[j+1, i])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if i < length-1 :
                pt2 = Point()
                pt2.x = float((i+1)*x_step + grid_size[0])
                pt2.y = float(j*y_step + grid_size[1])
                pt2.z = float(gnd_label[j, i+1])
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

    marker_pub.publish(gnd_marker)

def get_transformation(translation, rotation, timestamp, child_frame_id, header_frame_id):
    """Returning a ros2 tf2 transformation object based on the old ros arguments"""
    tr = TransformStamped()

    tr.header.stamp = timestamp
    tr.header.frame_id = header_frame_id
    tr.child_frame_id = child_frame_id
    tr.transform.translation.x = float(translation[0])
    tr.transform.translation.x = float(translation[1])
    tr.transform.translation.x = float(translation[2])
    tr.transform.rotation.x = float(rotation[0])
    tr.transform.rotation.y = float(rotation[1])
    tr.transform.rotation.z = float(rotation[2])
    tr.transform.rotation.w = float(rotation[3])

    return tr


def broadcast_TF(node, pose, timestamp):
    quat = quaternion_from_matrix(pose)
    br = tf2.TransformBroadcaster(node=node)
    br.sendTransform(get_transformation((pose[0,3], pose[1,3], pose[2,3]), quat,
                     timestamp,
                     "/kitti/zoe_odom_origin",
                     "map"))

    # br.sendTransform((1.5, 0, 1.732), (0,0,0,1),
    # br.sendTransform((1.5, 0, 2.1), (0,0,0,1),
    br.sendTransform(get_transformation((-0.27, 0, 1.73), (0,0,0,1),
                     timestamp,
                     "/kitti/velo_link",
                     "map"))

    # br.sendTransform((3.334, 0, 0.34), (0,0,0,1),
    br.sendTransform(get_transformation((2.48, 0, 0), (0,0,0,1),
                     timestamp,
                     "/kitti/base_link",
                     "map"))
    

class KittiSemanticDataGeneratorNode(rclpy.node.Node):

    def __init__(self, data_generator, fig) -> None:
        super().__init__('gnd_data_provider')

        self.data_generator = data_generator
        data_generator.logger = self.get_logger().info

        self.pcl_pub = self.create_publisher(PointCloud2, "/kitti/velo/pointcloud", 10)
        self.pcl_pub2 = self.create_publisher(PointCloud2, "/kitti/raw/pointcloud", 10)
        self.marker_pub = self.create_publisher(Marker, "/kitti/ground_marker", 10)

        self.timer = self.create_timer(0.1, self.kitti_semantic_data_generate)

        if fig != None:
            self.create_timer(0.05, fig.canvas.flush_events)

    def kitti_semantic_data_generate(self):
        hasNextFrame = self.data_generator.kitti_semantic_data_generate()

        if not hasNextFrame:
            self.get_logger().info('Stop timer')
            self.timer.cancel()
        
        else:
            timestamp = self.get_clock().now().to_msg()
            #broadcast_TF(self,self.poses[current_frame],timestamp)
            #np2ros_pub(cloud, self.pcl_pub, timestamp)
            np2ros_pub_2(self.data_generator.points, self.pcl_pub2, timestamp, self.data_generator.seg.T)
            
            gnd_marker_pub(self.data_generator.gnd_label, self.marker_pub, timestamp)
            # print(points.shape)
            # pdb.set_trace()

def ros_init(data_generator, fig, cfg):
    global grid_size, length, width
    grid_size = cfg.grid_range
    length = int(grid_size[2] - grid_size[0]) # x direction
    width = int(grid_size[3] - grid_size[1])    # y direction

    rclpy.init(args=sys.argv)
    node = KittiSemanticDataGeneratorNode(data_generator, fig)
    rclpy.spin(node)
    rclpy.shutdown()