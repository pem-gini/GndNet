import open3d
import matplotlib.pyplot as plt
import numpy as np


def DisplayImage(input, colormap):
    plt.imshow(input, cmap=colormap)
    plt.show()

# Set camera intrinsic parmeters.
# intrinsic = open3d.camera.PinholeCameraIntrinsic()
# width (int) Width of the image.
# height (int) Height of the image.
# fx (float) X-axis focal length
# fy (float) Y-axis focal length.
# cx (float) X-axis principle point.
# cy (float) Y-axis principle point.
# intrinsic.set_intrinsics(850, 638, 790, 790, 0,0)
###default for sample data
intrinsic = open3d.camera.PinholeCameraIntrinsic(open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

images = [open3d.data.SampleSUNRGBDImage()] #, open3d.data.SampleNYURGBDImage()

# Visualize
vis = open3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width = 600, height = 400)

for i, image in enumerate(images):
    depth_raw = open3d.io.read_image(image.depth_path)
    DisplayImage(depth_raw, colormap=None)
    pcl = open3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic)
    pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(open3d.io.write_point_cloud("cloud%s.pcd"%i, pcl, write_ascii=True, compressed=False, print_progress=True))

    vis.add_geometry(pcl)
    vis.run()
    vis.destroy_window()