from open3d.cpu.pybind.visualization import draw_geometries

from Dev.Tools.MKVTools import MKVTools
import open3d as o3d
import numpy as np



# new_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
#     depth = depth_o3d,
#     intrinsic = intrinsics,
#     depth_scale = 1000.0,  # Kinect v2 depth is in millimeters
#     depth_trunc = 10.0,  # Truncate depth values at 10 meters
#     stride = 1
# )

m = MKVTools()
file = m.readfile()
frame = file.get_frame(0)
depth = np.reshape(frame.depth, (576, 640)).astype(np.uint8)

intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = 576,
            height = 640,
            fx = 366.1,
            fy = 366.1,
            cx = 258.2,
            cy = 204.2
        )

depth_o3d = o3d.geometry.Image(depth)
new_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth = depth_o3d,
            intrinsic = intrinsics
        )

draw_geometries([new_point_cloud])