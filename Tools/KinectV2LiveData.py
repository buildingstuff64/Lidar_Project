#from time import sleep
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from numpy.array_api import trunc
from open3d.cpu.pybind.t.geometry import PointCloud
from pykinect2024 import PyKinectRuntime
from pykinect2024.PyKinect2024 import FrameSourceTypes_Color, FrameSourceTypes_Infrared, FrameSourceTypes_Depth
import numpy as np
import keyboard

np.set_printoptions(threshold=np.inf)

kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color | FrameSourceTypes_Infrared | FrameSourceTypes_Depth)

intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = 512,
            height = 424,
            fx = 366.1,
            fy = 366.1,
            cx = 258.2,
            cy = 204.2
        )
vis_pt = o3d.visualization.VisualizerWithKeyCallback()
vis_pt.create_window(window_name="Kinect V2 Point Cloud")
vis_vx = o3d.visualization.VisualizerWithKeyCallback()
vis_vx.create_window(window_name="Kinect V2 Voxel")

pcd = o3d.geometry.PointCloud().voxel_down_sample(0.05)
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size = 0.05)


vis_pt.add_geometry(voxel_grid)
vis_pt.reset_view_point(True)

vis_vx.add_geometry(pcd)
vis_vx.reset_view_point(True)

while True:
    frame_color = kinect.get_last_color_frame()
    frame_color = np.reshape(frame_color, (1080, 1920, 4)).astype(np.uint8)
    #cv2.imshow('color', frame_color)

    frame_ir = kinect.get_last_infrared_frame()
    frame_ir = np.reshape(frame_ir, (kinect.infrared_frame_desc.Height, kinect.infrared_frame_desc.Width)).astype(
        np.uint16)
    #cv2.imshow('ir', frame_ir)


    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()
        frameD = kinect._depth_frame_data
        frame = frame.astype(np.uint16)
        frame = np.reshape(frame, (424, 512))
        frame_cv = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        depth_o3d = o3d.geometry.Image(frame)

        new_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth = depth_o3d,
            intrinsic = intrinsics,
            depth_scale = 1000.0,  # Kinect v2 depth is in millimeters
            depth_trunc = 10.0,  # Truncate depth values at 10 meters
            stride = 1
        )

        depth_values = frame.flatten()
        valid_mask = (depth_values > 0) & ~np.isnan(depth_values)

        # Normalize depth values to [0, 1] and apply colormap
        depth_values_normalized = (depth_values - np.min(depth_values[valid_mask])) / (
                np.ptp(depth_values[valid_mask]) + 1e-8
        )
        colormap = plt.get_cmap("viridis")
        colors = colormap(depth_values_normalized)[:, :3]  # Extract RGB

        # Filter colors for valid points
        filtered_colors = colors[valid_mask]
        new_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

        # Create a mask for valid depth points
        valid_mask = (depth_values > 0) & ~np.isnan(depth_values)

        # Filter colors using the valid mask
        filtered_colors = colors[valid_mask]

        # Assign the filtered colors to the point cloud
        new_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)


        pt = new_point_cloud.voxel_down_sample(0.025)
        new_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(new_point_cloud,0.025)

        vis_pt.remove_geometry(pcd, False)
        pcd = pt
        vis_pt.add_geometry(pcd, False)

        vis_vx.remove_geometry(voxel_grid, False)
        voxel_grid = new_voxel_grid
        vis_vx.add_geometry(voxel_grid, False)

        if keyboard.is_pressed('r'):
            o3d.visualization.draw_geometries([voxel_grid])
            break


        #vis.update_geometry(voxel_grid)
        #vis.clear_geometries()  # Clear previous frame
        #vis.add_geometry(voxel_grid)  # Add the new voxelized point cloud

        # Step 3: Update the visualizer window
        vis_pt.poll_events()
        vis_pt.update_renderer()

        vis_vx.poll_events()
        vis_vx.update_renderer()

        if keyboard.is_pressed('q'):
            vis_pt.reset_view_point(True)
            vis_vx.reset_view_point(True)

        if keyboard.is_pressed('e'):
            break

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
            if event == cv2.EVENT_RBUTTONDOWN:
                Pixel_Depth = frameD[((y * 512) + x)]
                print(Pixel_Depth)


        ##output = cv2.bilateralFilter(output, 1, 150, 75)
        #cv2.imshow('KINECT Video Stream', frame_cv)
        #cv2.setMouseCallback('KINECT Video Stream', click_event)
        output = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
print("hello")