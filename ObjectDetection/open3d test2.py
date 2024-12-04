import open3d as o3d
import numpy as np

reader = o3d.io.AzureKinectMKVReader()
reader.open(r"C:\Users\bc-jo\Documents\Uni\Meng group project\Kinect\moving_person_output_3.mkv")
print(reader.is_opened())


glfw_key_escape = 256
glfw_key_space = 32
vis = o3d.visualization.VisualizerWithKeyCallback()
vis_geometry_added = False
vis.create_window('reader', 1920, 540)

print(
    "MKV reader initialized. Press [SPACE] to pause/start, [ESC] to exit."
)

while not reader.is_eof():
    color, depth = reader.next_frame()
    if rgbd is None:
        continue

    if not vis_geometry_added:
        vis.add_geometry(rgbd)
        vis_geometry_added = True



reader.close()