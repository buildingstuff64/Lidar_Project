import cv2
import keyboard
import open3d as o3d
from pyk4a import PyK4APlayback
from typing import Optional, Tuple

from Prod.Tools.Frame import Frame
from Prod.Tools.ObjectDetectionTools import ObjectDetectionTools, ObjectDetectionFrame
from Prod.Tools.Tools import Tools
import numpy as np


class AzureKinectTools:
    def __init__(self, path):
        self.playback = PyK4APlayback(path)
        self.playback.open()
        self.calibration = self.playback.calibration
        self.frames = list()

        objects_ids_set = set()
        self.obj = ObjectDetectionTools(path)
        for r in self.obj.results:
            try:
                new_frame = self.playback.get_next_capture()
                if new_frame is None:
                    continue

                for _id, _cls in zip(r.boxes.id.int().cpu().tolist(), r.boxes.cls.int().cpu().tolist()):
                    objects_ids_set.add((_id, r.names[_cls]))

                self.frames.append(Frame(new_frame, self.playback, new_frame.color, new_frame.depth, new_frame.ir,
                                         ObjectDetectionFrame(r)))
            except EOFError:
                print(f"oh dear...")
                break

        print(f"Succesfully imported MKV file {path} \n frame count {self.frames.count}")
        self.object_ids = list(objects_ids_set)
        self.selected_id = 0

    def info(self):
        """Prints out mkv file and camera info"""
        print(self.playback.configuration)
        print(self.playback.calibration)

    def select_object(self):
        """Prompts user to select the required object"""
        print(f"objects found {self.object_ids}")
        self.selected_id = input("select which object to track \n --> ")

    def get_frame(self, index) -> Frame:
        return self.frames[0]

    def get_frames(self, _range: Tuple[Optional[int], Optional[int]] = (None, None)) -> list[Frame]:
        return self.frames[_range[0]:_range[1]]

    def show_masked_point_cloud_video(self):
        """Shows the masked point cloud using open3d"""
        pcd = o3d.geometry.PointCloud()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name = "Object Detection on Depth")
        vis.add_geometry(pcd)
        vis.reset_view_point(True)

        for _frame in self.get_frames():
            masked_points = _frame.get_masked_point_cloud(self.selected_id)

            pts = o3d.geometry.PointCloud()
            pts.points = o3d.utility.Vector3dVector(masked_points)
            _img = _frame.get_masked_image(self.selected_id)
            colored_points = _img[_frame.objframe.get_mask(self.selected_id) == 127]
            pts.colors = o3d.utility.Vector3dVector(_frame.get_point_cloud_colors(self.selected_id))
            pts.voxel_down_sample(0.05)

            vis.remove_geometry(pcd, False)
            pcd = pts
            vis.add_geometry(pcd, False)

            vis.poll_events()
            vis.update_renderer()

            if keyboard.is_pressed('q'):
                vis.reset_view_point(True)

    def show_images(self):
        """Shows the images from the mkv file"""
        for _f in self.get_frames():
            _f.show_image()

    def show_masked_images(self):
        """Shows the masked images, uses the selected_ID as reference"""
        for _f in self.get_frames():
            _f.show_masked_image(self.selected_id)

    def show_transformed_depth(self):
        """Shows the transformed depth images from the MKV file"""
        for _f in self.get_frames():
            _f.show_transformed_depth()

    def show_ir_images(self):
        """Shows the ir images from the MKV file"""
        for _f in self.get_frames():
            _f.show_ir_image()

    def show_depth(self):
        for _f in self.get_frames():
            _f.show_depth()

    def video_saver(self, name, function, size: Tuple[int, int]):
        """Saves a video of a view, function is the function name of the function you want to call e.g. _f.get_masked_image(_id)"""
        print(f"saving file --> {name}")
        vid = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 30, size)
        for _f in self.get_frames():
            eval(f'vid.write({function})')
        vid.release()
        print("save complete")


if __name__ == "__main__":
    # ObjectDetectionTools.run_single_image(Tools.getPaths(), True, 0.2)
    ak = AzureKinectTools(Tools.getPath())


