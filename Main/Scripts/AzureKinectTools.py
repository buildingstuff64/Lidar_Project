from datetime import datetime

import cv2
import keyboard
import open3d as o3d
from pyk4a import PyK4APlayback
from typing import Optional, Tuple

from Main.Scripts.Denoiser import Denoiser
from Main.Scripts.Frame import Frame
from Main.Scripts.ObjectDetectionTools import ObjectDetectionTools, ObjectDetectionFrame
from Main.Scripts.Tools import Tools


class AzureKinectTools:
    def __init__(self, path, progress_callback = None):
        self.frame_count = self.calc_frame_count(path)
        self.playback = PyK4APlayback(path)
        self.playback.open()
        self.calibration = self.playback.calibration
        self.frames = list()
        objects_ids_set = set()
        self.obj = ObjectDetectionTools(path)

        print(self.frame_count)
        i = 0
        while True:
            try:
                start_time = datetime.now()
                new_frame = self.playback.get_next_capture()

                if new_frame is None or new_frame.color is None or new_frame.depth is None:
                    continue

                colorBGR = cv2.imdecode(new_frame.color, cv2.IMREAD_COLOR)
                if colorBGR is None:
                    continue

                results = self.obj.model.track(source = colorBGR, show = False, device = 0, conf = 0.4, save = False, persist = True)
                print(len(results))
                for r in results:
                    for _id, _cls in zip(r.boxes.id.int().cpu().tolist(), r.boxes.cls.int().cpu().tolist()):
                        objects_ids_set.add((_id, r.names[_cls]))

                    #enhanced_img = cv2.cvtColor(self.denoiser.run_img(cv2.cvtColor(colorBGR, cv2.COLOR_BGR2RGB)), cv2.COLOR_RGB2BGR)
                    #cv2.imshow("Denoised Image", enhanced_img)
                    #cv2.waitKey(1)

                    self.frames.append(Frame(new_frame, self.playback, cv2.cvtColor(colorBGR, cv2.COLOR_BGR2BGRA), new_frame.depth, new_frame.ir, ObjectDetectionFrame(r), self))
                est_time = datetime.now() - start_time

                if progress_callback is not None:
                    i+=1
                    progress_callback(f"{i} / {self.frame_count} : avgtime {est_time.total_seconds()}s : estimated time left {est_time * (self.frame_count-i)}")

            except EOFError:
                print(f"End of File")
                break
            except Exception as e:
                print(f"!!! {e} !!!")

        print(f"Succesfully imported MKV file {path} \n frame count {self.frames.count}")
        self.object_ids = list(objects_ids_set)
        self.selected_ids = [0]


    @staticmethod
    def calc_frame_count(path):
        frame_counter = PyK4APlayback(path)
        frame_counter.open()
        frame_count = 0
        while True:
            try:
                f = frame_counter.get_next_capture()
                frame_count += 1
            except:
                print(f"End of File")
                break
        frame_counter.close()
        return frame_count

    def info(self):
        """Prints out mkv file and camera info"""
        print(self.playback.configuration)
        print(self.playback.calibration)

    def select_object(self):
        """Prompts user to select the required object"""
        print(f"objects found {self.object_ids}")
        str = input("select which object to track \n --> ")
        self.selected_ids = [int(x.strip()) for x in str.split(',')]
        print(self.selected_ids)

    def get_frame(self, index) -> Frame:
        return self.frames[index]

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
            masked_points = _frame.get_masked_point_cloud(self.selected_ids)

            pts = o3d.geometry.PointCloud()
            pts.points = o3d.utility.Vector3dVector(masked_points)
            _img = _frame.get_masked_image(self.selected_ids)
            colored_points = _img[_frame.objframe.get_mask(self.selected_ids) == 127]
            pts.colors = o3d.utility.Vector3dVector(_frame.get_point_cloud_colors(self.selected_ids))
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
            _f.show_masked_image(self.selected_ids)

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
    a = AzureKinectTools(Tools.getPath())



