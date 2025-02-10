from idlelib.pyparse import trans
from tkinter.filedialog import askopenfilenames, askdirectory

import time

import keyboard
import open3d as o3d
import pyk4a
import cv2
from pyk4a import PyK4APlayback, ImageFormat, PyK4A, PyK4ACapture, CalibrationType
import numpy as np
from typing import Optional, Tuple

from Tools.ObjectDetectionTools import ObjectDetectionTools, ObjectDetectionFrame


class Tools:
    @staticmethod
    def getPath():
        return askopenfilenames(defaultextension = '.mkv')[0]

class AzureKinectTools:
    def __init__(self, path):
        self.playback = PyK4APlayback(path)
        self.playback.open()
        self.calibration = self.playback.calibration
        self.frames = list()

        obj = ObjectDetectionTools(path)
        for r in obj.getResults():
            try:
                f = self.playback.get_next_capture()
                if f is None:
                    continue
                self.frames.append(
                    Frame(f, self.playback, f.color, f.depth, f.ir, ObjectDetectionFrame(r)))
            except EOFError:
                print(f"Succesfully imported MKV file {path} \n frame count {self.frames.count}")
                break

        """i = 0
        while(1):
            x = obj.getFrameResult(i)
            i+=1
            try:
                f = self.playback.get_next_capture()
                if f is None:
                    continue
                self.frames.append(Frame(f, self.playback, f.color, f.depth, f.ir, obj.getFrameResults(self.frames.count)))
            except EOFError:
                print(f"Succesfully imported MKV file {path} \n frame count {self.frames.count}")
                break"""

    def info(self):
        print(self.playback.configuration)
        print(self.playback.calibration)

    def getFrame(self, index):
        return self.frames[0]

    def getFrames(self, range: Tuple[Optional[int], Optional[int]] = (None, None)):
        return self.frames[range[0]:range[1]]

    def colorize(image: np.ndarray,clipping_range: Tuple[Optional[int], Optional[int]] = (None, None), colormap: int = cv2.COLORMAP_HSV, ) -> np.ndarray:
        if clipping_range[0] or clipping_range[1]:
            img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
        else:
            img = image.copy()
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        img = cv2.applyColorMap(img, colormap)
        return img

    def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
        # examples for all possible pyk4a.ColorFormats
        if color_format == ImageFormat.COLOR_MJPG:
            color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
        elif color_format == ImageFormat.COLOR_NV12:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
            # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
            # h, w = color_image.shape[0:2]
            # h = h // 3 * 2
            # luminance = color_image[:h]
            # chroma = color_image[h:, :w//2]
            # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
        elif color_format == ImageFormat.COLOR_YUY2:
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
        return color_image


class Frame:
    def __init__(self, capture: PyK4ACapture, info: PyK4APlayback, color, depth, ir, objframe: ObjectDetectionFrame):
        self.capture = capture
        self.info = info
        self.color = color
        self.depth = depth
        self.ir = ir
        self.objframe = objframe


    def show(self, image_type = ["Depth", "Color", "IR", "Depth_T", "Color_T", "IR_T"]):
        if self.capture is None:
            return

        if "Depth" in image_type and self.capture.depth is not None:
            cv2.imshow("Depth", AzureKinectTools.colorize(self.depth, (None, 5000)))
        if "Color" in image_type and self.capture.color is not None:
            cv2.imshow("Color", AzureKinectTools.convert_to_bgra_if_required(self.info.configuration["color_format"], self.color))
        if "IR" in image_type and self.capture.ir is not None:
            cv2.imshow("IR", AzureKinectTools.colorize(self.ir, (None, 500), colormap=cv2.COLORMAP_JET))

        cv2.waitKey(int(1000/60))

    def get_transformed_depth(self):
        return self.capture.transformed_depth


if __name__ == "__main__":
    ak = AzureKinectTools(Tools.getPath())
    ak.info()

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name = "Object Detection on Depth")
    vis.add_geometry(pcd)
    vis.reset_view_point(True)

    for f in ak.getFrames():
        """for x in f.objframe.mask:
            for y in x:
                if y != 0:
                    print(y)"""



        masked_points = f.capture.transformed_depth_point_cloud[f.objframe.mask == 127]
        print(f"{f.capture.transformed_depth_point_cloud.shape} , {f.objframe.mask.shape}, {masked_points.shape}")

        """t = time.time()
        #masked_img = f.get_transformed_depth()[f.objframe.mask.astype(bool)]
        print(f"start {time.time() - t}")
        masked_img = cv2.bitwise_and(f.capture.transformed_depth_point_cloud, f.capture.transformed_depth_point_cloud, mask = f.objframe.mask)
        print(f"bitwise {time.time() - t}")
        masked_img = masked_img.copy()
        print(f"copy {time.time() - t}")
        masked_img = masked_img.reshape(-1, 3)
        realPoints = list()
        for p in f.capture.transformed_depth_point_cloud.reshape(-1, 3):
            if p[0] != 0 and p[1] != 0 and p[2] != 0:
                realPoints.append(p)"""

        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(masked_points)
        pts.voxel_down_sample(0.05)

        vis.remove_geometry(pcd, False)
        pcd = pts
        vis.add_geometry(pcd, False)


        vis.poll_events()
        vis.update_renderer()

        if keyboard.is_pressed('q'):
            vis.reset_view_point(True)


        #cv2.imshow("masked_image", f.objframe.mask)
        #cv2.waitKey(int(1000 / 60))


    """
    pcd = o3d.geometry.PointCloud()
    for f in ak.getFrames():
        if f.get_transformed_depth() is not None:
            print(f.capture.transformed_depth.shape)
            cv2.imshow("Transformed depth", AzureKinectTools.colorize(f.capture.transformed_depth[0:500, 0:1000], (None, 5000)))
            cv2.waitKey(int(1000/60))


            points = f.capture.transformed_depth_point_cloud[0:500, 0:1000, : ].reshape(-1, 3)
            realPoints = list()
            for p in points:
                if p[0] != 0 and p[1] != 0 and p[2] != 0:
                    realPoints.append(p)

            pcd.points = o3d.utility.Vector3dVector(realPoints)
            o3d.visualization.draw_geometries([pcd])
    """
    #ak.playback.close()



