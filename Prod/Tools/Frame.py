import cv2
import numpy as np
from pyk4a import PyK4ACapture, PyK4APlayback

from Prod.Tools.Tools import Tools
from Prod.Tools.ObjectDetectionTools import ObjectDetectionFrame


class Frame:
    def __init__(self, capture: PyK4ACapture, info: PyK4APlayback, color, depth, ir, objframe: ObjectDetectionFrame):
        self.capture = capture
        self.info = info
        self.color = color
        self.depth = depth
        self.ir = ir
        self.objframe = objframe

    def check_frame(self):
        """check if there is a frame (is not None)"""
        return not (self.capture is None)

    def get_ir_image(self):
        """returns the IR image"""
        return self.depth

    def get_transformed_depth(self):
        """returns the transformed depth image"""
        return self.capture.transformed_depth

    def get_masked_point_cloud(self, _id):
        """returns the masked point cloud data , as a list with shape (N, 3)"""
        return self.capture.transformed_depth_point_cloud[self.objframe.get_mask(_id) == 127]

    def get_point_cloud(self):
        """returns the entire point cloud for a frame, as a list with shape(W, H, 3) -> (1080, 1920, 3).
         Use .reshape(-1, 3) for format (N, 3)"""
        return self.capture.transformed_depth_point_cloud

    def get_image(self):
        """returns the frame image"""
        return self.objframe.og_image

    def get_point_cloud_colors(self, _id):
        """returns the point cloud colors for open3d in the correct format,

         to use pointcloud.colors = o3d.utility.Vector3dVector(get_point_could_colors(id))"""
        _image = cv2.cvtColor(self.get_image(), cv2.COLOR_BGR2RGB)
        colored_points = _image[self.objframe.get_mask(_id) == 127]
        return colored_points.astype(np.float32) / 255.0

    def get_masked_image(self, _id):
        """returns the frame masked image"""
        return self.objframe.get_masked_og_img(_id)

    def show_yolo_image(self):
        """shows the yolo image #note uses the yolo API not opencv"""
        return self.objframe.result.show()

    def show_image(self):
        """shows the frame image"""
        cv2.imshow("Image", self.get_image())
        cv2.waitKey(int(1000/60))

    def show_masked_image(self, id):
        """shows the masked frame image"""
        cv2.imshow("Masked Image", self.get_masked_image(id))
        cv2.waitKey(int(1000/60))

    def show_transformed_depth(self):
        """shows the transformed depth image"""
        cv2.imshow("Transformed Depth", Tools.colorize(self.get_transformed_depth(), (None, 5000)))
        cv2.waitKey(int(1000/60))

    def show_ir_image(self):
        """shows the ir image"""
        cv2.imshow("IR Image", self.get_ir_image())
        cv2.waitKey(int(1000/60))
