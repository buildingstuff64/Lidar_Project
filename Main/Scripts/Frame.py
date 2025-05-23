import cv2
import numpy as np
from pyk4a import PyK4ACapture, PyK4APlayback
from sympy import false

#from Main.Scripts.AzureKinectTools import AzureKinectTools
from Main.Scripts.Denoiser import Denoiser
from Main.Scripts.FileManager import SCUNET_PATH
from Main.Scripts.Tools import Tools
from Main.Scripts.ObjectDetectionTools import ObjectDetectionFrame
import open3d as o3d


class Frame:
    def __init__(self, capture: PyK4ACapture, info: PyK4APlayback, color, depth, ir, objframe: ObjectDetectionFrame, ak):
        self.capture = capture
        self.info = info
        self.color = color
        self.depth = depth
        self.ir = ir
        self.objframe = objframe
        self.ak = ak
        self.denoised = None

    def rerun_with_denoise(self):
        denoiser = Denoiser(SCUNET_PATH.resolve())
        den_img = denoiser.run_img(cv2.cvtColor(self.color, cv2.COLOR_BGRA2RGB))

        results = self.ak.obj.model.track(source = cv2.cvtColor(den_img, cv2.COLOR_RGB2BGR), show = False, device = 0, conf = 0.4, save = False, persist = False)
        for r in results:
            self.objframe = ObjectDetectionFrame(r)

        den_img = cv2.cvtColor(den_img, cv2.COLOR_RGB2BGRA)
        self.denoised = den_img


    def check_frame(self):
        """check if there is a frame (is not None)"""
        return not (self.capture is None)

    def get_ir_image(self):
        """returns the IR image"""
        return self.ir

    def get_depth(self):
        """returns the Depth image"""
        return self.capture.depth

    def get_transformed_depth(self):
        """returns the transformed depth image"""
        return self.capture.transformed_depth

    def get_mask(self, id):
        return self.objframe.get_mask(id)

    def get_masked_point_cloud(self, _id):
        """returns the masked point cloud data , as a list with shape (N, 3)"""
        return self.get_point_cloud()[self.objframe.get_mask(_id) == 127]

    def get_point_cloud(self):
        """returns the entire point cloud for a frame, as a list with shape(W, H, 3) -> (1080, 1920, 3).
         Use .reshape(-1, 3) for format (N, 3)"""
        return self.capture.transformed_depth_point_cloud

    def get_image(self):
        """returns the frame image"""
        return self.color

    def get_obj_image(self):
        return self.objframe.result.plot()

    def get_point_cloud_colors(self, _id, d = False):
        """returns the point cloud colors for open3d in the correct format,

         to use pointcloud.colors = o3d.utility.Vector3dVector(get_point_could_colors(id))"""
        if d and self.denoised is not None:
            _image = cv2.cvtColor(self.denoised, cv2.COLOR_BGR2RGB)
        else:
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

    def show_depth(self):
        cv2.imshow("Depth", Tools.colorize(self.get_depth(), (None, 5000)))
        cv2.waitKey(int(1000/60))

    def show_transformed_depth(self):
        """shows the transformed depth image"""
        cv2.imshow("Transformed Depth", Tools.colorize(self.get_transformed_depth(), (None, 5000)))
        cv2.waitKey(int(1000/60))

    def show_ir_image(self):
        """shows the ir image"""
        cv2.imshow("IR Image", self.get_ir_image())
        cv2.waitKey(int(1000/60))

    def show_point_cloud_colored(self, _id):
        """shows the frame point cloud with colores attached, uses open3d"""
        pcd = o3d.geometry.PointCloud()
        masked_points = self.get_masked_point_cloud(_id)
        pcd.points = o3d.utility.Vector3dVector(masked_points)
        pcd.colors = o3d.utility.Vector3dVector(self.get_point_cloud_colors(_id))
        o3d.visualization.draw_geometries([pcd])

    def save_point_cloud_colored(self, _id, path, d = False):
        pcd = o3d.geometry.PointCloud()
        masked_points = self.get_masked_point_cloud(_id)
        pcd.points = o3d.utility.Vector3dVector(masked_points)
        pcd.colors = o3d.utility.Vector3dVector(self.get_point_cloud_colors(_id, d))
        o3d.io.write_point_cloud(f"{path}.ply", pcd, write_ascii = False)

    def get_ids(self):
        """returns the tracking ids for this frame"""
        return self.objframe.object_ids
