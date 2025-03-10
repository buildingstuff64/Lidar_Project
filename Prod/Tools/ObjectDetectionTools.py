import numpy as np
from sympy.codegen.cnodes import static
from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Results


class ObjectDetectionTools:
    def __init__(self, path):
        self.path = path
        self.model = YOLO("../../Dev/Tools/yolo11n-seg.pt")
        self.results = self.model.track(source = self.path, show = False, device = 0, conf = 0.4, save = False,
                                        stream = True)

    @staticmethod
    def run_single_image(paths, save, conf):
        model = YOLO("../../Dev/Tools/yolo11n-seg.pt")
        for p in paths:
            model(source = p, show=True, device = 0, conf=conf, save=save)

    @staticmethod
    def live_camera():
        print("starting live camera detection")
        model = YOLO("../../Dev/Tools/yolo11n-seg.pt")
        model.track(source=0, show=True, device=0, conf=0.4, save=False)


class ObjectDetectionFrame:
    def __init__(self, result: Results):
        self.og_image = result.orig_img
        self.object_ids = list()
        print(result.names.values())
        for id, cls in zip(result.boxes.id.int().cpu().tolist(), result.boxes.cls.int().cpu().tolist()):
            self.object_ids.append((id, result.names[cls]))
        self.result = result

    def get_mask(self, id):
        """return the mask of the object #note 127 value is high, don't ask why"""
        mask = np.zeros(self.result.orig_shape[:2], dtype = np.int8)
        if len(self.object_ids) < 1:
            return mask

        index = self.find_id(id)
        pts = np.asarray(self.result.masks.xy[index], dtype = np.int32)
        cv2.fillPoly(mask, [pts], color = (255, 255, 255))
        return mask

    def get_masked_og_img(self, _id):
        """returns the original image with the mask applied"""
        return cv2.bitwise_and(self.og_image, self.og_image, mask = self.get_mask(_id))

    def find_id(self, _id):
        """helper function for finding the index of the object using its id"""
        for i, t in enumerate(self.object_ids):
            if int(t[0]) == int(_id):
                return i
        return None
