
import numpy as np
from ultralytics import YOLO
import cv2

class ObjectDetectionTools:
    def __init__(self, path):
        self.path = path
        self.model = YOLO("yolo11n-seg.pt")

    def getResults(self):
        return self.model.track(source = self.path, show = False, device=0, conf=0.4, save=False, stream = True)

class ObjectDetectionFrame:
    def __init__(self, result):
        self.og_image = result.orig_img
        pts = np.asarray(result.masks.xy[0], dtype = np.int32)
        self.mask = np.zeros(result.orig_shape[:2], dtype = np.int8)
        cv2.fillPoly(self.mask, [pts], color = (255, 255, 255))
        self.masked_image = cv2.bitwise_and(self.og_image, self.og_image, mask = self.mask)

    def show(self):
        cv2.imshow("masked_image", self.masked_image)
        cv2.waitKey(int(1000 / 60))


