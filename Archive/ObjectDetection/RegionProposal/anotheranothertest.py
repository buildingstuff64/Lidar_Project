import numpy as np
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt")

results = model.track(source = 'PersonWalking.mp4', show = False, device=0, conf=0.4, save=False)

for r in results:
    og_img = r.orig_img
    points = np.asarray(r.masks.xy[0], dtype = np.int32)
    mask = np.zeros(r.orig_shape[:2], dtype = np.int8)
    cv2.fillPoly(mask, [points], color = (255, 255, 255))
    masked_image = cv2.bitwise_and(og_img, og_img, mask = mask)

    cv2.imshow("masked_image", masked_image)
    cv2.waitKey(int(1000/60))

cv2.destroyAllWindows()