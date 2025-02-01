import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    print(frame)
    bbox, label, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, label, conf)

    cv2.imshow("object detection", output_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break