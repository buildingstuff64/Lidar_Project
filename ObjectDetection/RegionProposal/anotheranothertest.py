from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-seg.pt")

results = model.track(source = 0, show = True, device=0, conf=0.4, save=True)
