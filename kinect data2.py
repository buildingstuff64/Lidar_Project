from time import sleep
import open3d as o3d
import cv2
from pykinect2024 import PyKinectRuntime
from pykinect2024.PyKinect2024 import FrameSourceTypes_Color, FrameSourceTypes_Infrared, FrameSourceTypes_Depth
import numpy as np

kinect = PyKinectRuntime.PyKinectRuntime(FrameSourceTypes_Color | FrameSourceTypes_Infrared | FrameSourceTypes_Depth)

while True:
    frame_color = kinect.get_last_color_frame()
    frame_color = np.reshape(frame_color, (1080, 1920, 4)).astype(np.uint8)
    cv2.imshow('color', frame_color)

    frame_ir = kinect.get_last_infrared_frame()
    frame_ir = np.reshape(frame_ir, (kinect.infrared_frame_desc.Height, kinect.infrared_frame_desc.Width)).astype(
        np.uint16)
    cv2.imshow('ir', frame_ir)


    if kinect.has_new_depth_frame():
        frame = kinect.get_last_depth_frame()
        kinect.get_last_depth_frame()
        frameD = kinect._depth_frame_data
        frame = frame.astype(np.uint8)
        frame = np.reshape(frame, (424, 512))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(x, y)
            if event == cv2.EVENT_RBUTTONDOWN:
                Pixel_Depth = frameD[((y * 512) + x)]
                print(Pixel_Depth)


        ##output = cv2.bilateralFilter(output, 1, 150, 75)
        cv2.imshow('KINECT Video Stream', frame)
        cv2.setMouseCallback('KINECT Video Stream', click_event)
        output = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kinect.close()
print("hello")