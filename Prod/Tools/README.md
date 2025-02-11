# Tools

## Setup
Reqirements are found in [Requirements.txt](requirements.txt)
```requirements
opencv-python~=4.10.0.84
numpy~=2.0.2
open3d~=0.19.0
matplotlib~=3.9.2
keyboard~=0.13.5
pykinect2024~=0.1.2024
ursina~=7.0.0
ultralytics~=8.3.70
torch~=2.6.0+cu126
cvlib~=0.2.7
pyk4a~=1.5.0
```

For usage of each module refer below

## [AzureKinectTools](AzureKinectTools.py)

Example of functions that can be used

```python
ak = AzureKinectTools(Tools.getPath())
ak.info()

ak.show_masked_point_cloud_video()

ak.show_images()
ak.show_ir_images()
ak.show_masked_images()
ak.show_transformed_depth()

ak.get_frame(index)
for frame in ak.get_frames():
    frame.show_yolo_image()
    frame.show_image()
    frame.show_masked_iamge()
    frame.show_transformed_depth()
    frame.show_ir_image()
```
