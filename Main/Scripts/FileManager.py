from pathlib import Path

SAVED_FRAMES_PATH = Path("../SavedFrames")
POINT_CLOUDS_PATH = Path("../PointClouds")
JSON_SETTINGS = Path("../settings.json")
YOLO_PATH = Path("../Models/yolo11m-seg.pt")
SCUNET_PATH = Path("../Models/scunet_color_real_psnr.pth")

def get_settings_dir():
    return JSON_SETTINGS.resolve()

def get_saved_frame_dir():
    return SAVED_FRAMES_PATH.resolve()

def get_bundle_dir(bundle_name):
    return (SAVED_FRAMES_PATH.joinpath(Path(bundle_name))).resolve()

def get_frame_dir(bundle_name, frame_name):
    return (SAVED_FRAMES_PATH.joinpath(Path(bundle_name), Path(frame_name))).resolve()

def get_point_cloud_dir():
    return POINT_CLOUDS_PATH.resolve()

print(get_saved_frame_dir())
print(get_bundle_dir("test bundle"))
print(get_frame_dir("test Bundle", "test frame"))
print(get_point_cloud_dir())