import json
import os.path
import uuid
from pathlib import Path

import cv2
import numpy as np
from dearpygui import dearpygui as dpg

from Main.Scripts.AzureKinectTools import AzureKinectTools
from Main.Scripts.Frame import Frame
from Main.Scripts.RegistrationTools import Registrator
from Main.Scripts.Tools import Tools
from Main.Scripts.KNN_upsample import run_main

import open3d as o3d

ak = None
current_frame = Frame
total_frames = 0
slider = None

SETTINGS_DEFAULT = {
    "input_file_path": "",
    "frame_bundle" : "",
    "upsample_factor": 19,
    "knn_factor" : 14,
    "cov_factor" : 0.4,
    "upsample_filename": ""
}

default_json_path = "settings.json"

def load_settings():
    global settings
    if not os.path.exists(default_json_path):
        with open(default_json_path, "w") as f:
            json.dump(SETTINGS_DEFAULT, f, indent = 4)

    with open(default_json_path, "r") as f:
        settings = json.load(f)

def save_settings():
    global settings
    with open(default_json_path, "w") as f:
        json.dump(settings, f, indent = 4)


def load_mkv_callback(sender, app_data):
    global settings
    settings['input_file_path'] = app_data['file_path_name']
    import_mkv()

def save_default_json_callback():
    global settings
    if not os.path.exists(default_json_path):
        with open(default_json_path, "w") as f:
            json.dump(settings, f)
    with open(default_json_path, "w") as f:
        json.dump(settings, f, indent = 4)
    refresh_bundles()

def show_warning():
    dpg.configure_item("warning_popup", show=True)

def close_warning():
    dpg.configure_item("warning_popup", show=False)

def update_frame():
    dpg.set_value("rgb_video_frame", convert_img(current_frame.get_image()))
    dpg.set_value("obj_video_frame", convert_img(current_frame.get_obj_image()))
    dpg.set_value("depth_video_frame", convert_img(Tools.colorize(current_frame.get_transformed_depth())))
    if current_frame.denoised is not None:
        dpg.show_item("Denoised_view")
        dpg.set_value("denoised_image", convert_img(current_frame.denoised))
    else:
        dpg.hide_item("Denoised_view")

    update_list()

global_check_obj = {}
global_selected_ids = {}

def import_mkv():
    global settings, ak, total_frames, current_frame, global_check_obj
    dpg.show_item("file_import")
    ak = AzureKinectTools(settings['input_file_path'], progress_callback=update_progress_bar)
    current_frame = ak.get_frame(0)
    dpg.set_value("progress_bar", f"All Frames Processed")
    total_frames = len(ak.get_frames())

    refresh_bundles()

    if dpg.does_item_exist("global_object_ids"):
        dpg.delete_item("global_object_ids")
        global_check_obj.clear()
    with dpg.child_window(label = "Global Objects in Frame", tag = "global_object_ids", parent = "object_list", show = True, height = 100):
        for item in ak.object_ids:
            print(item)
            global_check_obj[item] = dpg.add_checkbox(label = item, callback = on_item_select)
    on_item_select()

    dpg.configure_item("video_slider", max_value = total_frames)
    dpg.show_item("frame_tools")
    dpg.show_item("video_window")
    dpg.hide_item("file_import")

    update_frame()

def update_progress_bar(progress):
    dpg.set_value("progress_bar", f"Frames Processed - {progress}")

def update_frame_viewer(sender):
    global ak, current_frame
    current_frame = ak.get_frame(dpg.get_value(sender))
    update_frame()

def convert_img(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return image.astype(np.float32) / 255.0

checkboxes = {}
selected_ids = {}
def update_list():
    global checkboxes
    if dpg.does_item_exist("object_ids"):
        dpg.delete_item("object_ids")
        checkboxes.clear()
    with dpg.child_window(label = "Objects in Frame", tag = "object_ids", parent = "object_list", show = True):
        dpg.add_text("Frame Objects")
        for item in current_frame.get_ids():
            checkboxes[item] = dpg.add_checkbox(label = item, callback = on_item_select)
    on_item_select()

def show_frame_point_cloud():
    global checkboxes, selected_ids
    current_frame.show_point_cloud_colored(selected_ids)

def on_item_select():
    global checkboxes, selected_ids, global_selected_ids, global_check_obj
    selected_ids = []
    for item, checkbox_id in checkboxes.items():
        if dpg.get_value(checkbox_id):
            selected_ids.append(item[0])
    global_selected_ids = []
    for item, checkbox_id in global_check_obj.items():
        if dpg.get_value(checkbox_id):
            global_selected_ids.append(item[0])

    selected_ids = list(set(selected_ids + global_selected_ids))
    dpg.set_value("mask_video_frame", convert_img(current_frame.get_masked_image(selected_ids)))

def save_frame():
    global current_frame
    dir_path = Path(f"../Main/SavedFrames/{dpg.get_value('bundle_name')}/Frame_{dpg.get_value('video_slider')}_{uuid.uuid4().hex[:4]}")
    dir_path.mkdir(parents = True, exist_ok = True)

    if dpg.get_value("save_denoised") and current_frame.denoised is not None:
        save_img(current_frame.denoised, dir_path,"rgb_image")
        print("saved denoised")
    else:
        save_img(current_frame.get_image(), dir_path, "rgb_image")

    save_img(current_frame.get_obj_image(), dir_path, "obj_image")
    save_img(current_frame.get_masked_image(selected_ids), dir_path, "mask_image")
    save_img(Tools.colorize(current_frame.get_transformed_depth()), dir_path, "depth_image")
    save_img(Tools.colorize(current_frame.get_depth()), dir_path, "depth_raw_image")
    save_img(current_frame.get_mask(selected_ids), dir_path, "mask_raw_image")

    current_frame.save_point_cloud_colored(selected_ids, f"{dir_path}/point_cloud", dpg.get_value("save_denoised"))

    info = {
        "selected_ids": selected_ids
    }

    with open(f"{dir_path}/frame_info.json", "w") as file:
        json.dump(info, file, indent = 4)

    refresh_bundles()

bundles = [""]
frames = []
selected_bundle = bundles[0]
def refresh_bundles():
    global bundles
    root = Path("../Main/SavedFrames")
    root.mkdir(parents = True, exist_ok = True)
    bundles = [d.name for d in root.iterdir() if d.is_dir()]
    refresh_frames()

def refresh_frames():
    global frames
    if len(bundles) > 0:
        print(selected_bundle)
        root = Path(f"../Main/SavedFrames/{selected_bundle}")
        frames = [d.name for d in root.iterdir() if d.is_dir()]
        dpg.configure_item("bundle_name", default_value = selected_bundle)
        update_save_list()

def save_img(data, path, name):
    cv2.imwrite(f"{path}/{name}.png", data)

def update_save_list():
    global bundles, frames
    dpg.configure_item("bundles_list", items = bundles)
    dpg.configure_item("bundle_selector", items = bundles)
    dpg.configure_item("frame_list", items = frames)

def bundle_select():
    global selected_bundle
    selected_bundle = dpg.get_value("bundles_list")
    print(selected_bundle)

def bundle_select_reg():
    global selected_bundle
    selected_bundle = dpg.get_value("bundle_selector")
    print(selected_bundle)
    update_editor_bundle()

def frame_select():
    global current_frame
    selected_frame = dpg.get_value("frame_list")
    dir = f"../Main/SavedFrames/{selected_bundle}/{selected_frame}"
    open_folder(dir)

def open_folder(path):
    path = Path(path).resolve()
    os.startfile(path)

def __link_callback(sender, app_data):
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)


def __delink_callback(_, app_data):
    dpg.delete_item(app_data)

def create_node(name, pos = (100, 100), io = [[], []], tag="", image_path = ""):
    global selected_bundle, bundles
    with dpg.node(label = name, parent = "node_editor", pos=pos, tag=tag) as node_id:
        with dpg.popup(dpg.last_item(), mousebutton = dpg.mvMouseButton_Right):
            dpg.add_button(label = "Create New Node", callback = create_new_node)
        for c in io[0]:
            with dpg.node_attribute(attribute_type = dpg.mvNode_Attr_Input) as node:
                dpg.add_text("In")
            if c is not None:
                dpg.add_node_link(c, node, parent = "node_editor")

        if image_path != "":
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            width = int(img.shape[1] * 0.1)  # Scale to 50% of the original width
            height = int(img.shape[0] * 0.1)  # Scale to 50% of the original height

            img = cv2.cvtColor(cv2.resize(img, (width, height)), cv2.COLOR_BGR2RGBA)
            height, width, _ = img.shape
            # Normalize pixel values to [0, 1]
            img_data = img.astype(np.float32) / 255.0
            img_data = img_data.flatten()

            # Create texture
            with dpg.texture_registry():
                tex_id = dpg.add_static_texture(width, height, img_data)

            with dpg.node_attribute(attribute_type = dpg.mvNode_Attr_Static):
                dpg.add_image(width = width, height = height, texture_tag = tex_id)

        for c in io[1]:
            with dpg.node_attribute(attribute_type = dpg.mvNode_Attr_Output) as node:
                dpg.add_text("OUT")
            if c is not None:
                dpg.add_node_link(node, c, parent = "node_editor")

    return node_id

def create_new_node():
    selected_nodes = dpg.get_selected_nodes("node_editor")
    if len(selected_nodes) > 0:
        labname = ""
        for node in selected_nodes:
            labname += f"{dpg.get_item_label(node).split('_')[1]}-"
        labname = f"[{labname[:-1]}]"

        new_label = f"Frame_[{labname}]_pcd"
        y = 0
        for node in selected_nodes:
            y += dpg.get_item_pos(node)[1]
        y /= len(selected_nodes)

        pos = [dpg.get_item_pos(selected_nodes[0])[0] + 200, y]
        connections = []
        for node in selected_nodes:
            connections.append(dpg.get_item_children(node, 1)[-1])

        create_node(new_label, io = [connections,[None]], pos = pos)

def update_editor_bundle():
    global selected_bundle, bundles, frames
    if len(bundles) < 1:
        return
    if selected_bundle == "":
        selected_bundle = bundles[0]

    if dpg.does_item_exist("node_editor"):
        dpg.delete_item("node_editor")

    with dpg.node_editor(tag = "node_editor",parent = "node_editor_tab" ,width = 1500, height = 1000, callback=__link_callback, delink_callback=__delink_callback):

        refresh_bundles()
        i = 0
        for f in frames:
            create_node(f, io = [[],[None]], pos = [0, i], image_path = f"../Main/SavedFrames/{selected_bundle}/{f}/obj_image.png")
            i+= 200
        create_node("Final Point Cloud", io = ([None], []), pos = [500, i/2], tag="FinalPointCloud")
        print("bundles")

def run_registration():
    global bundles, selected_bundle
    # Get all node links
    graph = {}
    links = dpg.get_item_children("node_editor", 0)  # 0 = mvNode_Link
    for link_id in links:
        link_data = dpg.get_item_configuration(link_id)
        output_attr = link_data['attr_1']
        input_attr = link_data['attr_2']

        # Get parent nodes
        output_node = dpg.get_item_parent(output_attr)
        input_node = dpg.get_item_parent(input_attr)

        # Initialize graph if not present
        if output_node not in graph:
            graph[output_node] = []

        # Append connection
        graph[output_node].append(input_node)

    print(graph)

    outgoing_nodes = set(graph.keys())
    incoming_nodes = set()
    for targets in graph.values():
        incoming_nodes.update(targets)
    root_nodes = outgoing_nodes - incoming_nodes

    reg = Registrator(root_nodes, dpg.get_value("output_filename"), graph, selected_bundle, dpg.get_value("voxel_size"), callback = reg_progress)

def reg_progress(progress):
    dpg.set_value("registration_progress", progress)

upsample_file = ""
def load_upsample_file(sender, app_data):
    global upsample_file
    upsample_file = app_data['file_path_name']
    dpg.set_value("upsample_file", upsample_file)
    dpg.show_item("parameter_group")

def run_upsample():
    global settings, upsample_file
    print("starting upsample")
    settings['upsample_factor'] = dpg.get_value("upsample_factor")
    settings['knn_factor'] = dpg.get_value("knn_factor")
    settings['cov_factor'] = dpg.get_value("cov_factor")
    settings['upsample_filename'] = dpg.get_value('upsample_filename')

    run_main(upsample_file, settings)

def show_point_cloud(sender, app_data):
    pcd = o3d.io.read_point_cloud(app_data["file_path_name"])
    o3d.visualization.draw_geometries_with_editing([pcd])

def run_denoising():
    global current_frame
    dpg.show_item("Denoising_Loading")
    current_frame.rerun_with_denoise()
    dpg.show_item("save_denoised")
    dpg.set_value("save_denoised", True)
    dpg.hide_item("Denoising_Loading")

    update_frame()
    print("Finished Denoising")


dpg.create_context()
load_settings()

with dpg.texture_registry():
    initial_texture = np.zeros((1920, 1080, 4), dtype=np.uint8).flatten()
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "rgb_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "obj_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "mask_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "depth_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "denoised_image")

with dpg.window(label="Warning !!!", modal=True, show=False, tag="warning_popup", no_title_bar=False):
    dpg.add_text("Are you sure you want to continue...")
    with dpg.group(horizontal=True):
        dpg.add_button(label="No", width=75, callback=close_warning)
        dpg.add_button(label="Yes", width=75, callback=lambda: (close_warning(), save_default_json_callback))

with dpg.window(label="File Import", modal=True, show=False, tag="file_import", no_title_bar=False, pos = [800, 500]):
    dpg.add_text(f"Importing {settings['input_file_path']}")
    dpg.add_text(default_value = "Loading ... ", tag = "progress_bar")
    dpg.add_text("", tag = "patience_string")

with dpg.window(label = "Denoising", modal = True, show = False, tag = "Denoising_Loading", no_title_bar = True, pos = [800, 600]):
    dpg.add_text("Loading...")

with dpg.file_dialog(directory_selector = False, show=False, callback = load_mkv_callback, tag = "file_dialog_id", width = 500, height = 700):
    dpg.add_file_extension("Video Files (*.mkv){.mkv}")

with dpg.file_dialog(directory_selector = False, show=False, callback = load_upsample_file, tag = "file_dialog_id_upsample", width = 500, height = 700):
    dpg.add_file_extension("Point Cloud Files (*.ply){.ply}")

with dpg.file_dialog(directory_selector = False, show=False, callback = show_point_cloud, tag = "show_point_cloud", width = 500, height = 700):
    dpg.add_file_extension("Point Cloud Files (*.ply){.ply}")

with dpg.handler_registry():

    #tools
    with dpg.window(label = "Frame Tools", tag="frame_tools", show = False, width = 500, height = 500):
        #slider = dpg.add_slider_int(label = "Select Frame", tag = "video_slider", default_value = 0, min_value = 0, max_value = total_frames, callback = update_frame_viewer)
        dpg.add_input_int(label = "Select Frame", tag = "video_slider", default_value = 0, max_value = total_frames, callback = update_frame_viewer)
        dpg.add_text("Object in Frame")
        dpg.add_button(label = "View Point Cloud (open3d)", callback = show_frame_point_cloud)
        dpg.add_button(label = "Run Denoising on frame", callback = run_denoising)
        dpg.add_tooltip(parent = dpg.last_item(), label = "WARNING many take time to run be patient, and will override original results")
        dpg.add_checkbox(label = "Save With Denoised", default_value = False, tag = "save_denoised", show = False)
        with dpg.tab_bar():
            with dpg.tab(label = "Saved Data", tag = "frame_save"):
                dpg.add_input_text(default_value = selected_bundle, label = "Frame Bundle Name",
                                   tag = "bundle_name")
                dpg.add_button(label = "Save", callback = save_frame)
                with dpg.group(label = "Saved Data", tag = "saved_data", horizontal = True):
                    dpg.add_listbox(items = bundles, label = "", tag = "bundles_list", num_items = 5, width = 200, callback = refresh_bundles)
                    dpg.add_listbox(items = frames, label = "", tag = "frame_list", num_items = 5, width = 200, callback = frame_select)
            with dpg.tab(label = "Object Select", tag = "object_list"):
                dpg.add_text("Global Objects")


    #video viewer
    with dpg.window(label = "Frame Viewer",tag = "video_window", show = False, pos=[450, 75]):
        with dpg.tab_bar(tag = "frame_viewer_tabs"):
            with dpg.tab(label = "Object Detection"):
                dpg.add_image("obj_video_frame")
            with dpg.tab(label = "Masked Image"):
                dpg.add_image("mask_video_frame")
            with dpg.tab(label = "Depth Image"):
                dpg.add_image("depth_video_frame")
            with dpg.tab(label = "Raw RGB Image"):
                dpg.add_image("rgb_video_frame")
            with dpg.tab(label = "Denoised Image", tag = "Denoised_view", show = False):
                dpg.add_image("denoised_image")

    with dpg.window(label = "Point Cloud Registation", tag = "registration_window", show=False, pos = [450, 75], width = 1500, height = 1000):
        with dpg.tab_bar():
            with dpg.tab(label = "Registration Settings"):
                dpg.add_text("VoxelSize")
                dpg.add_input_int(default_value = 11, tag="voxel_size")
                dpg.add_text("Select Frame Bundle")
                dpg.add_listbox(items = bundles, tag = "bundle_selector", num_items = 5, width = 200, callback = bundle_select_reg)
                dpg.add_button(label = "Refresh", callback = refresh_bundles)
                dpg.add_spacer(height = 50)
                dpg.add_input_text(label = "Filename", tag="output_filename")
                dpg.add_button(label = "Run Registration", callback = run_registration)
                dpg.add_text(label = "", tag = "registration_progress")
            dpg.add_tab(label = "Point Cloud Merging", tag = "node_editor_tab")


    with dpg.window(label = "Point Cloud Upsampling", tag = "upsample_window", show = False, pos = [450, 75], width = 400, height = 600):
        with dpg.group(horizontal = True):
            dpg.add_button(label = "Import File", callback = lambda : dpg.show_item("file_dialog_id_upsample"))
            dpg.add_text("", tag="upsample_file")
        with dpg.group(label = "Parameters", tag = "parameter_group", show = False):
            dpg.add_text("Upsample Factor")
            dpg.add_input_int(tag = "upsample_factor", default_value = settings["upsample_factor"])
            dpg.add_text("Knn Factor")
            dpg.add_input_int(tag = "knn_factor", default_value = settings["knn_factor"])
            dpg.add_text("Cov Spread Factor")
            dpg.add_input_float(tag = "cov_factor", default_value = settings["cov_factor"])
            dpg.add_spacer(height = 100)
            dpg.add_text("Output Filename")
            dpg.add_input_text(tag = "upsample_filename", default_value = "New Upsampled Point Cloud")
            dpg.add_button(label = "Generate", callback = run_upsample)

dpg.create_viewport(title = "LiDAR project")

with dpg.viewport_menu_bar():
    with dpg.menu(label = "File"):
        dpg.add_menu_item(label = "Import .mkv", callback = lambda: dpg.show_item("file_dialog_id"))
        dpg.add_menu_item(label = f"Import {settings['input_file_path']}", callback = import_mkv)
        dpg.add_menu_item(label = "Open .ply", callback = lambda : dpg.show_item("show_point_cloud"))
    with dpg.menu(label = "Settings"):
        dpg.add_menu_item(label = "Load (settings.json)", callback = load_settings)
        dpg.add_menu_item(label = "Save (settings.json)", callback = save_settings)
    dpg.add_menu_item(label = "Registration", callback = lambda : dpg.show_item("registration_window"))
    dpg.add_menu_item(label = "Upsampling", callback = lambda: dpg.show_item("upsample_window"))


refresh_bundles()
update_editor_bundle()
dpg.setup_dearpygui()

dpg.show_viewport()
dpg.maximize_viewport()
dpg.start_dearpygui()

dpg.destroy_context()