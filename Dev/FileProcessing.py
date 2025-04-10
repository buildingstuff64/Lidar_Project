import json
import os.path
import pickle
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from dearpygui import dearpygui as dpg

from Prod.Tools.AzureKinectTools import AzureKinectTools
from Prod.Tools.RegistrationTools import Registrator
from Prod.Tools.Tools import Tools

ak = None
current_frame = None
total_frames = 0
slider = None

default_json_path = "settings.json"
with open(default_json_path, "r") as f:
    settings = json.load(f)

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
    update_list()

def import_mkv():
    global settings, ak, total_frames, current_frame
    dpg.show_item("file_import")
    ak = AzureKinectTools(settings['input_file_path'], progress_callback=update_progress_bar)
    dpg.set_value("progress_bar", f"All Frames Processed")
    total_frames = len(ak.get_frames())

    refresh_bundles()
    dpg.configure_item("video_slider", max_value = total_frames)
    dpg.show_item("frame_tools")
    dpg.show_item("video_window")
    dpg.hide_item("file_import")

    current_frame = ak.get_frame(0)
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
        for item in current_frame.get_ids():
            checkboxes[item] = dpg.add_checkbox(label = item, callback = on_item_select)
    on_item_select()

def show_frame_point_cloud():
    global checkboxes, selected_ids
    current_frame.show_point_cloud_colored(selected_ids)

def on_item_select():
    global checkboxes, selected_ids
    selected_ids = []
    for item, checkbox_id in checkboxes.items():
        if dpg.get_value(checkbox_id):
            selected_ids.append(item[0])
    dpg.set_value("mask_video_frame", convert_img(current_frame.get_masked_image(selected_ids)))

def save_frame():
    dir_path = Path(f"SavedFrames/{dpg.get_value('bundle_name')}/Frame_{dpg.get_value('video_slider')}_{uuid.uuid4().hex[:4]}")
    dir_path.mkdir(parents = True, exist_ok = True)

    save_img(current_frame.get_image(), dir_path, "rgb_image")
    save_img(current_frame.get_obj_image(), dir_path, "obj_image")
    save_img(Tools.colorize(current_frame.get_transformed_depth()), dir_path, "depth_image")

    current_frame.save_point_cloud_colored(selected_ids, f"{dir_path}/point_cloud")

    info = {
        "selected_ids": selected_ids
    }

    with open(f"{dir_path}/frame_info.json", "w") as file:
        json.dump(info, file, indent = 4)

    refresh_bundles()

bundles = [""]
frames = []
selected_bundle = 0
def refresh_bundles():
    global bundles
    root = Path("SavedFrames")
    bundles = [d.name for d in root.iterdir() if d.is_dir()]
    refresh_frames()

def refresh_frames():
    global frames
    if len(bundles) > 0:
        root = Path(f"SavedFrames/{bundles[selected_bundle]}")
        frames = [d.name for d in root.iterdir() if d.is_dir()]
        dpg.configure_item("bundle_name", default_value = bundles[selected_bundle])
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

def bundle_select_reg():
    global selected_bundle
    selected_bundle = dpg.get_value("bundle_selector")

def frame_select():
    global current_frame
    selected_frame = dpg.get_value("frame_list")
    dir = f"SavedFrames/{bundles[selected_bundle]}/{selected_frame}"
    open_folder(dir)

def open_folder(path):
    path = Path(path).resolve()
    os.startfile(path)

def __link_callback(sender, app_data):
    dpg.add_node_link(app_data[0], app_data[1], parent=sender)


def __delink_callback(_, app_data):
    dpg.delete_item(app_data)

def create_node(name, pos = (100, 100), io = [[], []], tag=""):
    global selected_bundle, bundles
    with dpg.node(label = name, parent = "node_editor", pos=pos, tag=tag) as node_id:
        with dpg.popup(dpg.last_item(), mousebutton = dpg.mvMouseButton_Right):
            dpg.add_button(label = "Create New Node", callback = create_new_node)
        for c in io[0]:
            with dpg.node_attribute(attribute_type = dpg.mvNode_Attr_Input) as node:
                dpg.add_text("In")
            if c is not None:
                dpg.add_node_link(c, node, parent = "node_editor")

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
    if dpg.does_item_exist("node_editor"):
        dpg.delete_item("node_editor")

    with dpg.node_editor(tag = "node_editor", width = 1500, height = 1000, callback=__link_callback, delink_callback=__delink_callback):

        refresh_bundles()
        i = 0
        for f in frames:
            print(f)
            create_node(f, io = [[],[None]], pos = [0, i])
            i+=100
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

    reg = Registrator(root_nodes, dpg.get_value("output_filename"), graph, bundles[selected_bundle], dpg.get_value("voxel_size"), callback = reg_progress)

def reg_progress(progress):
    dpg.set_value("registration_progress", progress)

dpg.create_context()

with dpg.texture_registry():
    initial_texture = np.zeros((1920, 1080, 4), dtype=np.uint8).flatten()
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "rgb_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "obj_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "mask_video_frame")
    dpg.add_dynamic_texture(width = 1920, height = 1080, default_value = initial_texture, tag = "depth_video_frame")

with dpg.window(label="Warning !!!", modal=True, show=False, tag="warning_popup", no_title_bar=False):
    dpg.add_text("Are you sure you want to continue...")
    with dpg.group(horizontal=True):
        dpg.add_button(label="No", width=75, callback=close_warning)
        dpg.add_button(label="Yes", width=75, callback=lambda: (close_warning(), save_default_json_callback))

with dpg.window(label="File Import", modal=True, show=False, tag="file_import", no_title_bar=False, pos = [800, 500]):
    dpg.add_text(f"Importing {settings['input_file_path']}")
    dpg.add_text(default_value = "Loading ... ", tag = "progress_bar")
    dpg.add_text("", tag = "patience_string")

with dpg.file_dialog(directory_selector = False, show=False, callback = load_mkv_callback, tag = "file_dialog_id", width = 500, height = 700):
    dpg.add_file_extension("Video Files (*.mkv){.mkv}")

with dpg.handler_registry():

    #tools
    with dpg.window(label = "Frame Tools", tag="frame_tools", show = False, width = 500, height = 500):
        slider = dpg.add_slider_int(label = "Select Frame", tag = "video_slider", default_value = 0, min_value = 0, max_value = total_frames, callback = update_frame_viewer)
        dpg.add_text("Object in Frame")
        dpg.add_button(label = "View Point Cloud (open3d)", callback = show_frame_point_cloud)
        with dpg.tab_bar():
            with dpg.tab(label = "Saved Data", tag = "frame_save"):
                dpg.add_input_text(default_value = bundles[selected_bundle], label = "Frame Bundle Name",
                                   tag = "bundle_name")
                dpg.add_button(label = "Save", callback = save_frame)
                with dpg.group(label = "Saved Data", tag = "saved_data", horizontal = True):
                    dpg.add_listbox(items = bundles, label = "", tag = "bundles_list", num_items = 5, width = 200, callback = refresh_bundles)
                    dpg.add_listbox(items = frames, label = "", tag = "frame_list", num_items = 5, width = 200, callback = frame_select)
            dpg.add_tab(label = "Object Select", tag = "object_list")

    #video viewer
    with dpg.window(label = "Frame Viewer",tag = "video_window", show = False, pos=[450, 75]):
        with dpg.tab_bar():
            with dpg.tab(label = "Object Detection"):
                dpg.add_image("obj_video_frame")
            with dpg.tab(label = "Masked Image"):
                dpg.add_image("mask_video_frame")
            with dpg.tab(label = "Depth Image"):
                dpg.add_image("depth_video_frame")
            with dpg.tab(label = "Raw RGB Image"):
                dpg.add_image("rgb_video_frame")

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
            with dpg.tab(label = "Point Cloud Merging"):
                update_editor_bundle()






dpg.create_viewport(title = "LiDAR project")

with dpg.viewport_menu_bar():
    with dpg.menu(label = "File"):
        dpg.add_menu_item(label = "Import .mkv", callback = lambda: dpg.show_item("file_dialog_id"))
        dpg.add_menu_item(label = f"Import {settings['input_file_path']}", callback = import_mkv)
    with dpg.menu(label = "Settings"):
        dpg.add_menu_item(label = "Load (settings.json)")
        dpg.add_menu_item(label = "Save (settings.json)")
    dpg.add_menu_item(label = "Registration", callback = lambda : dpg.show_item("registration_window"))


refresh_bundles()
dpg.setup_dearpygui()

dpg.show_viewport()
dpg.maximize_viewport()
dpg.start_dearpygui()

dpg.destroy_context()