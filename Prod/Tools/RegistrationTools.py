from pathlib import Path

from dash import callback
from dearpygui import dearpygui as dpg
from sympy import roots
import open3d as o3d
from Prod.Tools.registration_funcs import get_clusters, number_of_points
from Prod.Tools.registration_funcs import ransac_icp_registration


class Registrator():
    def __init__(self, root_paths: list, output_path, merge_structure: dict[int, int], bundle_path, voxel_size = 11, callback=None):
        self.roots = root_paths
        self.graph = merge_structure
        self.output_path = output_path
        self.bundle_path = bundle_path
        self.voxel_size = voxel_size

        self.cluster_num = [1, 1, 1, 1, 0]
        self.id_pcd = {}
        self.callback = callback

        self.run()

    def run(self):
        pairs = self.pair_nodes_and_map_to_value(self.graph)

        print(pairs)

        progress = "Loading "
        self.callback(progress)
        for pair, id in pairs.items():
            print(f"starting reg between {pair}")
            pcda = self.get_pcd(pair[0])
            pcdb = self.get_pcd(pair[1])

            print(pcda)
            print(pcdb)
            print(id[0])
            self.id_pcd[id[0]] = ransac_icp_registration(pcda, pcdb, self.voxel_size)
            o3d.visualization.draw_geometries([self.id_pcd[id[0]]])

            progress = f"{progress}."
            if self.callback is not None:
                self.callback(progress)

        print(dpg.get_alias_id("FinalPointCloud"))
        final_pcd = None
        for k, v in self.graph.items():
            print(f"k {k}, v {v}, {dpg.get_alias_id('FinalPointCloud')}")
            if v[-1] == dpg.get_alias_id("FinalPointCloud"):
                final_pcd = self.id_pcd[k]

        if final_pcd is not None:
            dir_path = Path(f"PointClouds/{self.bundle_path}")
            dir_path.mkdir(parents = True, exist_ok = True)
            o3d.io.write_point_cloud(f"{dir_path}/{self.output_path}.ply", final_pcd)

            saved_pcd = o3d.io.read_point_cloud(f"{dir_path}/{self.output_path}.ply")
            o3d.visualization.draw_geometries([saved_pcd])




    def get_pcd(self, id):
        if id in self.id_pcd:
            return self.id_pcd[id]
        else:
            return self.load_frame_bundle(dpg.get_item_label(id))

    def find_all_keys_by_value(self, d, x):
        return [key for key, value in d.items() if value == x]

    def pair_nodes_and_map_to_value(self, d):
        # A dictionary to store lists of keys by their values
        value_to_keys = {}

        # Iterate through the dictionary and group keys by their values
        for key, value in d.items():
            value_tuple = tuple(value)  # Convert list to tuple (hashable) for dictionary keys
            if value_tuple not in value_to_keys:
                value_to_keys[value_tuple] = []
            value_to_keys[value_tuple].append(key)

        # Now create the mapping of node pairs to the common value
        paired_mapping = {}
        for keys in value_to_keys.values():
            if len(keys) > 1:  # Only consider pairs or more keys
                # Iterate over the combinations of keys and assign them the value
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        pair = sorted([keys[i], keys[j]])  # Sort the pair to ensure consistency
                        paired_mapping[tuple(pair)] = d[keys[i]]  # Map pair to the shared value

        return paired_mapping

    def load_frame_bundle(self, name):
        dir = f"SavedFrames/{self.bundle_path}/{name}/point_cloud.ply"
        print(dir)
        pcd = o3d.io.read_point_cloud(dir)
        print(f"Point Count {number_of_points(pcd)}")
        voxel = pcd.voxel_down_sample(voxel_size = self.voxel_size)
        print(f"Point Count {number_of_points(voxel)}")
        clusters, aabbs = get_clusters(voxel)
        best_cluster = clusters[0]
        best_cluster_id = 0
        for i, c in enumerate(clusters):
            if number_of_points(c) > number_of_points(best_cluster):
                best_cluster = c
                best_cluster_id = i

        o3d.visualization.draw_geometries([best_cluster])
        crp = pcd.crop(aabbs[best_cluster_id]) # crop raw point cloud based on calculated clusters
        crp_ds = crp.voxel_down_sample(voxel_size = self.voxel_size)
        print(f"Point Count {number_of_points(crp_ds)}")# get clusters and bounding boxes
        return crp_ds
