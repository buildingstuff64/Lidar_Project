import open3d as o3d
import numpy as np
import json
import pprint
import os

class ScanNetImporter:
    def __init__(self):
        self.folders = next(os.walk('../ObjectDetection/DatasetTest/scans'))[1]

    def get(self, i):
        return ScanNetPointCloud(self.folders[i])

class ScanNetPointCloud:
    def __init__(self, folder):
        self.sceneID = folder
        self.pcd = o3d.io.read_point_cloud(self.getfile('_vh_clean_2.ply'))
        self.info = json.load(open(self.getfile('_vh_clean.aggregation.json')))
        self.segments = json.load(open(self.getfile('_vh_clean_2.0.010000.segs.json')))
        self.segIndices = np.array(self.segments['segIndices'])

    def getfile(self, x):
        return f'../ObjectDetection/DatasetTest/scans/{self.sceneID}/{self.sceneID}{x}'

    def viewPointCloud(self):
        o3d.visualization.draw_geometries([self.pcd])

    def viewVoxel(self, size):
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size = size)
        o3d.visualization.draw_geometries([voxel_grid])

    def getSegementation(self):
        segments = []
        for s in self.info['segGroups']:
            segments.append(SubPointClouds(s, self))
        return segments

    def viewObjects(self, objects=[]):
        if len(objects) > 0:
            for i in self.getSegementation():
                if i.label in objects:
                    print(i)
                    i.viewVoxel(0.025)
        else:
            for i in self.getSegementation():
                print(i)
                i.viewVoxel(0.025)

class SubPointClouds:
    def __init__(self, objdata, ScanNetPT):
        self.points = []
        self.colors = []
        for s in objdata['segments']:
            for i in np.where(ScanNetPT.segIndices == s)[0]:
                self.points.append(np.asarray(ScanNetPT.pcd.points[i]))
                self.colors.append(np.asarray(ScanNetPT.pcd.colors[i]))

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        self.pcd.colors = o3d.utility.Vector3dVector(self.colors)

        self.label = objdata['label']
        self.id = objdata['id']

    def __str__(self):
        return f'Type: {self.label}, Object_ID: {self.id}'

    def viewPointCloud(self):
        o3d.visualization.draw_geometries([self.pcd])

    def viewVoxel(self, size):
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd, voxel_size = size)
        o3d.visualization.draw_geometries([voxel_grid])

if __name__ == '__main__':
    importer = ScanNetImporter()
    pt1 = importer.get(0)
    pt1.viewPointCloud()

    pt2 = importer.get(2)
    pt2.viewPointCloud()
    pt2.viewVoxel(0.025)
    pt2.viewObjects(['chair'])