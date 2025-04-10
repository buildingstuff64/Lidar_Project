# Registration example from mkv

from registration_funcs import *

VOXEL_SIZE = 11

video = [AzureKinectTools.AzureKinectTools('output_1_light.mkv'),
         AzureKinectTools.AzureKinectTools('output_3_light.mkv'),
         AzureKinectTools.AzureKinectTools('output_7.mkv'),
         AzureKinectTools.AzureKinectTools('output_9.mkv'),
         AzureKinectTools.AzureKinectTools('output_11.mkv')]

object = [2,2,2,1,1] # object detection id
frame_num = [1,1,1,1,1] # frame number
cluster_num = [1,1,1,1,0] # cluster number

pcd = []
pcd_ds = []
pcd_crp = []
pcd_crp_ds = []

for k,v in enumerate(video):
    pcd.append(get_pcd_from_mkv(v,frame_num[k],object[k])) # get raw point cloud
    pcd_ds.append(pcd[k].voxel_down_sample(voxel_size=VOXEL_SIZE)) # down sample raw point cloud
    clusters, aabbs = get_clusters(pcd_ds[k]) # get clusters and bounding boxes
    pcd_crp.append(pcd[k].crop(aabbs[cluster_num[k]])) # crop raw point cloud based on calculated clusters
    pcd_crp_ds.append(pcd_crp[k].voxel_down_sample(voxel_size=VOXEL_SIZE))      

pcd_1_2 = ransac_icp_registration(pcd_crp_ds[0], pcd_crp_ds[1], VOXEL_SIZE)
pcd_4_6 = ransac_icp_registration(pcd_crp_ds[2], pcd_crp_ds[4], VOXEL_SIZE)
pcd_4_5_6 = ransac_icp_registration(pcd_4_6, pcd_crp_ds[3], VOXEL_SIZE)
pcd_1_2_4_5_6 = ransac_icp_registration(pcd_1_2, pcd_4_5_6, VOXEL_SIZE)
o3d.visualization.draw_geometries([pcd_1_2_4_5_6])

# Example from point cloud files

pcd = [o3d.io.read_point_cloud('./view_1_rgb_pcd_light.ply'),
o3d.io.read_point_cloud('./view_2_rgb_pcd_light.ply'),
o3d.io.read_point_cloud('./view_4_rgb_pcd_light.ply'),
o3d.io.read_point_cloud('./view_5_rgb_pcd_light.ply'),
o3d.io.read_point_cloud('./view_6_rgb_pcd_light.ply')]

for k,p in enumerate(pcd):
    pcd_ds.append(p.voxel_down_sample(voxel_size=VOXEL_SIZE)) # down sample raw point cloud
    clusters, aabbs = get_clusters(pcd_ds[k]) # get clusters and bounding boxes
    pcd_crp.append(pcd[k].crop(aabbs[cluster_num[k]])) # crop raw point cloud based on calculated clusters 
    pcd_crp_ds.append(pcd_crp[k].voxel_down_sample(voxel_size=VOXEL_SIZE))      

pcd_1_2 = ransac_icp_registration(pcd_crp_ds[0], pcd_crp_ds[1], VOXEL_SIZE)
o3d.visualization.draw_geometries([pcd_1_2])
pcd_4_6 = ransac_icp_registration(pcd_crp_ds[2], pcd_crp_ds[4], VOXEL_SIZE)
o3d.visualization.draw_geometries([pcd_4_6])
pcd_4_5_6 = ransac_icp_registration(pcd_4_6, pcd_crp_ds[3], VOXEL_SIZE)
o3d.visualization.draw_geometries([pcd_4_5_6])
pcd_1_2_4_5_6 = ransac_icp_registration(pcd_1_2, pcd_4_5_6, VOXEL_SIZE)
o3d.visualization.draw_geometries([pcd_1_2_4_5_6])
