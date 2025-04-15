import open3d as o3d
import numpy as np 
import copy
from matplotlib import pyplot as plt
import cv2
import time

# References
# [1] https://www.open3d.org/html/tutorial/Advanced/global_registration.html
# [2] Reference https://github.com/isl-org/Open3D/issues/5501

# Get registered point cloud
def get_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.transform(transformation)
    return source_temp + target_temp

# Compute feature points
def preprocess_point_cloud(pcd, voxel_size):
    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh

# Load point clouds and tranform to arbitary positions
def prepare_dataset(source, target, voxel_size):
    #print(":: Load two point clouds and disturb initial pose.")

    source.estimate_normals()
    target.estimate_normals()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

# Registration using RANSAC
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   a distance threshold of %.3f is used." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# Refine registration results using ICP
def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time a strict")
    #print("   distance threshold %.3f is used." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# Compute clusters using DBSCAN
def get_clusters(pcd, apply_label_colors = False, compute_aabbs = True, eps=500, min_points=10):
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
        labels = np.array(
            pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

        if labels.size == 0:
            return None, None

        max_label = labels.max()
        
        if apply_label_colors:
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        clusters = [pcd.select_by_index(list(np.where(labels == i)[0])) for i in range(max_label + 1)]
        aabbs    = [cluster.get_axis_aligned_bounding_box() for cluster in clusters] if compute_aabbs else []

        for aabb in aabbs:
            aabb.color = [0, 0, 0]

        return clusters, aabbs

# Return number of points in point cloud
def number_of_points(pcd): return len(np.asarray(pcd.points))

# Full RANSAC + ICP pipeline with point cloud output
def ransac_icp_registration(pcd_A, pcd_B, voxel_size):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_A, pcd_B, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size)
    print("For a voxel size of", voxel_size, ":", result_icp)
    output_pcd = get_registration_result(source, target, result_icp.transformation)
    return output_pcd

# Full RANSAC + ICP pipeline with transform output
def ransac_icp_registration_transform(pcd_A, pcd_B, voxel_size):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_A, pcd_B, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size)
    print("For a voxel size of", voxel_size, ":", result_icp)
    return result_icp.transformation

# Get point cloud of segmented object from given mkv frame
def get_pcd_from_mkv(video,frame_num,object):
    frame = video.get_frame(frame_num) # get frame number n
    pcd_points = frame.get_masked_point_cloud(object) # get masked point cloud for correct mask
    pcd_colours = frame.get_point_cloud_colors(object) # get rgb data for correct mask
    pcd = o3d.geometry.PointCloud() # create point cloud variable
    pcd.points = o3d.utility.Vector3dVector(pcd_points) # add points
    pcd.colors = o3d.utility.Vector3dVector(pcd_colours) # add colours
    return pcd

# Crop point cloud
def crop_pcd(pcd,cluster_num):
    clusters, aabbs = get_clusters(pcd) # get clusters and bounding boxes
    pcd_crp = pcd.crop(aabbs[cluster_num]) # crop raw point cloud based on calculated clusters
    return pcd_crp

def registrate_raw_pcd(pcd_A,pcd_B,pcd_A_ds,pcd_B_ds,voxel_size): # assumes normals have been calclulated for raw point clouds
    transform = ransac_icp_registration_transform(pcd_A_ds,pcd_B_ds,voxel_size) # compute RANSAC registration transformation on downsampled data for speed
    output_pcd = get_registration_result(pcd_A,pcd_B,transform) # apply transformation to raw data
    return output_pcd

# Full RANSAC + ICP pipeline return RMSE
def ransac_icp_registration_rmse(pcd_A, pcd_B, voxel_size):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_A, pcd_B, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size)
    return result_icp.inlier_rmse

# Full RANSAC + Coloured ICP pipeline
def ransac_colour_icp_registration(pcd_A, pcd_B, voxel_size):
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcd_A, pcd_B, voxel_size)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    result_icp = refine_registration_colour(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size)
    print("For a voxel size of", voxel_size, ":", result_icp)
    output_pcd = get_registration_result(source, target, result_icp.transformation)
    return output_pcd 

def refine_registration_colour(source, target, source_fpfh, target_fpfh, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time a strict")
    #print("   distance threshold %.3f is used." % distance_threshold)
    result = o3d.pipelines.registration.registration_colored_icp(
        source, target, distance_threshold, result_ransac.transformation)
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def registrate_raw_pcd(pcd_A,pcd_B,voxel_size):
    # in order to apply transformation to raw data, the normals have to be calculated first
    pcd_A.estimate_normals()
    pcd_B.estimate_normals() 
    # down sample raw point cloud
    pcd_A_ds = pcd_A.voxel_down_sample(voxel_size=voxel_size)
    pcd_B_ds = pcd_B.voxel_down_sample(voxel_size=voxel_size)
    transformation = ransac_icp_registration_transform(pcd_A_ds,pcd_B_ds,voxel_size) # compute RANSAC registration transformation on downsampled data for speed
    output_pcd = get_registration_result(pcd_A,pcd_B,transform) # apply transformation to raw data
    #output_pcd = pcd_A.transform(transformation) + pcd_B
    return output_pcd
