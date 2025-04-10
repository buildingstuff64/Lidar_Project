import open3d as o3d
import numpy as np


def load_pcd(pcd_path):

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return points, colors, pcd

def upsample_pcd_knn(points, colors, upsample_factor, knn_factor, cov_spread_factor):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    new_points = []
    new_colors = [] if colors is not None else None

    for i in range(len(points)):
        _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], knn_factor)
        neighbors = np.asarray(pcd.points)[idx]

        #returns covariance matrix for each group of neighbours
        mean_neighbor = np.mean(neighbors, axis=0)
        centered_neighbors = neighbors - mean_neighbor
        cov_matrix = np.cov(centered_neighbors, rowvar=False)

        #performs element wise operation to apply spread_factor linearly
        scaled_cov = cov_spread_factor * cov_matrix

        for _ in range(upsample_factor):
            noise = np.random.multivariate_normal(np.zeros(3), scaled_cov)
            new_point = mean_neighbor + noise
            new_points.append(new_point)
            #print(new_points)

            if colors is not None:
                interp_color = np.mean(colors[idx], axis=0)
                new_colors.append(interp_color)


    upsampled_points = np.vstack((points, np.array(new_points)))
    upsampled_colors = np.vstack((colors, np.array(new_colors))) if colors is not None else None

    return upsampled_points, upsampled_colors


def save_pcd_path(points, colors, output_path):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)


def main():
    input_file = "C:\\Users\julia\Documents\B30UD_Sensor_Fusion_Project\Point Clouds\lowres_transmission.ply"
    output_file = "C:\\Users\julia\Documents\B30UD_Sensor_Fusion_Project\Point Clouds\\upsampled_trans_4000pts.pcd"

    original_points, original_colors, original_pcd = load_pcd(input_file)

    #all control variables are modified here
    upsampled_points, upsampled_colors = upsample_pcd_knn(
        original_points, original_colors, upsample_factor=19, knn_factor=14, cov_spread_factor=0.4)

    save_pcd_path(upsampled_points, upsampled_colors, output_file)

    upsampled_pcd = o3d.geometry.PointCloud()
    upsampled_pcd.points = o3d.utility.Vector3dVector(upsampled_points)
    if upsampled_colors is not None:
        upsampled_pcd.colors = o3d.utility.Vector3dVector(upsampled_colors)

    o3d.visualization.draw_geometries([upsampled_pcd])
    print(upsampled_pcd)


if __name__ == "__main__":
    main()
