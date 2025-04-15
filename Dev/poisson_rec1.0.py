import open3d as o3d
import numpy as np

from Main.Scripts.Tools import Tools

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(Tools.getPath())
o3d.visualization.draw([pcd])
print(pcd)
#o3d.visualization.draw_geometries([pcd],
#o3d.visualization.draw_geometries([pcd],
#                                  zoom=0.3412,
#                                  front=[0.4257, -0.2125, -0.8795],
#                                  lookat=[2.6172, 2.0475, 1.532],
#                                  up=[-0.0694, -0.9768, 0.2024])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 16)

vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)
mesh_out = mesh.filter_smooth_taubin(number_of_iterations=5)
mesh_out.compute_vertex_normals()
#o3d.visualization.draw_geometries([mesh_out])
#o3d.visualization.draw_geometries([mesh_out], zoom=0.8, mesh_show_wireframe=True)
mesh_out = mesh_out.subdivide_loop(number_of_iterations=2)
print(mesh_out)
o3d.visualization.draw_geometries([mesh_out],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

pcdup = mesh_out.sample_points_uniformly(number_of_points=100)
pcdup = mesh_out.sample_points_poisson_disk(number_of_points=50000, init_factor=10)
print(pcdup)
o3d.visualization.draw_geometries([pcdup],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])



