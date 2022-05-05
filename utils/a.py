import numpy as np
import open3d as o3d

if __name__ == '__main__':
    pcd_a = o3d.io.read_point_cloud('./logs/debug2a/20220505_215940/pointcloud/10000.pcd')
    # pcd_a = o3d.io.read_point_cloud('./data/east_west/east2.pcd')
    pcd_b = o3d.io.read_point_cloud('./logs/debug2a/20220505_211126/pointcloud/10000.pcd')
    # pcd_b = o3d.io.read_point_cloud('./data/east_west/west2.pcd')

    transform = np.load('./data/transforms/east_west_refine.npy')

    pcd_a.transform(transform)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.array(pcd_a.points), np.array(pcd_b.points)), axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate((np.array(pcd_a.colors), np.array(pcd_b.colors)), axis=0))
    # o3d.visualization.draw_geometries([pcd])

    # pcd_d = pcd.uniform_down_sample(every_k_points=10)
    # cl, ind = pcd_d.remove_radius_outlier(nb_points=16, radius=0.05)

    o3d.io.write_point_cloud('./data/east_west/both_10000.pcd', pcd)