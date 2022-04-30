import numpy as np
import open3d as o3d

if __name__ == '__main__':
    pcd_a = o3d.io.read_point_cloud('./data/pointcloud_6000_east.pcd')
    pcd_b = o3d.io.read_point_cloud('./data/pointcloud_6000_west.pcd')

    transform = np.load('./data/transforms/east_west.npy')

    pcd_a.transform(transform)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.array(pcd_a.points), np.array(pcd_b.points)), axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate((np.array(pcd_a.colors), np.array(pcd_b.colors)), axis=0))
    # o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud('./data/pointcloud_6000_east_west.pcd', pcd)