import numpy as np
import open3d as o3d
import  matplotlib.pyplot as plt
import sys

from matplotlib import cm

if __name__ == '__main__':
    # points = np.load('./logs/efficient_sampling/20220409_215348/pointcloud/pointcloud.npy')
    # points = np.load('./logs/priority3/20220423_144756/pointcloud/3000.npy')
    # points = np.load('./logs/priority3/20220423_150517/pointcloud/3000.npy')
    # points = np.load('./logs/priority3_real/20220423_152823/pointcloud/3000.npy')
    # points = np.load('./logs/long_debug/20220425_203727/pointcloud/3000.npy')
    # points = np.load('./data/surface_points_50000_0.03.npy')
    # points = np.load('./data/points_00001.npy')
    # points = np.load('./data/points.npy')

    filepath = sys.argv[1]
    thresh = int(sys.argv[2])

    points = np.load(filepath)

    points, colors = points[..., :3], points[..., 3:]
    # points = points[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
    # sigmas = sigmas[sigmas > thresh][..., None]
    print(points.shape)

    # points = points[np.broadcast_to(points[..., 2, None], (points.shape[0], 3)) > -0.55].reshape(-1, 3)

    # points = points[np.all(np.isfinite(points), axis=1)]
    # points = points[np.linalg.norm(points, axis=1) < 1000]
    # print(points.shape)
    # print(np.amax(points))
    # print(np.amin(points))

    # points = 1 - (points - np.amin(points)) / (np.amax(points - np.amin(points)))

    # colors = np.broadcast_to(points[..., 0, None], (len(points), 3))
    # colors = points[..., 2]
    # colors = (colors - np.amin(colors)) / (np.amax(colors - np.amin(colors)))
    # colors[colors < 0] = 0
    # colors[colors > 1] = 1
    # colors = cm.get_cmap(plt.get_cmap('jet'))(colors)[..., :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.io.write_point_cloud('./data/pointcloud_latent_embed2.pcd', pcd)

    o3d.visualization.draw([pcd])