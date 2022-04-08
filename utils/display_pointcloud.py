import numpy as np
import open3d as o3d
import  matplotlib.pyplot as plt

from matplotlib import cm

if __name__ == '__main__':
    points = np.load('./logs/alpha/20220408_141949/pointcloud/10000.npy')
    # points = np.load('./data/points_00001.npy')
    # points = np.load('./data/points.npy')

    thresh = 50

    points, sigmas = points[..., :3], points[..., 3]
    points = points[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas[sigmas > thresh][..., None]
    print(points.shape)

    points = points[np.broadcast_to(points[..., 0, None], (points.shape[0], 3)) > -0.5].reshape(-1, 3)

    # points = points[np.all(np.isfinite(points), axis=1)]
    # points = points[np.linalg.norm(points, axis=1) < 1000]
    # print(points.shape)
    print(np.amax(points))
    print(np.amin(points))

    # points = 1 - (points - np.amin(points)) / (np.amax(points - np.amin(points)))

    # colors = np.broadcast_to(points[..., 0, None], (len(points), 3))
    colors = points[..., 0]
    colors = (colors - np.amin(colors)) / (np.amax(colors - np.amin(colors)))
    colors = cm.get_cmap(plt.get_cmap('jet'))(colors)[..., :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud('./data/pointcloud.pcd', pcd)

    o3d.visualization.draw([pcd])