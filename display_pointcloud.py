import numpy as np
import open3d as o3d

if __name__ == '__main__':
    points = np.load('./data/points_100_aneal.npy')
    # points = np.load('./data/points_00001.npy')
    # points = np.load('./data/points.npy')

    thresh = 50

    points, sigmas = points[..., :3], points[..., 3]
    points = points[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas[sigmas > thresh][..., None]
    print(points.shape)

    # points = points[np.all(np.isfinite(points), axis=1)]
    # points = points[np.linalg.norm(points, axis=1) < 1000]
    # print(points.shape)
    print(np.amax(points))
    print(np.amin(points))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw([pcd])