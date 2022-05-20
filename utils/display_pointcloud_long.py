import numpy as np
import open3d as o3d
import  matplotlib.pyplot as plt

from matplotlib import cm

from loaders.camera_geometry_loader import meta_camera_geometry_real, camera_geometry_loader_real

if __name__ == '__main__':

    pcds = []

    # for i in range(40, 280, 20):
    #     images, depths, intrinsics, extrinsics = meta_camera_geometry_real(cfg.scene.scene_path, cfg.scene.frame_range)

    filenames = [
        './logs/long/20220425_205329/pointcloud/3000.npy',
        './logs/long/20220425_210453/pointcloud/3000.npy',
        './logs/long/20220425_211608/pointcloud/3000.npy',
        './logs/long/20220425_212701/pointcloud/3000.npy',
        './logs/long/20220425_213756/pointcloud/3000.npy',
        './logs/long/20220425_214923/pointcloud/3000.npy',
        './logs/long/20220425_220050/pointcloud/3000.npy',
        './logs/long/20220425_221305/pointcloud/3000.npy',
        './logs/long/20220425_222457/pointcloud/3000.npy',
        './logs/long/20220425_223647/pointcloud/3000.npy',
        './logs/long/20220425_224832/pointcloud/3000.npy',
        './logs/long/20220425_230012/pointcloud/3000.npy',
    ]

    for i, filename in zip([i for i in range(40, 280, 20)], filenames):

        _, _, _, extrinsics, _ = camera_geometry_loader_real('/home/casey/Documents/PhD/data/conan_scans/ROW_349_WEST_LONG_0002/scene.json', image_scale=0.25, frame_range=(i, i+20))
        # print(extrinsics)
        offset = np.mean(extrinsics[..., :3, 3], axis=0, keepdims=True) / 2

        points = np.load(filename)

        thresh = 100

        points, sigmas = points[..., :3], points[..., 3]
        points = points[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
        sigmas = sigmas[sigmas > thresh][..., None]
        print(points.shape)

        points += offset

        a = np.amax(points[..., 1]) - (np.amax(points[..., 1]) - np.amin(points[..., 1])) * 0.2
        b = np.amin(points[..., 1]) + (np.amax(points[..., 1]) - np.amin(points[..., 1])) * 0.2

        points = points[np.broadcast_to(points[..., 1, None], (points.shape[0], 3)) < a].reshape(-1, 3)
        points = points[np.broadcast_to(points[..., 1, None], (points.shape[0], 3)) > b].reshape(-1, 3)

        # points = points[np.all(np.isfinite(points), axis=1)]
        # points = points[np.linalg.norm(points, axis=1) < 1000]
        # print(points.shape)
        print(np.amax(points))
        print(np.amin(points))

        # points = 1 - (points - np.amin(points)) / (np.amax(points - np.amin(points)))

        # colors = np.broadcast_to(points[..., 0, None], (len(points), 3))
        # colors = points[..., 2]
        # colors = (colors - np.amin(colors)) / (np.amax(colors - np.amin(colors))) * 2 - 0.5
        # colors[colors < 0] = 0
        # colors[colors > 1] = 1
        # colors = cm.get_cmap(plt.get_cmap('jet'))(colors)[..., :3]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(np.random.rand(3)[None, ...])
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        pcds.append(points)

        # o3d.io.write_point_cloud('./data/pointcloud_latent_embed2.pcd', pcd)

    points = np.concatenate(pcds, axis=0)

    # colors = np.broadcast_to(points[..., 0, None], (len(points), 3))
    colors = (points[..., 1]**2 + points[..., 2]**2)**0.5
    colors = (colors - np.amin(colors)) / (np.amax(colors - np.amin(colors)))
    colors[colors < 0] = 0
    colors[colors > 1] = 1
    colors = cm.get_cmap(plt.get_cmap('jet'))(colors)[..., :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw(pcd)