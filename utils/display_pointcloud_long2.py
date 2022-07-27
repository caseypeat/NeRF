import numpy as np
import open3d as o3d
import  matplotlib.pyplot as plt

from matplotlib import cm

# from loaders.camera_geometry_loader import meta_camera_geometry_real, camera_geometry_loader_real
from loaders.camera_geometry_loader2 import CameraGeometryLoader

if __name__ == '__main__':

    pcds = []

    # for i in range(40, 280, 20):
    #     images, depths, intrinsics, extrinsics = meta_camera_geometry_real(cfg.scene.scene_path, cfg.scene.frame_range)

    filenames = [
        './logs/west_long_0002/20220607_193718/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_201305/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_204843/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_212433/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_220025/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_223615/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_231201/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220607_234745/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220608_002328/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220608_005920/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220608_013459/pointclouds/pcd/10000.pcd',
        './logs/west_long_0002/20220608_021039/pointclouds/pcd/10000.pcd',
    ]

    pcds = o3d.geometry.PointCloud()

    for i, filename in zip([i for i in range(20, 260, 20)], filenames):

        loader = CameraGeometryLoader(['/mnt/maara/conan_scans/blenheim-21-6-28/30-06_13-05/ROW_349_WEST_LONG_0002/scene.json'], frame_ranges=[[i, i+40]], transforms=[None, None], image_scale=0.25)
        
        pcd = o3d.io.read_point_cloud(filename)

        transform = np.eye(4)
        transform[:3, 3] = loader.translation_center

        pcd = pcd.transform(transform)

        pcds += pcd

    o3d.io.write_point_cloud('./data/test.pcd', pcds)

        # if i == 60:
        #     o3d.io.write_point_cloud('./data/test.pcd', pcds)
            # break

        # print(transform)
        
        # print(extrinsics)
        # offset = np.mean(extrinsics[..., :3, 3], axis=0, keepdims=True) / 2

        # points = np.load(filename)

        # thresh = 100

        # points, sigmas = points[..., :3], points[..., 3]
        # points = points[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
        # sigmas = sigmas[sigmas > thresh][..., None]
        # print(points.shape)

        # points += offset

        # a = np.amax(points[..., 1]) - (np.amax(points[..., 1]) - np.amin(points[..., 1])) * 0.2
        # b = np.amin(points[..., 1]) + (np.amax(points[..., 1]) - np.amin(points[..., 1])) * 0.2

        # points = points[np.broadcast_to(points[..., 1, None], (points.shape[0], 3)) < a].reshape(-1, 3)
        # points = points[np.broadcast_to(points[..., 1, None], (points.shape[0], 3)) > b].reshape(-1, 3)

        # points = points[np.all(np.isfinite(points), axis=1)]
        # points = points[np.linalg.norm(points, axis=1) < 1000]
        # print(points.shape)
        # print(np.amax(points))
        # print(np.amin(points))

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

        # pcds.append(points)

        # o3d.io.write_point_cloud('./data/pointcloud_latent_embed2.pcd', pcd)

    # points = np.concatenate(pcds, axis=0)

    # # colors = np.broadcast_to(points[..., 0, None], (len(points), 3))
    # colors = (points[..., 1]**2 + points[..., 2]**2)**0.5
    # colors = (colors - np.amin(colors)) / (np.amax(colors - np.amin(colors)))
    # colors[colors < 0] = 0
    # colors[colors > 1] = 1
    # colors = cm.get_cmap(plt.get_cmap('jet'))(colors)[..., :3]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw(pcds)