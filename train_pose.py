import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d

from loaders.camera_geometry_loader import meta_camera_geometry_real
from utils.allign_pointclouds import allign_pointclouds

from nets import NeRFNetwork

from config import cfg

if __name__ == '__main__':

    thresh = 100

    # images, depths, intrinsics, extrinsics = meta_camera_geometry_real(cfg.scene.scene_path, cfg.scene.frame_range)

    N_a = 1176
    model_a = NeRFNetwork(
        # renderer
        intrinsics=torch.zeros((N_a, 3, 3)),
        extrinsics=torch.zeros((N_a, 4, 4)),

        # net
        N = N_a
    ).to('cuda')
    model_a.load_state_dict(torch.load('./logs/east_west/20220429_175050/model/6000.pth'))

    points_a = np.load('./logs/east_west/20220429_175050/pointcloud/6000.npy')
    points_a, sigmas_a = points_a[..., :3], points_a[..., 3]
    points_a = points_a[np.broadcast_to(sigmas_a[..., None], (sigmas_a.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas_a[sigmas_a > thresh][..., None]
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(points_a)
    pcd_a.paint_uniform_color([1, 0, 0])
    
    N_b = 972
    model_b = NeRFNetwork(
        # renderer
        intrinsics=torch.zeros((N_b, 3, 3)),
        extrinsics=torch.zeros((N_b, 4, 4)),

        # net
        N = N_b
    ).to('cuda')
    model_b.load_state_dict(torch.load('./logs/east_west/20220429_183352/model/6000.pth'))

    points_b = np.load('./logs/east_west/20220429_183352/pointcloud/6000.npy')
    points_b, sigmas_b = points_b[..., :3], points_b[..., 3]
    points_b = points_b[np.broadcast_to(sigmas_b[..., None], (sigmas_b.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas_b[sigmas_b > thresh][..., None]
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(points_b)
    pcd_b.paint_uniform_color([0, 1, 1])


    result_ransac, result_icp = allign_pointclouds(pcd_a, pcd_b, voxel_size=0.01)
    # pcd_a.transform(result_ransac.transformation)
    # o3d.visualization.draw_geometries([pcd_a, pcd_b])
    pcd_a.transform(result_icp.transformation)
    o3d.visualization.draw_geometries([pcd_a, pcd_b])
    
    # np.save('./data/transforms/east_west.npy', result_icp.transformation)