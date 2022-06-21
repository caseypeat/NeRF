import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import open3d as o3d

from loaders.camera_geometry_loader import camera_geometry_loader, camera_geometry_loader_real, meta_camera_geometry, meta_camera_geometry_real
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from inference import Inferencer

from misc import color_depthmap

from config import cfg


if __name__ == '__main__':

    thresh = 100

    images_a, depths_a, intrinsics_a, extrinsics_a = meta_camera_geometry_real(cfg.scene.scene_path_a, cfg.scene.frame_range)
    images_b, depths_b, intrinsics_b, extrinsics_b = meta_camera_geometry_real(cfg.scene.scene_path_b, cfg.scene.frame_range)

    transform = np.load('./data/transforms/east_west_refine.npy')
    extrinsics_a = torch.Tensor(transform) @ extrinsics_a

    images = torch.cat([images_a, images_b], dim=0)
    depths = None
    intrinsics = torch.cat([intrinsics_a, intrinsics_b], dim=0)
    extrinsics = torch.cat([extrinsics_a, extrinsics_b], dim=0)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - torch.mean(extrinsics[..., :3, 3], dim=0, keepdims=True)

    N, H, W, C = images.shape

    model = NeRFNetwork(
        # renderer
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        # net
        N=N
    ).to('cuda')

    model.load_state_dict(torch.load(cfg.nets.load_path))

    inferencer = Inferencer(model=model)

    points, colors = inferencer.extract_geometry(N, H, W, intrinsics, extrinsics)
    points = points.cpu().detach().numpy()
    colors = colors.cpu().detach().numpy()

    # points, sigmas = points[..., :3], points[..., 3]
    # points = points[np.broadcast_to(sigmas[..., None], (sigmas.shape[0], 3)) > thresh].reshape(-1, 3)
    # sigmas = sigmas[sigmas > thresh][..., None]
    # print(points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud('./data/pointcloud_both_dense_color.pcd', pcd)