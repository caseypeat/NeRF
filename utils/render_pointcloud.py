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

    images_a, depths_a, intrinsics_a, extrinsics_a = meta_camera_geometry_real(cfg.scene.scene_path_a, cfg.scene.frame_range)
    images_b, depths_b, intrinsics_b, extrinsics_b = meta_camera_geometry_real(cfg.scene.scene_path_b, cfg.scene.frame_range)

    transform = np.load('./data/transforms/east_west.npy')
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

    ids = []
    for i in range(N):
        # if i % 6 == 2 or i % 6 == 3 or i % 6 == 4 or i % 6 == 1:
        if (i > int(images_a.shape[0]*0.25) and i < int(images_a.shape[0]*0.75)) or (i > images_a.shape[0] + int(images_b.shape[0]*0.25) and i < images_a.shape[0] + int(images_b.shape[0]*0.75)):
            if i % 6 == 2 or i % 6 == 3:
                if i//6 % 10 == 0:
                    ids.append(i)
    print(len(ids))
    ids = torch.Tensor(np.array(ids)).to(int)
    # ids = torch.Tensor(np.array([303, 304])).to(int)

    points, colors = inferencer.extract_geometry_rays(H, W, intrinsics[ids], extrinsics[ids], thresh=100)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud('./data/poincloud_both_10000_miss.pcd', pcd)