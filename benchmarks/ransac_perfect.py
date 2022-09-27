import numpy as np
import torch
import open3d as o3d
import os
import time

from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy

from loaders.camera_geometry_loader_re2 import CameraGeometryLoader
from render import get_rays
from metrics import pose_inv_error

from allign.ransac import global_allign


def load_pointcloud(scan_path):
    dataloader = CameraGeometryLoader(
        scan_paths=[scan_path],
        scan_pose_paths=[None],
        frame_ranges=[None],
        frame_strides=[None],
        image_scale=0.25,
        load_images_bool=True,
        load_depths_bool=True,
        )

    n, h, w, K, E, color, _, depth = dataloader.get_pointcloud_batch(
        cams=[0, 1, 2, 3, 4, 5],
        freq=3,
        side_margin=0,
        device="cpu"
    )
    prefix = depth.shape
    n_f = n.reshape((-1,))
    h_f = h.reshape((-1,))
    w_f = w.reshape((-1,))
    K_f = K.reshape((-1, 3, 3))
    E_f = E.reshape((-1, 4, 4))
    color_f = color.reshape((-1, 3))
    depth_f = depth.reshape((-1,))

    max_depth = 1
    n_fm = n_f[depth_f < max_depth]
    h_fm = h_f[depth_f < max_depth]
    w_fm = w_f[depth_f < max_depth]
    K_fm = K_f[depth_f < max_depth]
    E_fm = E_f[depth_f < max_depth]
    color_fm = color_f[depth_f < max_depth]
    depth_fm = depth_f[depth_f < max_depth]

    rays_o_fm, rays_d_fm = get_rays(h_fm, w_fm, K_fm, E_fm)

    points_fm = rays_o_fm + rays_d_fm * depth_fm[:, None]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_fm)
    pcd.colors = o3d.utility.Vector3dVector(color_fm)

    return pcd


def log_str(log_filepath, string):
    with open(log_filepath, 'a') as f:
        dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        output_str = f'[{dt}] {string}\n'
        print(output_str, end='')
        f.write(output_str)


if __name__ == "__main__":

    num = 7

    os.makedirs(f"./data/pose_estimates/perfect/vine_C1_{num}", exist_ok=True)


    pcd_a = load_pointcloud(f"/home/cpe44/data/vine_C1_{num}/front/cameras.json")
    pcd_b = load_pointcloud(f"/home/cpe44/data/vine_C1_{num}/back/cameras.json")

    ransac_result, icp_result = global_allign(pcd_a, pcd_b, voxel_size=0.01)

    transform_ransac = torch.Tensor(ransac_result.transformation)
    transform_icp = torch.Tensor(icp_result.transformation)

    error_rot_ransac, error_trans_ransac = pose_inv_error(transform_ransac, torch.eye(4))
    error_rot_icp, error_trans_icp = pose_inv_error(transform_icp, torch.eye(4))

    log_str(f"./data/pose_estimates/perfect/vine_C1_{num}/events.log", f"Rotation Error (degrees) - RANSAC: {torch.rad2deg(error_rot_ransac).item():.4f}")
    log_str(f"./data/pose_estimates/perfect/vine_C1_{num}/events.log", f"Translation Error (mm) - RANSAC: {(error_trans_ransac * 1000).item():.4f}")

    log_str(f"./data/pose_estimates/perfect/vine_C1_{num}/events.log", f"Rotation Error (degrees) - ICP: {torch.rad2deg(error_rot_icp).item():.4f}")
    log_str(f"./data/pose_estimates/perfect/vine_C1_{num}/events.log", f"Translation Error (mm) - ICP: {(error_trans_icp * 1000).item():.4f}")



    np.save(f"./data/pose_estimates/perfect/vine_C1_{num}/ransac.npy", transform_ransac.numpy())
    np.save(f"./data/pose_estimates/perfect/vine_C1_{num}/icp.npy", transform_icp.numpy())

    pcd_a_ransac = deepcopy(pcd_a)
    pcd_a_ransac.transform(transform_ransac.numpy())
    pcd_both_ransac = o3d.geometry.PointCloud()
    pcd_both_ransac += pcd_a_ransac
    pcd_both_ransac += pcd_b
    o3d.io.write_point_cloud(f"./data/pose_estimates/perfect/vine_C1_{num}/ransac.pcd", pcd_both_ransac)

    pcd_a_icp = deepcopy(pcd_a)
    pcd_a_icp.transform(transform_icp.numpy())
    pcd_both_icp = o3d.geometry.PointCloud()
    pcd_both_icp += pcd_a_icp
    pcd_both_icp += pcd_b
    o3d.io.write_point_cloud(f"./data/pose_estimates/perfect/vine_C1_{num}/icp.pcd", pcd_both_icp)