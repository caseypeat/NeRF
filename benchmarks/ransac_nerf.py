import numpy as np
import torch
import open3d as o3d
import os
import time

from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from loaders.camera_geometry_loader_re2 import CameraGeometryLoader
from render import get_rays
from metrics import pose_inv_error
from copy import deepcopy

from allign.ransac import global_allign

from render import Render
from nets import NeRFNetwork, NeRFCoordinateWrapper
from inference import generate_pointcloud


def load_pointcloud_nerf(cfg_path, model_path):
    cfg = OmegaConf.load(cfg_path)
    dataloader = CameraGeometryLoader(
        scan_paths=cfg.scan.scan_paths,
        scan_pose_paths=cfg.scan.scan_pose_paths,
        frame_ranges=cfg.scan.frame_ranges,
        frame_strides=cfg.scan.frame_strides,
        image_scale=cfg.scan.image_scale/2,
        load_images_bool=False,
        load_depths_bool=False,
        )

    model = NeRFNetwork(
        N = len(dataloader.extrinsics),
        encoding_precision=cfg.nets.encoding.precision,
        encoding_n_levels=cfg.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg.nets.sigma.hidden_dim,
        sigma_num_layers=cfg.nets.sigma.num_layers,
        encoding_dir_precision=cfg.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg.nets.encoding_dir.degree,
        latent_embedding_dim=cfg.nets.latent_embedding.features,
        color_hidden_dim=cfg.nets.color.hidden_dim,
        color_num_layers=cfg.nets.color.num_layers,
    ).to('cuda')
    model.load_state_dict(torch.load(model_path))

    model_coord = NeRFCoordinateWrapper(
        model=model,
        transform=None,
        inner_bound=cfg.scan.inner_bound,
        outer_bound=cfg.scan.outer_bound,
        translation_center=dataloader.translation_center
    )

    renderer = Render(
        models=model_coord,
        steps_firstpass=[256],
        z_bounds=[0.1, 1],
        steps_importance=128,
        alpha_importance=0.2,
    )

    n, h, w, K, E, color, _, _ = dataloader.get_pointcloud_batch(
        cams=[0, 1, 2, 3, 4, 5],
        freq=3,
        side_margin=0,
        device="cpu"
    )
    prefix = n.shape
    n_f = n.reshape((-1,))
    h_f = h.reshape((-1,))
    w_f = w.reshape((-1,))
    K_f = K.reshape((-1, 3, 3))
    E_f = E.reshape((-1, 4, 4))

    pointcloud = generate_pointcloud(renderer, n_f, h_f, w_f, K_f, E_f, 4096, 0.05, 0.5)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud["points"])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud["colors"])

    return pcd


def get_file_names(num):
    time_stamp_front = os.listdir(f"./logs/synthetic_allign_depth/vine_C1_{num}/front/")[0]
    vine_front_cfg = f"./logs/synthetic_allign_depth/vine_C1_{num}/front/{time_stamp_front}/config.yaml"
    vine_front_model = f"./logs/synthetic_allign_depth/vine_C1_{num}/front/{time_stamp_front}/model/10000.pth"

    time_stamp_back = os.listdir(f"./logs/synthetic_allign_depth/vine_C1_{num}/back/")[0]
    vine_back_cfg = f"./logs/synthetic_allign_depth/vine_C1_{num}/back/{time_stamp_back}/config.yaml"
    vine_back_model = f"./logs/synthetic_allign_depth/vine_C1_{num}/back/{time_stamp_back}/model/10000.pth"

    return vine_front_cfg, vine_front_model, vine_back_cfg, vine_back_model


def log_str(log_filepath, string):
    with open(log_filepath, 'a') as f:
        dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        output_str = f'[{dt}] {string}\n'
        print(output_str, end='')
        f.write(output_str)


if __name__ == "__main__":

    num = 2

    os.makedirs(f"./data/pose_estimates/nerf/vine_C1_{num}", exist_ok=True)


    vine_front_cfg, vine_front_model, vine_back_cfg, vine_back_model = get_file_names(num)

    pcd_a = load_pointcloud_nerf(vine_front_cfg, vine_front_model)
    pcd_b = load_pointcloud_nerf(vine_back_cfg, vine_back_model)

    ransac_result, icp_result = global_allign(pcd_a, pcd_b, voxel_size=0.01)

    transform_ransac = torch.Tensor(ransac_result.transformation)
    transform_icp = torch.Tensor(icp_result.transformation)

    error_rot_ransac, error_trans_ransac = pose_inv_error(transform_ransac, torch.eye(4))
    error_rot_icp, error_trans_icp = pose_inv_error(transform_icp, torch.eye(4))

    log_str(f"./data/pose_estimates/nerf/vine_C1_{num}/events.log", f"Rotation Error (degrees) - RANSAC: {torch.rad2deg(error_rot_ransac).item():.4f}")
    log_str(f"./data/pose_estimates/nerf/vine_C1_{num}/events.log", f"Translation Error (mm) - RANSAC: {(error_trans_ransac * 1000).item():.4f}")

    log_str(f"./data/pose_estimates/nerf/vine_C1_{num}/events.log", f"Rotation Error (degrees) - ICP: {torch.rad2deg(error_rot_icp).item():.4f}")
    log_str(f"./data/pose_estimates/nerf/vine_C1_{num}/events.log", f"Translation Error (mm) - ICP: {(error_trans_icp * 1000).item():.4f}")



    np.save(f"./data/pose_estimates/nerf/vine_C1_{num}/ransac.npy", transform_ransac.numpy())
    np.save(f"./data/pose_estimates/nerf/vine_C1_{num}/icp.npy", transform_icp.numpy())

    pcd_a_ransac = deepcopy(pcd_a)
    pcd_a_ransac.transform(transform_ransac.numpy())
    pcd_both_ransac = o3d.geometry.PointCloud()
    pcd_both_ransac += pcd_a_ransac
    pcd_both_ransac += pcd_b
    o3d.io.write_point_cloud(f"./data/pose_estimates/nerf/vine_C1_{num}/ransac.pcd", pcd_both_ransac)

    pcd_a_icp = deepcopy(pcd_a)
    pcd_a_icp.transform(transform_icp.numpy())
    pcd_both_icp = o3d.geometry.PointCloud()
    pcd_both_icp += pcd_a_icp
    pcd_both_icp += pcd_b
    o3d.io.write_point_cloud(f"./data/pose_estimates/nerf/vine_C1_{num}/icp.pcd", pcd_both_icp)