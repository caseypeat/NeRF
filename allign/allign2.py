import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import os

from omegaconf import OmegaConf

from tqdm import tqdm

from loaders.camera_geometry_loader import CameraGeometryLoader
from renderer import NerfRenderer
from nets import NeRFNetwork

from misc import configurator

from allign.ransac import global_allign
from allign.rotation import vec2skew, Exp, matrix2xyz_extrinsic
from allign.trainer import TrainerPose
from allign.extract_dense_geometry import ExtractDenseGeometry
from allign.logger import AllignLogger
from allign.metrics import Measure


def calculate_euler_angle_from_rotation_matrix(R):
    theta = np.arctan2(R[2, 1], R[2, 2])
    phi = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([theta, phi, psi])
    


class Transform(nn.Module):
    def __init__(self, init_transform):
        super().__init__()

        self.init_transform = torch.nn.Parameter(init_transform, requires_grad=False)

        self.R = torch.nn.Parameter(torch.zeros((3,)), requires_grad=True)
        self.T = torch.nn.Parameter(torch.zeros((3,)), requires_grad=True)

    def forward(self, xyzs_a):

        transform = torch.eye(4, dtype=torch.float32, device='cuda')
        transform[:3, :3] = Exp(self.R)
        transform[:3, 3] = self.T

        b = xyzs_a.reshape(-1, 3)
        b1 = torch.cat([b, b.new_ones((b.shape[0], 1))], dim=1)
        b2 = (self.init_transform @ b1.T).T
        b3 = (transform @ b2.T).T
        xyzs_b = b3[:, :3].reshape(*xyzs_a.shape)

        return xyzs_b


if __name__ == '__main__':

    cfg = configurator('./allign/config.yaml')

    logger = AllignLogger(cfg.log_dir, cfg)

    # if not os.path.exists(cfg.output_dir):
    #     os.mkdir(cfg.output_dir)

    cfg_a = OmegaConf.load(f'{cfg.input_paths.logdir_a}/config.yaml')
    cfg_b = OmegaConf.load(f'{cfg.input_paths.logdir_b}/config.yaml')

    dataloader_a = CameraGeometryLoader(
        cfg_a.scene.scene_paths,
        cfg_a.scene.frame_ranges,
        cfg_a.scene.transforms,
        cfg_a.scene.image_scale,
        )

    dataloader_b = CameraGeometryLoader(
        cfg_b.scene.scene_paths,
        cfg_b.scene.frame_ranges,
        cfg_b.scene.transforms,
        cfg_b.scene.image_scale,
        )

    pcd_a = o3d.io.read_point_cloud(cfg.input_paths.pointcloud_a)
    pcd_b = o3d.io.read_point_cloud(cfg.input_paths.pointcloud_b)

    model_a = NeRFNetwork(
        N = len(dataloader_a.images),
        encoding_precision=cfg_a.nets.encoding.precision,
        encoding_n_levels=cfg_a.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg_a.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg_a.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg_a.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg_a.nets.sigma.hidden_dim,
        sigma_num_layers=cfg_a.nets.sigma.num_layers,
        encoding_dir_precision=cfg_a.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg_a.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg_a.nets.encoding_dir.degree,
        latent_embedding_dim=cfg_a.nets.latent_embedding.features,
        color_hidden_dim=cfg_a.nets.color.hidden_dim,
        color_num_layers=cfg_a.nets.color.num_layers,
    ).to('cuda')
    model_a.load_state_dict(torch.load(f'{cfg.input_paths.model_a}'))

    model_b = NeRFNetwork(
        N = len(dataloader_b.images),
        encoding_precision=cfg_b.nets.encoding.precision,
        encoding_n_levels=cfg_b.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg_b.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg_b.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg_b.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg_b.nets.sigma.hidden_dim,
        sigma_num_layers=cfg_b.nets.sigma.num_layers,
        encoding_dir_precision=cfg_b.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg_b.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg_b.nets.encoding_dir.degree,
        latent_embedding_dim=cfg_b.nets.latent_embedding.features,
        color_hidden_dim=cfg_b.nets.color.hidden_dim,
        color_num_layers=cfg_b.nets.color.num_layers,
    ).to('cuda')
    model_b.load_state_dict(torch.load(f'{cfg.input_paths.model_b}'))


    pcd_a_rs = pcd_a.uniform_down_sample(100)
    pcd_b_rs = pcd_b.uniform_down_sample(100)

    a = np.array(pcd_a_rs.points)
    thresh1 = (np.amax(a[:, 2]) - np.amin(a[:, 2])) * 0.25 + np.amin(a[:, 2])
    thresh2 = (np.amax(a[:, 2]) - np.amin(a[:, 2])) * 0.75 + np.amin(a[:, 2])
    a = a[np.broadcast_to(a[:, 2, None], (a.shape[0], 3)) > thresh1].reshape(-1, 3)
    a = a[np.broadcast_to(a[:, 2, None], (a.shape[0], 3)) < thresh2].reshape(-1, 3)

    thresh1 = (np.amax(a[:, 1]) - np.amin(a[:, 1])) * 0.25 + np.amin(a[:, 1])
    thresh2 = (np.amax(a[:, 1]) - np.amin(a[:, 1])) * 0.75 + np.amin(a[:, 1])
    a = a[np.broadcast_to(a[:, 1, None], (a.shape[0], 3)) > thresh1].reshape(-1, 3)
    a = a[np.broadcast_to(a[:, 1, None], (a.shape[0], 3)) < thresh2].reshape(-1, 3)

    # thresh = (np.amax(a[:, 1]) - np.amin(a[:, 1])) * 0.5 + np.amin(a[:, 1])
    # a = a[np.broadcast_to(a[:, 1, None], (a.shape[0], 3)) < thresh].reshape(-1, 3)


    pcd_a_rs = o3d.geometry.PointCloud()
    pcd_a_rs.points = o3d.utility.Vector3dVector(a)

    # 0 - x: front to back
    # 1 - y: left to right
    # 2 - z: up to down
    b = np.array(pcd_b_rs.points)
    thresh1 = (np.amax(b[:, 2]) - np.amin(b[:, 2])) * 0.25 + np.amin(b[:, 2])
    thresh2 = (np.amax(b[:, 2]) - np.amin(b[:, 2])) * 0.75 + np.amin(b[:, 2])
    b = b[np.broadcast_to(b[:, 2, None], (b.shape[0], 3)) > thresh1].reshape(-1, 3)
    b = b[np.broadcast_to(b[:, 2, None], (b.shape[0], 3)) < thresh2].reshape(-1, 3)

    thresh1 = (np.amax(b[:, 1]) - np.amin(b[:, 1])) * 0.25 + np.amin(b[:, 1])
    thresh2 = (np.amax(b[:, 1]) - np.amin(b[:, 1])) * 0.75 + np.amin(b[:, 1])
    b = b[np.broadcast_to(b[:, 1, None], (b.shape[0], 3)) > thresh1].reshape(-1, 3)
    b = b[np.broadcast_to(b[:, 1, None], (b.shape[0], 3)) < thresh2].reshape(-1, 3)

    # thresh = (np.amax(b[:, 1]) - np.amin(b[:, 1])) * 0.5 + np.amin(b[:, 1])
    # b = b[np.broadcast_to(b[:, 1, None], (b.shape[0], 3)) < thresh].reshape(-1, 3)


    pcd_b_rs = o3d.geometry.PointCloud()
    pcd_b_rs.points = o3d.utility.Vector3dVector(b)

    logger.save_pointcloud(pcd_a_rs, 'pcd_ransac_a')
    logger.save_pointcloud(pcd_b_rs, 'pcd_ransac_b')

    # result_ransac, result_icp = global_allign(pcd_dense_a, pcd_dense_b, voxel_size=0.01)
    result_ransac, result_icp = global_allign(pcd_a_rs, pcd_b_rs, voxel_size=0.01)

    # init_transform = np.eye(4)
    # init_transform[:3, 3] = result_ransac.transformation[:3, 3]
    # init_transform[:3, :3] = result_ransac.transformation[:3, :3]
    # init_transform = torch.Tensor(init_transform)

    init_transform = torch.Tensor(np.array(result_ransac.transformation))
    transform = Transform(init_transform).cuda()

    ransac_transform = torch.Tensor(np.array(result_ransac.transformation))
    transform_ransac = Transform(ransac_transform).cuda()

    icp_transform = torch.Tensor(np.array(result_icp.transformation))
    transform_icp = Transform(icp_transform).cuda()

    renderer = NerfRenderer(
        model=model_a,
        inner_bound=cfg_a.scene.inner_bound,
        outer_bound=cfg_a.scene.outer_bound,
        z_bounds=cfg.renderer.z_bounds,
        steps_firstpass=cfg.renderer.steps_firstpass,
        steps_importance=cfg_a.renderer.importance_steps,
        alpha_importance=cfg_a.renderer.alpha,
    )

    measure = Measure(
        # transform=transform,
        renderer=renderer,
        depth_thresh=0.7,
        translation_center_a=dataloader_a.translation_center,
        translation_center_b=dataloader_b.translation_center,)

    trainer = TrainerPose(
        logger=logger,
        transform=transform,
        transform_ransac=transform_ransac,
        transform_icp=transform_icp,
        model_a=model_a,
        model_b=model_b,
        pointcloud_a=pcd_a,
        pointcloud_b=pcd_b,
        dataloader_a=dataloader_a,
        dataloader_b=dataloader_b,
        renderer=renderer,
        measure=measure,
        iters_per_epoch=cfg.iters_per_epoch,
        num_epochs=cfg.num_epochs,
        n_rays=cfg.n_rays
        )

    trainer.train()