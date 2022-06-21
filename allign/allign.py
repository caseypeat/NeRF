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

    if not os.path.exists(cfg.output_dir):
        os.mkdir(cfg.output_dir)

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

    dense_inference_a = ExtractDenseGeometry(
        model_a, 
        dataloader_a, 
        cfg.extract_geometry_values.mask_res, 
        cfg.extract_geometry_values.voxel_res, 
        cfg_a.renderer.importance_steps*cfg.n_rays, 
        cfg.extract_geometry_values.sigma_thresh, 
        cfg_a.scene.outer_bound)
    pcd_dense_a = dense_inference_a.generate_dense_pointcloud()

    dense_inference_b = ExtractDenseGeometry(
        model_b, 
        dataloader_b, 
        cfg.extract_geometry_values.mask_res, 
        cfg.extract_geometry_values.voxel_res, 
        cfg_b.renderer.importance_steps*cfg.n_rays, 
        cfg.extract_geometry_values.sigma_thresh, 
        cfg_b.scene.outer_bound)
    pcd_dense_b = dense_inference_b.generate_dense_pointcloud()

    result_ransac, result_icp = global_allign(pcd_dense_a, pcd_dense_b, voxel_size=0.01)

    init_transform = torch.Tensor(np.array(result_ransac.transformation))

    np.save(f'{cfg.output_dir}/transform_ransac.npy', init_transform)

    transform = Transform(init_transform).cuda()

    renderer = NerfRenderer(
        model=model_a,
        inner_bound=cfg_a.scene.inner_bound,
        outer_bound=cfg_a.scene.outer_bound,
        z_bounds=cfg.renderer.z_bounds,
        steps_firstpass=cfg.renderer.steps_firstpass,
        steps_importance=cfg_a.renderer.importance_steps,
        alpha_importance=cfg_a.renderer.alpha,
    )

    trainer = TrainerPose(
        # translation_init=init_transform,
        transform=transform,
        model_a=model_a,
        model_b=model_b,
        dataloader_a=dataloader_a,
        renderer=renderer,
        num_iters=cfg.num_iters,
        n_rays=cfg.n_rays
        )

    trainer.train()

    transform_r = torch.eye(4, dtype=torch.float32, device='cuda')
    transform_r[:3, :3] = Exp(transform.R)
    transform_r[:3, 3] = transform.T
    transform_comb = np.array(transform_r.detach().cpu()) @ np.array(transform.init_transform.detach().cpu())

    pcd_a.transform(np.array(transform.init_transform.detach().cpu()))
    # o3d.visualization.draw_geometries([pcd_a, pcd_b])

    pcd_both = o3d.geometry.PointCloud()
    pcd_both += pcd_a
    pcd_both += pcd_b
    o3d.io.write_point_cloud(f'{cfg.output_dir}/ransac_allign.pcd', pcd_both)

    pcd_a.transform(np.array(transform_r.detach().cpu()))
    # o3d.visualization.draw_geometries([pcd_a, pcd_b])

    pcd_both = o3d.geometry.PointCloud()
    pcd_both += pcd_a
    pcd_both += pcd_b
    o3d.io.write_point_cloud(f'{cfg.output_dir}/zeronerf_allign.pcd', pcd_both)

    o3d.io.write_point_cloud(f'{cfg.output_dir}/pointcloud_a.pcd', pcd_a)
    o3d.io.write_point_cloud(f'{cfg.output_dir}/pointcloud_b.pcd', pcd_b)

    np.save(f'{cfg.output_dir}/transform_zeronerf.npy', transform_comb)