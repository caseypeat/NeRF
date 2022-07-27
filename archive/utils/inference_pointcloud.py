import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse
import os
import open3d as o3d

from omegaconf import OmegaConf

from loaders.camera_geometry_loader import CameraGeometryLoader

from nets import NeRFNetwork
from renderer import NerfRenderer
from trainer import Trainer
from logger import Logger
from inference import Inferencer


if __name__ == '__main__':

    run = '20220615_203119'

    inputdir = f'./logs/hyper_search/{run}'
    config_path = os.path.join(inputdir, 'config.yaml')
    model_path = os.path.join(inputdir, 'model', '1000.pth')

    output_path = f'./data/hyper/output_04_06_10.pcd'

    # scene_paths = ['/home/casey/Documents/PhD/data/conan_scans/ROW_349_WEST_LONG_0002/scene.json']


    cfg = OmegaConf.load(config_path)

    dataloader = CameraGeometryLoader(
        scene_paths=cfg.scene.scene_paths,
        frame_ranges=cfg.scene.frame_ranges,
        transforms=cfg.scene.transforms,
        image_scale=cfg.scene.image_scale,
        )

    model = NeRFNetwork(
        N = dataloader.images.shape[0],
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

    renderer = NerfRenderer(
        model=model,
        inner_bound=cfg.scene.inner_bound,
        outer_bound=cfg.scene.outer_bound,
        z_bounds=cfg.renderer.z_bounds,
        steps_firstpass=cfg.renderer.steps,
        steps_importance=cfg.renderer.importance_steps,
        alpha_importance=cfg.renderer.alpha,
    )

    inferencer = Inferencer(
        renderer=renderer,
        n_rays=cfg.trainer.n_rays,
        image_num=cfg.inference.image_num,
        )

    n, h, w, K, E, _, _ = dataloader.get_pointcloud_batch(cams=[2, 3, 4], freq=30)
    pointcloud = inferencer.extract_surface_geometry(n, h, w, K, E, max_variance=0.05)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud['points'])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud['colors'])

    transform = np.eye(4)
    transform[:3, 3] = dataloader.translation_center
    pcd = pcd.transform(transform)

    o3d.io.write_point_cloud(output_path, pcd)