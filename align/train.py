from calendar import c
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import os

from omegaconf import OmegaConf, DictConfig

from tqdm import tqdm

from loaders.camera_geometry_loader_re2 import CameraGeometryLoader
from render import Render
from nets import NeRFNetwork, NeRFCoordinateWrapper, Transform
from rotation import Exp, matrix2xyz_extrinsic

from align.ransac import global_align
from align.trainer import TrainerAlign
from align.logger import AlignLogger


def gen_random_transform(R_var, T_var):
    R = torch.rand((3), dtype=torch.float32, device='cuda')
    R = R / torch.norm(R) * R_var

    T = torch.rand((3), dtype=torch.float32, device='cuda')
    T = T / torch.norm(T) * T_var

    transform = torch.eye(4, dtype=torch.float32, device='cuda')
    transform[:3, :3] = Exp(R)
    transform[:3, 3] = T
    return transform


# @hydra.main(version_base=None, config_path="./configs/vines", config_name="C1_0")
@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    cfg_a = OmegaConf.load(cfg.input_paths.config_path_a)
    cfg_b = OmegaConf.load(cfg.input_paths.config_path_b)

    logger = AlignLogger(cfg.output_paths.log, cfg)

    # transform = Transform(gen_random_transform(0.01, 0.01)).to('cuda')
    transform = Transform(torch.Tensor(np.load(cfg.input_paths.transform_init))).to('cuda')

    dataloader_a = CameraGeometryLoader(
        scan_paths=cfg_a.scan.scan_paths,
        scan_pose_paths=cfg_a.scan.scan_pose_paths,
        frame_ranges=cfg_a.scan.frame_ranges,
        frame_strides=cfg_a.scan.frame_strides,
        image_scale=cfg_a.scan.image_scale,
        load_images_bool=False,
        load_depths_bool=False,
        )

    model_a = NeRFNetwork(
        N = len(dataloader_a.extrinsics),
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
    model_a.load_state_dict(torch.load(cfg.input_paths.model_path_a))

    model_coord_a = NeRFCoordinateWrapper(
        model=model_a,
        transform=None,
        inner_bound=cfg_a.scan.inner_bound,
        outer_bound=cfg_a.scan.outer_bound,
        translation_center=dataloader_a.translation_center
    )

    dataloader_b = CameraGeometryLoader(
        scan_paths=cfg_b.scan.scan_paths,
        scan_pose_paths=cfg_b.scan.scan_pose_paths,
        frame_ranges=cfg_b.scan.frame_ranges,
        frame_strides=cfg_b.scan.frame_strides,
        image_scale=cfg_b.scan.image_scale,
        load_images_bool=False,
        load_depths_bool=False,
        )

    model_b = NeRFNetwork(
        N = len(dataloader_b.extrinsics),
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
    model_b.load_state_dict(torch.load(cfg.input_paths.model_path_b))

    model_coord_b = NeRFCoordinateWrapper(
        model=model_b,
        transform=transform,
        inner_bound=cfg_b.scan.inner_bound,
        outer_bound=cfg_b.scan.outer_bound,
        translation_center=dataloader_b.translation_center
    )

    optimizer = torch.optim.Adam([
            {'name': 'translation', 'params': [transform.T], 'lr': 5e-4},
            {'name': 'rotation', 'params': [transform.R], 'lr': 5e-4},
            ], lr=1e-3, betas=(0.9, 0.99), eps=1e-15)
    num_iters = cfg.iters_per_epoch * cfg.num_epochs
    lmbda = lambda x: 0.01**(x/(num_iters))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    renderer_a = Render(
        models=model_coord_a,
        steps_firstpass=cfg.renderer.steps_firstpass,
        z_bounds=cfg.renderer.z_bounds,
        steps_importance=cfg.renderer.importance_steps,
        alpha_importance=cfg.renderer.alpha,
    )

    renderer_ab = Render(
        models=[model_coord_a, model_coord_b],
        steps_firstpass=cfg.renderer.steps_firstpass,
        z_bounds=cfg.renderer.z_bounds,
        steps_importance=cfg.renderer.importance_steps,
        alpha_importance=cfg.renderer.alpha,
    )

    trainer_align = TrainerAlign(
        logger=logger,
        transform=transform,
        dataloader_a=dataloader_a,
        renderer_a=renderer_a,
        renderer_ab=renderer_ab,
        optimizer=optimizer,
        scheduler=scheduler,
        iters_per_epoch=cfg.iters_per_epoch,
        num_epochs=cfg.num_epochs,
        n_rays=cfg.n_rays
        # starting_error=starting_error
    )

    trainer_align.train()


if __name__ == '__main__':
    train()