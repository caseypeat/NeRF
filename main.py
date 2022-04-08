from logging import root
import numpy as np
import torch
import cv2
import commentjson as json
import yaml
import os
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from box import Box

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from trainer import Trainer
from logger import Logger
from inference import Inference

from misc import extract_foreground, remove_background


@torch.no_grad()
def get_valid_positions(N, H, W, K, E, res):

    mask_full = torch.zeros((res, res, res), dtype=bool, device='cuda')

    for i in tqdm(range(res)):
        d = torch.linspace(-1, 1, res, device='cuda')
        D = torch.stack(torch.meshgrid(d[i], d, d), dim=-1)
        dist = torch.linalg.norm(D, dim=-1)[:, :, :, None].expand(-1, -1, -1, 3)
        mask = torch.zeros(dist.shape, dtype=bool, device='cuda')
        mask[dist < 1] = True

        # also mask out parts outside camera coverage
        rays_d = D - E[:, None, None, :3, -1]
        dirs_ = torch.inverse(E[:, None, None, :3, :3]) @ rays_d[..., None]
        dirs_ = K[:, None, None, ...] @ dirs_
        dirs = dirs_ / dirs_[:, :, :, 2, None, :]
        mask_dirs = torch.zeros((N, res, res), dtype=int, device='cuda')
        mask_dirs[((dirs[:, :, :, 0, 0] > 0) & (dirs[:, :, :, 0, 0] < H) & (dirs[:, :, :, 1, 0] > 0) & (dirs[:, :, :, 1, 0] < W) & (dirs_[:, :, :, 2, 0] > 0))] = 1
        mask_dirs = torch.sum(mask_dirs, dim=0)
        mask_dirs[mask_dirs > 0] = 1
        mask_dirs = mask_dirs.to(bool)
        mask_dirs = mask_dirs[None, :, :, None].expand(-1, -1, -1, 3)
        mask = torch.logical_and(mask, mask_dirs)

        mask_full[i, :, :] = mask[..., 0]

    return mask_full


def meta_camera_geometry(scene_path, remove_background_bool):

    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader(scene_path, image_scale=0.5)
    if remove_background_bool:
        images, depths, ids = remove_background(images, depths, ids, threshold=1)

    images_ds = images[:, ::4, ::4, :]
    depths_ds = depths[:, ::4, ::4]
    ids_ds = ids[:, ::4, ::4]
    images_ds_nb, depths_ds_nb, ids_ds_nb = remove_background(images_ds, depths_ds, ids_ds, threshold=1)

    xyz_min, xyz_max = helpers.calculate_bounds(images_ds_nb, depths_ds_nb, intrinsics, extrinsics)
    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - (xyz_max + xyz_min) / 2
    xyz_min_norm, xyz_max_norm = helpers.calculate_bounds_sphere(images_ds_nb, depths_ds_nb, intrinsics, extrinsics)
    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / xyz_max_norm
    depths = depths / xyz_max_norm

    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)

    return images, depths, intrinsics, extrinsics


if __name__ == '__main__':

    with open('./configs/config.yaml', 'r') as f:
        cfg = Box(yaml.safe_load(f))

    logger = Logger(
        root_dir=cfg.log.root_dir,
        )

    logger.log('Loading Data...')
    images, depths, intrinsics, extrinsics = meta_camera_geometry(cfg.scene.scene_path, cfg.scene.remove_background_bool)
    # images, depths, intrinsics, extrinsics = meta_camera_geometry()


    logger.log('Initilising Model...')
    model = NeRFNetwork(
        # Render args
        bound = cfg.scene.bound,

        inner_near=cfg.renderer.inner_near,
        inner_far=cfg.renderer.inner_far,
        inner_steps=cfg.renderer.inner_steps,
        outer_near=cfg.renderer.outer_near,
        outer_far=cfg.renderer.outer_far,
        outer_steps=cfg.renderer.outer_steps,

        # Net args
        n_levels=cfg.nets.encoding.n_levels,
        n_features_per_level=cfg.nets.encoding.n_features,
        log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
        encoding_precision=cfg.nets.encoding.precision,

        encoding_dir=cfg.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg.nets.encoding_dir.degree,
        encoding_dir_precision=cfg.nets.encoding_dir.precision,

        num_layers=cfg.nets.sigma.num_layers,
        hidden_dim=cfg.nets.sigma.hidden_dim,
        geo_feat_dim=cfg.nets.sigma.geo_feat_dim,

        num_layers_color=cfg.nets.color.num_layers,
        hidden_dim_color=cfg.nets.color.hidden_dim,
    ).to('cuda')

    logger.log('Generating Mask...')
    N, H, W = images.shape[:3]
    mask = get_valid_positions(N, H, W, intrinsics.to('cuda'), extrinsics.to('cuda'), res=256)

    logger.log('Initiating Inference...')
    inference = Inference(
        model=model,
        mask=mask,
        n_rays=cfg.inference.n_rays,
        voxel_res=cfg.inference.voxel_res,
        thresh=cfg.inference.thresh,
        batch_size=cfg.inference.batch_size,
        )

    logger.log('Initiating Optimiser...')
    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    # optimizer = torch.optim.Adam([
    #         {'name': 'encoding', 'params': list(model.encoder.parameters())},
    #         {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
    #     ], lr=1e-3, betas=(0.9, 0.99), eps=1e-15)

    logger.log('Initiating Trainer...')
    trainer = Trainer(
        model=model,
        logger=logger,
        inference=inference,
        images=images,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        optimizer=optimizer,

        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,
        eval_freq=cfg.trainer.eval_freq,
        eval_image_num=cfg.inference.image_num,
        )

    logger.log('Beginning Training...\n')
    trainer.train()