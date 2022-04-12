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

import cProfile, pstats, io
from pstats import SortKey

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader, camera_geometry_loader_real
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from trainer import Trainer
from logger import Logger
from inference import Inference

from misc import extract_foreground, remove_background, remove_background2

from config import cfg


@torch.no_grad()
def get_valid_positions(N, H, W, K, E, res):

    mask_full = torch.zeros((res, res, res), dtype=bool, device='cuda')

    for i in tqdm(range(res)):
        d = torch.linspace(-1, 1, res, device='cuda')
        D = torch.stack(torch.meshgrid(d[i], d, d, indexing='ij'), dim=-1)
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


def meta_camera_geometry_real(scene_path):

    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader_real(scene_path, image_scale=0.5)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - np.mean(extrinsics[..., :3, 3], axis=0, keepdims=True)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / 2

    images = torch.ByteTensor(images)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)

    return images, None, intrinsics, extrinsics


if __name__ == '__main__':

    images, depths, intrinsics, extrinsics = meta_camera_geometry_real(cfg.scene.scene_path)


    # model = NeRFNetwork(
    #     # Render args
    #     bound = cfg.scene.bound,

    #     inner_near=cfg.renderer.inner_near,
    #     inner_far=cfg.renderer.inner_far,
    #     inner_steps=cfg.renderer.inner_steps,
    #     # outer_near=cfg.renderer.outer_near,
    #     # outer_far=cfg.renderer.outer_far,
    #     # outer_steps=cfg.renderer.outer_steps,

    #     # Net args
    #     n_levels=cfg.nets.encoding.n_levels,
    #     n_features_per_level=cfg.nets.encoding.n_features,
    #     log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
    #     encoding_precision=cfg.nets.encoding.precision,

    #     encoding_dir=cfg.nets.encoding_dir.encoding,
    #     encoding_dir_degree=cfg.nets.encoding_dir.degree,
    #     encoding_dir_precision=cfg.nets.encoding_dir.precision,

    #     num_layers=cfg.nets.sigma.num_layers,
    #     hidden_dim=cfg.nets.sigma.hidden_dim,
    #     geo_feat_dim=cfg.nets.sigma.geo_feat_dim,

    #     num_layers_color=cfg.nets.color.num_layers,
    #     hidden_dim_color=cfg.nets.color.hidden_dim,

    #     N = images.shape[0]
    # ).to('cuda')

    # model.load_state_dict(torch.load('./logs/real_scenes/20220412_210427/model/20000.pth'))

    model = torch.load('./logs/real_scenes/20220412_210427/model/20000.pth')

    N, H, W = images.shape[:3]
    # mask = get_valid_positions(N, H, W, intrinsics.to('cuda'), extrinsics.to('cuda'), res=256)
    mask = torch.zeros([256]*3)

    inference = Inference(
        model=model,
        mask=mask,
        n_rays=cfg.inference.n_rays,
        voxel_res=cfg.inference.voxel_res,
        thresh=cfg.inference.thresh,
        batch_size=cfg.inference.batch_size,
        )

    ids = []
    for i in range(N):
        if i > 300 and i < 900:
            if i % 6 == 2 or i % 6 == 3:
                if i//6 % 10 == 0:
                    ids.append(i)
    print(len(ids))
    ids = torch.Tensor(np.array(ids)).to(int)

    inference.extract_geometry_rays(H, W, intrinsics[ids], extrinsics[ids])