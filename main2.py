from logging import root
import numpy as np
import torch
import cv2
import commentjson as json
import yaml
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from box import Box

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader
from loaders.synthetic import load_image_set

from nets2 import NeRFNetwork
from trainer2 import Trainer
from logger import Logger
from inference import Inference

from misc import extract_foreground, remove_background



def meta_loader(loader):
    if loader == 'synthetic':
        # scene_path = '/home/casey/Datasets/NeRF_Data/nerf_synthetic/lego'
        scene_path = '/local/NeRF/torch-ngp/data/nerf_synthetic/lego'
        images, depths, intrinsics, extrinsics, bds = load_image_set(scene_path, near=2, far=6, scale=0.04)
    elif loader == 'camera_geometry':
        scene_dir = '/mnt/maara/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'
        images, depths, intrinsics, extrinsics, bds = camera_geometry_loader(scene_dir, image_scale=1, frame_range=(0, 2))

    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)
    bds = torch.Tensor(bds)

    return images, depths, intrinsics, extrinsics, bds

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

    return mask_full.cpu().numpy()


def meta_camera_geometry():
    # scene_dir = '/local/v100/mnt/maara/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'
    scene_dir = '/home/casey/Documents/PhD/data/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'

    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader(scene_dir, image_scale=0.5)
    images_, depths_, _, _, ids_ = camera_geometry_loader(scene_dir, image_scale=0.125)

    images_nb, depths_nb = remove_background(images_, depths_, ids_, threshold=1)

    xyz_min, xyz_max = helpers.calculate_bounds(images_nb, depths_nb, intrinsics, extrinsics)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - (xyz_max + xyz_min) / 2

    xyz_min_norm, xyz_max_norm = helpers.calculate_bounds_sphere(images_nb, depths_nb, intrinsics, extrinsics)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / xyz_max_norm
    depths = depths / xyz_max_norm

    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)

    return images, depths, intrinsics, extrinsics


def vine_extraction_test():

    scene_file = '/home/casey/Documents/PhD/data/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'

    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader(scene_file, image_scale=0.5)
    # with open('/home/casey/Documents/PhD/data/synthetic_tree_assets/trees3/scenes/vine_C2_1/scene.json', 'r') as f:
    #     scene_ids = json.load(f)

    # images_, depths_ = extract_foreground(images, depths, ids, scene_ids)

    ids[ids > 256] = 0

    plt.imshow(ids[3])
    plt.show()

    exit()


if __name__ == '__main__':

    with open('./configs/config.yaml', 'r') as f:
        cfg = Box(yaml.safe_load(f))

    
    logger = Logger(
        root_dir=cfg.log.root_dir,
        )

    logger.log('Loading Data...')

    images, depths, intrinsics, extrinsics = meta_camera_geometry()

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

    inference = Inference(
        model=model,
        mask=mask,
        n_rays=cfg.inference.n_rays,
        voxel_res=cfg.inference.voxel_res,
        thresh=cfg.inference.thresh,
        batch_size=cfg.inference.batch_size,
        )

    # images, depths, intrinsics, extrinsics, bds = meta_loader('synthetic')
    # plt.imshow(images[0, ..., -1])
    # # plt.imshow(depths[0, ...])
    # plt.show()
    # exit()

    # N, H, W, C = images.shape
    # ids = []
    # for i in range(17):
    #     for j in range(6):
    #         if j > 2 and j < 5 and i > 4 and i < 13:
    #             ids.append(i*6+j)
    # torch.Tensor(ids).to(int)
    # mask = get_valid_positions(H, W, intrinsics[ids, ...].to('cuda'), extrinsics[ids, ...].to('cuda'))
    # np.save('./data/valid_positions.npy', mask)
    # print(mask.shape)
    # mask = torch.Tensor(np.load('./data/valid_positions.npy')).to(bool)

    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    trainer = Trainer(
        model=model,
        images=images,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        optimizer=optimizer,

        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,
        eval_freq=cfg.trainer.eval_freq,
        )

    trainer.train()