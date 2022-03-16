import numpy as np
import torch
import cv2
import commentjson as json
import tinycudann as tcnn
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader
from loaders.synthetic import load_image_set

# from nets import NerfHash, NerfRenderOccupancy, NerfRender
from nets2 import NerfHash, NeRFNetwork
from trainer import Trainer



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


if __name__ == '__main__':

    ## Params
    n_rays = 4096
    bound = 2
    device = 'cuda'

    # model = NerfHash().to(device)
    model = NeRFNetwork().to(device)

    images, depths, intrinsics, extrinsics, bds = meta_loader('synthetic')
    # print(torch.max(extrinsics[:, :3, 3]), torch.min(extrinsics[:, :3, 3]))
    # exit()
    # extrinsics[:, :3, 3] /= 16

    # optimizer = torch.optim.Adam([
    #         {'name': 'encoding', 'params': list(model.encoder_hash.parameters())},
    #         {'name': 'net', 'params': list(model.network_sigma.parameters()) + list(model.network_rgb.parameters()), 'weight_decay': 1e-6},
    #     ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)
    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    criterion = torch.nn.HuberLoss(delta=0.1)

    trainer = Trainer(model, images, depths, intrinsics, extrinsics, optimizer, criterion, n_rays, bound, device)

    trainer.train()