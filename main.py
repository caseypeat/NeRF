import numpy as np
import torch
import cv2
import commentjson as json
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
from nets import NerfHash, NeRFNetwork
from trainer import Trainer

from misc import remove_background



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


def get_valid_positions(H, W, K, E):
    with torch.no_grad():
        res = 1024

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
            mask_dirs = torch.zeros((E.shape[0], res, res), dtype=int, device='cuda')
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
    # images, depths, intrinsics, extrinsics = camera_geometry_loader(scene_dir, image_scale=0.25, frame_range=(0, 2))
    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader(scene_dir, image_scale=0.5)

    images_nb, depths_nb = remove_background(images, depths, ids, threshold=1)

    # for i in images_nb:
    #     plt.imshow(i)
    #     plt.show()

    # xyz_min, xyz_max = helpers.calculate_bounds(images_nb, depths_nb, intrinsics, extrinsics)

    # e_min = np.amin(extrinsics[..., :3, 3], axis=tuple(np.arange(len(extrinsics[..., :3, 3].shape[:-1]))))
    # e_max = np.amax(extrinsics[..., :3, 3], axis=tuple(np.arange(len(extrinsics[..., :3, 3].shape[:-1]))))

    # print(e_min, e_max)

    # t_min = np.minimum(e_min, xyz_min)
    # t_max = np.maximum(e_max, xyz_max)
    # print(t_min, t_max)

    # extrinsics[..., :3, 3] = ((extrinsics[..., :3, 3] - np.amin(xyz_min)) / (np.amax(xyz_max) - np.amin(xyz_min)) * 2) - 1
    # extrinsics[..., :3, 3] = (extrinsics[..., :3, 3] - np.amin(bds_min))
    # depths = depths / (np.amax(xyz_max) - np.amin(xyz_min)) * 2

    # xyz_min, xyz_max = np.array([-0.4991, -1.1396, -0.4194]), np.array([0.2367, 1.3705, 1.2255])
    # xyz_min_norm, xyz_max_norm = 0, 1.53529

    xyz_min, xyz_max = np.array([0.2985, -0.8025,  0.3925]), np.array([0.4028, 0.8005, 0.8941])
    xyz_min_norm, xyz_max_norm = 0.0, 1.2362564645858782

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - (xyz_max + xyz_min) / 2

    # xyz_min_norm, xyz_max_norm = helpers.calculate_bounds_sphere(images_nb, depths_nb, intrinsics, extrinsics)

    # print(xyz_min, xyz_max)
    # print(xyz_min_norm, xyz_max_norm)

    # exit()

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / xyz_max_norm
    depths = depths / xyz_max_norm

    # xyz_min, xyz_max = helpers.calculate_bounds(images, depths, intrinsics, extrinsics)
    # xyz_min_norm, xyz_max_norm = helpers.calculate_bounds_sphere(images, depths, intrinsics, extrinsics)

    # print(xyz_min, xyz_max)
    # print(xyz_min_norm, xyz_max_norm)
    # exit()

    # exit()

    # print(np.amax(extrinsics[:,:3,3]), np.amin(extrinsics[:,:3,3]))

    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)
    # bds = torch.Tensor(bds)

    return images, depths, intrinsics, extrinsics


if __name__ == '__main__':

    # print('test...')

    ## Params
    # n_rays = 4096
    # n_rays = 3072
    n_rays = 2048
    # n_rays = 1536
    # n_rays = 1024
    # n_rays = 512
    bound = 1.125
    # bound = 3
    device = 'cuda'

    model = NeRFNetwork().to(device)

    # images, depths, intrinsics, extrinsics, bds = meta_loader('synthetic')
    images, depths, intrinsics, extrinsics = meta_camera_geometry()
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
    # exit()
    mask = torch.Tensor(np.load('./data/valid_positions.npy')).to(bool)

    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters())},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': 1e-6},
        ], lr=1e-2, betas=(0.9, 0.99), eps=1e-15)

    criterion = torch.nn.HuberLoss(delta=0.1)

    trainer = Trainer(model, images, depths, intrinsics, extrinsics, optimizer, criterion, n_rays, bound, device, mask)

    trainer.train()