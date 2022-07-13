import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse

from omegaconf import OmegaConf
from tqdm import tqdm

import helpers

from loaders.camera_geometry_loader import CameraGeometryLoader



def homogeneous_to_cartesian(homogeneous):
    """
    Convert homogeneous coordinates to cartesian coordinates.
    """
    return homogeneous[:, :-1] / homogeneous[:, -1, None]


def homogeneous_divide(homogeneous):
    """
    Convert homogeneous coordinates to cartesian coordinates.
    """
    return homogeneous / homogeneous[:, -1, None]


def cartesian_to_homogeneous(cartesian):
    """
    Convert cartesian coordinates to homogeneous coordinates.
    """
    return torch.cat([cartesian, cartesian.new_ones(cartesian.shape[0], 1)], dim=1)


@torch.no_grad()
def n_view(dataloader, image_num_a, image_num_b):

    n, h, w, K, E, rgb_gt, color_bg, depth = dataloader.get_image_batch(image_num_a)
    n_b, h_b, w_b, K_b, E_b, rgb_gt_b, color_bg_b, depth_b = dataloader.get_image_batch(image_num_b)

    n_f = torch.reshape(n, (-1,))
    h_f = torch.reshape(h, (-1,))
    w_f = torch.reshape(w, (-1,))

    K_f = torch.reshape(K, (-1, 3, 3))
    E_f = torch.reshape(E, (-1, 4, 4))

    depth_f = torch.reshape(depth, (-1,))

    rays_o, rays_d = helpers.get_rays(h_f, w_f, K_f, E_f)

    xyzs = rays_o + rays_d * depth_f[:, None]
    xyzs_h = cartesian_to_homogeneous(xyzs)
    a = (torch.inverse(E_b[0, 0]) @ xyzs_h.T).T
    a1 = homogeneous_to_cartesian(a)
    b = homogeneous_divide(a1)
    c = homogeneous_to_cartesian((K_b[0, 0] @ b.T).T)

    output = torch.reshape(c, (*depth.shape, 2))

    return output



if __name__ == '__main__':

    dataloader = CameraGeometryLoader(
        scene_paths=['/home/casey/PhD/data/renders_close/vine_C1_1/front/cameras.json'],
        frame_ranges=[None],
        transforms=[None],
        image_scale=0.5,
        )

    image_num = len(dataloader.images)//6 * 3 + 3
    image_num2 = len(dataloader.images)//6 * 3 + 3 + 6

    n_views_count = torch.zeros(dataloader.images.shape[1:3])

    for i in tqdm(range(len(dataloader.images))):
        output = n_view(dataloader, image_num, i)
        
        binary = output.new_ones(output.shape[:2])
        binary[output[..., 0] <= 0] = 0
        binary[output[..., 0] >= output.shape[0]] = 0
        binary[output[..., 1] <= 0] = 0
        binary[output[..., 1] >= output.shape[1]] = 0

        n_views_count += binary

        # plt.imshow(binary)
        # plt.show()

    plt.imshow(n_views_count)
    plt.show()



    # # K = dataloader.intrinsics
    # # E = dataloader.extrinsics

    # image_num = len(dataloader.images)//6 * 3 + 3
    # n, h, w, K, E, rgb_gt, color_bg, depth = dataloader.get_image_batch(image_num)

    # n_b, h_b, w_b, K_b, E_b, rgb_gt_b, color_bg_b, depth_b = dataloader.get_image_batch(image_num+1)

    # # h = torch.arange(10, 20)
    # # w = torch.arange(10, 20)

    # output = depth.new_zeros(depth.shape)

    # for i in tqdm(range(depth.shape[0])):

    #     rays_o, rays_d = helpers.get_rays(h[i], w[i], K[i], E[i])

    #     # distance = 0.5

    #     # print(rays_d.shape, depth[i, None].shape)

    #     xyzs = rays_o + rays_d * depth[i, :, None]
    #     # print(xyzs.shape)

    #     xyzs = torch.cat((xyzs, xyzs.new_ones(xyzs.shape[0], 1)), dim=1)

    #     a = torch.inverse(E_b[0, 0]) @ xyzs.T
    #     a1 = a[:3, :]
    #     a2 = a1 / a1[2, :]
    #     b = K_b[0, 0] @ a2

    #     output[i, :] = b[0, :]

    # plt.imshow(output)
    # plt.show()