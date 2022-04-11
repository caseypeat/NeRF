import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

import helpers

from tqdm import tqdm

from config import cfg


class Inference(object):
    def __init__(
        self,
        model,
        mask,
        # image_scale,
        n_rays,
        voxel_res,
        thresh,
        batch_size):
        
        self.model = model
        self.mask = mask
        # self.image_scale = image_scale
        self.n_rays = n_rays
        self.voxel_res = voxel_res
        self.thresh = thresh
        self.batch_size = batch_size
    
    @torch.no_grad()
    def render_image(self, H, W, K, E):

        K = K.to('cuda')
        E = E.to('cuda')

        h = torch.arange(0, H, device='cuda')
        w = torch.arange(0, W, device='cuda')

        h, w = torch.meshgrid(h, w, indexing='ij')

        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        image_f = torch.zeros((*h_f.shape, 3))
        invdepth_f = torch.zeros(h_f.shape)

        for a in tqdm(range(0, len(h_f), self.n_rays)):
            b = min(len(h_f), a+self.n_rays)

            h_fb = h_f[a:b]
            w_fb = w_f[a:b]

            rays_o, rays_d = helpers.get_rays(h_fb, w_fb, K[None, ...], E[None, ...])

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            image_fb, invdepth_fb, _, _ = self.model.render(rays_o, rays_d, bg_color=color_bg)

            image_f[a:b] = image_fb
            invdepth_f[a:b] = invdepth_fb

        image = torch.reshape(image_f, (*h.shape, 3))
        invdepth = torch.reshape(invdepth_f, h.shape)

        return image, invdepth

    @torch.no_grad()
    def extract_geometry(self):

        scale = self.voxel_res / self.mask.shape[0]

        voxels = torch.linspace(-1+1/self.voxel_res, 1-1/self.voxel_res, self.voxel_res, device='cuda')

        num_samples = self.voxel_res**3

        points = torch.zeros((0, 4), device='cuda')

        for a in tqdm(range(0, num_samples, self.batch_size)):
            b = min(num_samples, a+self.batch_size)

            n = torch.arange(a, b)

            x = voxels[torch.div(n, self.voxel_res**2, rounding_mode='floor')]
            y = voxels[torch.div(n, self.voxel_res, rounding_mode='floor') % self.voxel_res]
            z = voxels[n % self.voxel_res]

            xyz = torch.stack((x, y, z), dim=-1)

            x_i = ((x+1)/2*self.mask.shape[0]).to(int)
            y_i = ((y+1)/2*self.mask.shape[1]).to(int)
            z_i = ((z+1)/2*self.mask.shape[2]).to(int)

            xyz = xyz[self.mask[x_i, y_i, z_i]].view(-1, 3)

            if xyz.shape[0] > 0:
                sigmas = self.model.density(xyz).to(torch.float32)
                new_points = torch.cat((xyz[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3), sigmas[sigmas > self.thresh][..., None]), dim=-1)
                points = torch.cat((points, new_points))

        return points
