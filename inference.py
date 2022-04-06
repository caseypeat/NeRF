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


class Inference(object):
    def __init__(self):
        pass
    
    @torch.no_grad()
    def render_image(self, H, W, K, E):
        scale = 0.5

        n_rays = 1024

        K = K.to('cuda')
        E = E.to('cuda')

        h = torch.arange(0, H, device='cuda')
        w = torch.arange(0, W, device='cuda')

        h, w = torch.meshgrid(h, w)

        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        image_f = torch.zeros((*h_f.shape, 3))
        invdepth_f = torch.zeros(h_f.shape)

        for a in tqdm(range(0, len(h_f), n_rays)):
            b = min(len(h_f), a+n_rays)

            h_fb = h_f[a:b]
            w_fb = w_f[a:b]

            rays_o, rays_d = helpers.get_rays(h_fb, w_fb, K[None, ...], E[None, ...])

            color_bg = torch.ones(3, device=self.device) # [3], fixed white background

            image_fb, invdepth_fb, _, _ = self.model.render(rays_o, rays_d, bg_color=color_bg)

            image_f[a:b] = image_fb
            invdepth_f[a:b] = invdepth_fb

        image = torch.reshape(image, (*h.shape, 3))
        invdepth = torch.reshape(invdepth, h.shape)

        return image, invdepth

    @torch.no_grad()
    def extract_geometry(self, mask):
        res = 1024
        thresh = 10

        scale = res / mask.shape[0]

        voxels = torch.linspace(-1, 1, res, device='cuda')

        batch_size = 1048576
        num_samples = res**3

        points = torch.zeros((0, 4), device='cuda')

        for a in tqdm(range(0, num_samples, batch_size)):
            b = min(num_samples, a+batch_size)

            n = torch.arange(a, b)

            x = voxels[n//res**2]
            y = voxels[(n//res) % res]
            z = voxels[n % res]

            xyz = torch.stack((x, y, z), dim=-1)

            xyz = xyz[mask[(x/scale).to(int), (y/scale).to(int), (z/scale).to(int)]].view(-1, 3)

            if xyz.shape[0] > 0:
                sigmas = self.model.density(xyz).to(torch.float32)
                new_points = torch.cat((xyz[sigmas[..., None].expand(-1, 3) > thresh].view(-1, 3), sigmas[sigmas > thresh][..., None]), dim=-1)
                points = torch.cat((points, new_points))

        return points
