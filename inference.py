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
    def render_image(self):
        pass

    def extract_geometry_old(self, bound, mask):
        with torch.no_grad():
            scale = 4
            res = 512 * scale
            thresh = 10

            points = torch.zeros((0, 4), device='cuda')

            for i in tqdm(range(res)):
                d = torch.linspace(-1, 1, res, device='cuda')
                D = torch.stack(torch.meshgrid(d[i], d, d), dim=-1)

                mask_ = mask[i//scale, :, :, None].expand(-1, -1, 3)[None, ...].to('cuda')
                # mask_ = F.interpolate(mask_.permute(0, 3, 1, 2).to(float), (res, res), mode='nearest-exact').to(bool).permute(0, 2, 3, 1)
                mask_ = F.interpolate(mask_.permute(0, 3, 1, 2).to(float), (res, res), mode='nearest').to(bool).permute(0, 2, 3, 1)

                xyzs = D[mask_].view(-1, 3)

                if xyzs.shape[0] > 0:
                    sigmas = self.model.density(xyzs, bound).to(torch.float32)
                    new_points = torch.cat((xyzs[sigmas[..., None].expand(-1, 3) > thresh].view(-1, 3), sigmas[sigmas > thresh][..., None]), dim=-1)
                    points = torch.cat((points, new_points))

            print(points.shape)
            np.save('./data/points.npy', points.cpu().numpy())

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
