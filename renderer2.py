import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math as m

import helpers

from tqdm import tqdm

class NerfRenderer(nn.Module):
    def __init__(self,
                # z_vals
                inner_near=0.05,
                inner_far=1,
                # inner_steps=384,
                inner_steps=256,
                outer_near=1,
                outer_far=100,
                # outer_steps=192,
                outer_steps=128,
                ):
        super().__init__()

        self.inner_near = inner_near
        self.inner_far = inner_far
        self.inner_steps = inner_steps

        self.outer_near = outer_near
        self.outer_far = outer_far
        self.outer_steps = outer_steps

    def forward(self, x, d, bound):
        raise NotImplementedError()

    def density(self, x, d, bound):
        raise NotImplementedError()


    def render(self, rays_o, rays_d, bound, bg_color):

        z_vals_log = torch.cat([torch.linspace(m.log10(self.inner_near), m.log10(self.inner_far)-(m.log10(self.inner_far)-m.log10(self.inner_near))/self.inner_steps, self.inner_steps, device=rays_o.device), torch.linspace(m.log10(self.outer_near), m.log10(self.outer_far)-(m.log10(self.outer_far)-m.log10(self.outer_near))/self.outer_steps, self.outer_steps, device=rays_o.device)], dim=0)[None, ...].expand(rays_o.shape[0], -1)
        z_vals = torch.pow(10, z_vals_log)

        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, bound)

        sigmas, rgbs = self(s_xyzs, dirs, bound)
        sigmas, rgbs = sigmas.to(torch.float32), rgbs.to(torch.float32)

        image, invdepth, weights = helpers.render_rays_log_new(sigmas, rgbs, z_vals, z_vals_log)

        return image, invdepth, weights, z_vals_log