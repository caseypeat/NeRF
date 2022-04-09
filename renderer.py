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
                bound=1.125,
                # z_vals
                inner_near=0.05,
                inner_far=1,
                inner_steps=384,
                outer_near=1,
                outer_far=100,
                outer_steps=192,
                ):
        super().__init__()

        self.bound = bound

        self.inner_near = inner_near
        self.inner_far = inner_far
        self.inner_steps = inner_steps

        self.outer_near = outer_near
        self.outer_far = outer_far
        self.outer_steps = outer_steps

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()


    @torch.no_grad()
    def efficient_sampling(self, rays_o, rays_d, n_samples):
        z_vals_log = torch.linspace(m.log10(self.inner_near), m.log10(self.inner_far)-(m.log10(self.inner_far)-m.log10(self.inner_near))/n_samples, n_samples, device=rays_o.device)[None, ...].expand(rays_o.shape[0], -1)
        z_vals = torch.pow(10, z_vals_log)

        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, self.bound)

        sigmas = self.density(s_xyzs)

        delta = z_vals_log.new_zeros(sigmas.shape)  # [N_rays, N_samples]
        delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

        alpha = 1 - torch.exp(-sigmas * delta)  # [N_rays, N_samples]

        alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

        mask = weights.new_zeros(sigmas.shape, dtype=bool)
        mask[:, ::n_samples//(self.inner_steps//2)] = True

        _, indices = torch.sort(weights[~mask].view(rays_o.shape[0], -1), dim=-1, descending=True)

        mask[~mask] = (indices < self.inner_steps//2).view(-1)

        z_vals_log_inner = z_vals_log[mask].view(rays_o.shape[0], self.inner_steps)

        return z_vals_log_inner



    def render(self, rays_o, rays_d, bg_color):

        # z_vals_log_inner = self.efficient_sampling(rays_o, rays_d, n_samples=256)
        z_vals_log_inner = torch.linspace(m.log10(self.inner_near), m.log10(self.inner_far)-(m.log10(self.inner_far)-m.log10(self.inner_near))/self.inner_steps, self.inner_steps, device=rays_o.device).expand(rays_o.shape[0], -1)
        z_vals_log = torch.cat([z_vals_log_inner, torch.linspace(m.log10(self.outer_near), m.log10(self.outer_far)-(m.log10(self.outer_far)-m.log10(self.outer_near))/self.outer_steps, self.outer_steps, device=rays_o.device).expand(rays_o.shape[0], -1)], dim=-1)
        z_vals = torch.pow(10, z_vals_log)

        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, self.bound)

        sigmas, rgbs = self(s_xyzs, dirs)

        image, invdepth, weights = helpers.render_rays_log(sigmas, rgbs, z_vals, z_vals_log)

        image = image + (1 - torch.sum(weights, dim=-1)[..., None]) * bg_color

        z_vals_log_s = (z_vals_log - m.log10(self.inner_near)) / (m.log10(self.outer_far) - m.log10(self.inner_near))

        return image, invdepth, weights, z_vals_log_s