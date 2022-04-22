import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math as m

import tinycudann as tcnn

import helpers

from tqdm import tqdm

from config import cfg



class NerfRendererPlanes(nn.Module):
    def __init__(self):
        super().__init__()

        self.z1 = torch.Tensor((1/(0.3-0.05), 0, 0)).to('cuda')
        self.z2 = torch.Tensor((1/(0.3-1), 0, 0)).to('cuda')

        self.encoder1 = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 18,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=torch.float32
        )

        self.encoder2 = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 18,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=torch.float32
        )

        self.geo_feat_dim = 15
        self.sigma_net = tcnn.Network(
            n_input_dims=16*2*2,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.in_dim_color = self.geo_feat_dim
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )


    def forward(self, xy1, xy2):

        xy1 = (xy1 + 1) / 2
        xy2 = (xy2 + 1) / 2

        xy1_e = self.encoder1(xy1)
        xy2_e = self.encoder2(xy2)
        xy_e = torch.cat((xy1_e, xy2_e), dim=-1)

        h = self.sigma_net(xy_e)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        h = self.color_net(geo_feat)

        color = torch.sigmoid(h)

        return sigma, color


    def density(self, xy1, xy2):
        
        xy1 = (xy1 + 2) / 4
        xy2 = (xy2 + 2) / 4

        xy1_e = self.encoder1(xy1)
        xy2_e = self.encoder2(xy2)
        xy_e = torch.cat((xy1_e, xy2_e), dim=-1)

        h = self.sigma_net(xy_e)

        sigma = F.relu(h[..., 0])

        return sigma


    def calculate_intersection(self, o, d, p):

        d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
        
        # t = (1 - (p[0]*o[..., 0] + p[1]*o[..., 1] + p[2]*o[..., 2])) / (p[0]*d[..., 0] + p[1]*d[..., 1] + p[2]*d[..., 2])

        # x = o[..., 0] * d[..., 0] * t
        # y = o[..., 1] * d[..., 1] * t
        # z = o[..., 2] * d[..., 2] * t

        # t = (1 - p[0]*o[..., 0]) / (p[0]*d[..., 0])
        t = (1 / p[0] - o[..., 0]) / (d[..., 0])

        x = o[..., 0] + d[..., 0] * t
        y = o[..., 1] + d[..., 1] * t
        z = o[..., 2] + d[..., 2] * t

        # print(torch.amin(y), torch.amax(y))
        # print(torch.amin(z), torch.amax(z))

        xy = torch.stack([y, z], dim=-1)

        return xy


    def render(self, rays_o, rays_d, bg_color):

        xy1 = self.calculate_intersection(rays_o, rays_d, self.z1)
        xy2 = self.calculate_intersection(rays_o, rays_d, self.z2)

        sigma, rgb = self(xy1, xy2)

        color = rgb * sigma[..., None]
        color += (1 - sigma[..., None]) * bg_color

        return color




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



    def render(self, rays_o, rays_d, n, bg_color):

        # z_vals_log_inner = self.efficient_sampling(rays_o, rays_d, n_samples=256)
        z_vals_log_inner = torch.linspace(m.log10(self.inner_near), m.log10(self.inner_far)-(m.log10(self.inner_far)-m.log10(self.inner_near))/self.inner_steps, self.inner_steps, device=rays_o.device).expand(rays_o.shape[0], -1)
        z_vals_log = torch.cat([z_vals_log_inner, torch.linspace(m.log10(self.outer_near), m.log10(self.outer_far)-(m.log10(self.outer_far)-m.log10(self.outer_near))/self.outer_steps, self.outer_steps, device=rays_o.device).expand(rays_o.shape[0], -1)], dim=-1)
        z_vals = torch.pow(10, z_vals_log)

        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, self.bound)

        n_expand = n[:, None].expand(-1, z_vals.shape[-1])

        sigmas, rgbs = self(s_xyzs, dirs, n_expand)

        image, invdepth, weights = helpers.render_rays_log(sigmas, rgbs, z_vals, z_vals_log)

        image = image + (1 - torch.sum(weights, dim=-1)[..., None]) * bg_color

        z_vals_log_s = (z_vals_log - m.log10(self.inner_near)) / (m.log10(self.outer_far) - m.log10(self.inner_near))

        return image, invdepth, weights, z_vals_log_s


def log_base(x, base):
    return torch.log(x) / m.log(base)


class NerfRendererNB(nn.Module):
    def __init__(self,
                bound=1.125,
                # z_vals
                inner_near=0.05,
                inner_far=1,
                inner_steps=384,
                ):
        super().__init__()

        self.bound = bound

        self.inner_near = inner_near
        self.inner_far = inner_far
        self.inner_steps = inner_steps


    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()


    def render(self, rays_o, rays_d, bg_color):

        # z_vals_log_inner = self.efficient_sampling(rays_o, rays_d, n_samples=256)
        z_vals_log = torch.linspace(m.log10(self.inner_near), m.log10(self.inner_far)-(m.log10(self.inner_far)-m.log10(self.inner_near))/self.inner_steps, self.inner_steps, device=rays_o.device).expand(rays_o.shape[0], -1)
        z_vals = torch.pow(10, z_vals_log)

        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, self.bound)

        n_levels = 16
        n_features_per_level = 2

        b = 1.3819
        Vs = 1 / (n_levels*torch.pow(b, torch.arange(n_levels, device='cuda')))
        f = 1109
        Ps = z_vals / f

        # mip = 1 / (1 + torch.exp(10*(log_base(Ps, b)[..., None] - log_base(Vs, b))))
        # mip = torch.repeat_interleave(mip, n_features_per_level, dim=-1)
        # mip = mip.reshape(-1, n_levels*n_features_per_level)

        mip = torch.ones((*z_vals.shape, n_levels), device='cuda')
        # mip[..., 8:] = 0
        mip = torch.repeat_interleave(mip, n_features_per_level, dim=-1)
        mip = mip.reshape(-1, n_levels*n_features_per_level)


        sigmas, rgbs = self(s_xyzs, dirs, mip)

        image, invdepth, weights = helpers.render_rays_log(sigmas, rgbs, z_vals, z_vals_log)

        image = image + (1 - torch.sum(weights, dim=-1)[..., None]) * bg_color

        z_vals_log_s = (z_vals_log - m.log10(self.inner_near)) / (m.log10(self.inner_far) - m.log10(self.inner_near))

        return image, invdepth, weights, z_vals_log_s