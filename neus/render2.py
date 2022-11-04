import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m
import matplotlib.pyplot as plt

from typing import Union, Optional
from dataclasses import dataclass

# from nets import NeRFNetwork, NeRFCoordinateWrapper
from render import get_rays, get_uniform_z_vals, efficient_sampling, get_sample_points, render_rays_log, regularize_weights, sample_pdf

from neus.nets import SDFNetwork, RenderingNetwork, SingleVarianceNetwork


class NeuSRenderer:
    def __init__(self):

        self.sdf_network = SDFNetwork().to("cuda")
        self.color_network = RenderingNetwork().to("cuda")
        self.deviation_network = SingleVarianceNetwork(0.6).to("cuda")

        self.steps_firstpass = [256]
        # self.z_bounds = [2, 6]
        self.z_bounds = [0.1, 2]

    
    @torch.no_grad()
    def get_dry_sdf(self, n, h, w, K, E):
        rays_o, rays_d = get_rays(h, w, K, E)
        n_rays = rays_o.shape[0]

        z_vals_log, z_vals = get_uniform_z_vals(self.steps_firstpass, self.z_bounds, n_rays)
        dists = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], z_vals.new_zeros(z_vals.shape[0], 1)], dim=-1)
        batch_size, n_samples = z_vals.shape

        xyzs, dirs = get_sample_points(rays_o, rays_d, z_vals)

        xyzs = xyzs.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = self.sdf_network(xyzs)
        sdf = sdf_nn_output[:, :1]

        self.bias = 0.03 - torch.mean(sdf)

        return sdf

    def render(self, n, h, w, K, E, cos_anneal_ratio, bg_color=None):
        rays_o, rays_d = get_rays(h, w, K, E)
        n_rays = rays_o.shape[0]

        z_vals_log, z_vals = get_uniform_z_vals(self.steps_firstpass, self.z_bounds, n_rays)
        dists = torch.cat([z_vals[:, 1:] - z_vals[:, :-1], z_vals.new_zeros(z_vals.shape[0], 1)], dim=-1)
        batch_size, n_samples = z_vals.shape

        xyzs, dirs = get_sample_points(rays_o, rays_d, z_vals)

        xyzs = xyzs.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = self.sdf_network(xyzs)
        sdf = sdf_nn_output[:, :1] + self.bias
        feature_vector = sdf_nn_output[:, 1:]

        gradients = self.sdf_network.gradient(xyzs).squeeze()
        sampled_color = self.color_network(xyzs, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        weights = alpha * torch.cumprod(torch.cat([alpha.new_ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        invdepth = torch.sum(weights / z_vals, -1)  # [N_rays]

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if bg_color is not None:    # Fixed background, usually black
            color = color + bg_color * (1.0 - weights_sum)

        aux_outputs = {}
        aux_outputs['invdepth'] = invdepth.detach()
        aux_outputs['z_vals'] = z_vals.detach()
        aux_outputs['sdf'] = sdf.detach()

        return color, weights, gradients, aux_outputs