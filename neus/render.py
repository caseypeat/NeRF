import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m
import matplotlib.pyplot as plt

from typing import Union, Optional
from dataclasses import dataclass

from nets import NeRFNetwork, NeRFCoordinateWrapper
from render import get_rays, get_uniform_z_vals, efficient_sampling, get_sample_points, render_rays_log, regularize_weights, sample_pdf

from sdf.model.density import LaplaceDensity
from sdf.model.ray_sampler import ErrorBoundSampler


def neus_render_rays_log(sdf, color, z_vals, z_vals_logs, s):
    N_rays, N_samples = z_vals.shape[:2]

    sdf_scaled = sdf * s

    alpha = (torch.sigmoid(sdf_scaled[..., :-1]) - torch.sigmoid(sdf_scaled[..., 1:])) / torch.sigmoid(sdf_scaled[..., :-1])

    transmittance = torch.cumprod(1 - alpha, dim=-1)
    transmittance = torch.cat((transmittance.new_zeros((transmittance.shape[0], 1)), transmittance), dim=-1)

    alpha = torch.cat((alpha, alpha.new_zeros((alpha.shape[0], 1))), dim=-1).clip(0.0, 1.0)

    weight = transmittance * alpha

    # print("weight: ", torch.amax(torch.sum(weight, dim=-1)), torch.amin(torch.sum(weight, dim=-1)))
    # print("alpha: ", torch.amax(alpha), torch.amin(alpha))

    pixel = torch.sum(weight[..., None] * color, -2)  # [N_rays, 3]
    invdepth = torch.sum(weight / z_vals, -1)  # [N_rays]

    return pixel, invdepth, weight


@torch.no_grad()
def neus_efficient_sampling(
    model,
    rays_o,
    rays_d,
    steps_firstpass,
    z_bounds,
    steps_importance,
    alpha_importance,
    s):
    n_rays = rays_o.shape[0]
    z_vals_log, z_vals = get_uniform_z_vals(steps_firstpass, z_bounds, n_rays)

    xyzs, _ = get_sample_points(rays_o, rays_d, z_vals)

    sdf = model.sdf(xyzs)

    sdf_scaled = sdf * s

    alpha = (torch.sigmoid(sdf_scaled[..., :-1]) - torch.sigmoid(sdf_scaled[..., 1:])) / torch.sigmoid(sdf_scaled[..., :-1])

    transmittance = torch.cumprod(1 - alpha, dim=-1)
    transmittance = torch.cat((transmittance.new_zeros((transmittance.shape[0], 1)), transmittance), dim=-1)

    alpha = torch.cat((alpha, alpha.new_zeros((alpha.shape[0], 1))), dim=-1)

    weight = transmittance * alpha

    # print("weight_sampling: ", torch.sum(weight, dim=-1))

    weight_re = regularize_weights(weight, alpha_importance, steps_firstpass)

    # print("weight_re_sampling: ", torch.sum(weight_re, dim=-1))

    z_vals_log_fine = sample_pdf(z_vals_log, weight_re, steps_importance)
    z_vals_fine = torch.pow(10, z_vals_log_fine)

    return z_vals_log_fine, z_vals_fine


class NeusRender(nn.Module):
    def __init__(self,
        model,
        steps_firstpass,
        z_bounds,
        steps_importance,
        alpha_importance):
        super().__init__()

        self.model = model
        self.steps_firstpass = steps_firstpass
        self.z_bounds = z_bounds
        self.steps_importance = steps_importance
        self.alpha_importance = alpha_importance

        self.s = torch.nn.Parameter(torch.full((1,), fill_value=3.0), requires_grad=True)

    def render(self, n, h, w, K, E, bg_color):
        rays_o, rays_d = get_rays(h, w, K, E)
        z_vals_log, z_vals = neus_efficient_sampling(
            self.model, rays_o, rays_d,
            self.steps_firstpass, self.z_bounds, self.steps_importance, self.alpha_importance, self.s)

        xyzs, dirs = get_sample_points(rays_o, rays_d, z_vals)
        n_expand = n[:, None].expand(-1, z_vals.shape[-1])

        sdf, color = self.model(xyzs, dirs, n_expand)

        # print(torch.amax(sdf), torch.amin(sdf))

        # plt.plot(sdf[0].detach().cpu().numpy())
        # plt.show()

        pixel, invdepth, weight = neus_render_rays_log(sdf, color, z_vals, z_vals_log, self.s)

        if bg_color is not None:
            pixel = pixel + (1 - torch.sum(weight, dim=-1)[..., None]) * bg_color

        if self.training:
            grad_theta = self.model.gradient(xyzs)
        else:
            grad_theta = None

        aux_outputs = {}
        aux_outputs['invdepth'] = invdepth.detach()

        return pixel, weight, grad_theta, aux_outputs