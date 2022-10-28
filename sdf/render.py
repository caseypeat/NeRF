import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m

from typing import Union, Optional
from dataclasses import dataclass

from nets import NeRFNetwork, NeRFCoordinateWrapper
from render import get_rays, get_uniform_z_vals, efficient_sampling, get_sample_points, render_rays_log

from sdf.model.density import LaplaceDensity
from sdf.model.ray_sampler import ErrorBoundSampler


class SDFRender(nn.Module):
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

        # self.density = LaplaceDensity(
        #     params_init = {"beta": 0.1},
        #     beta_min = 0.0001
        # )
        self.scene_bounding_sphere = 100
        self.error_bound_sampler = ErrorBoundSampler(
            scene_bounding_sphere=self.scene_bounding_sphere,
            near=0.1,
            N_samples=64,
            N_samples_eval=128,
            N_samples_extra=32,
            eps=0.1,
            beta_iters=10,
            max_total_iters=5,
        )


    def render(self, n, h, w, K, E, bg_color):
        rays_o, rays_d = get_rays(h, w, K, E)
        # z_vals_log, z_vals = efficient_sampling(
        #     self.model, rays_o, rays_d,
        #     self.steps_firstpass, self.z_bounds, self.steps_importance, self.alpha_importance)

        z_vals, z_vals_eik = self.error_bound_sampler.get_z_vals(rays_d, rays_o, self.model)
        z_vals_log = torch.log10(z_vals)

        # print(z_vals_eik)
        # exit()

        xyzs, dirs = get_sample_points(rays_o, rays_d, z_vals)
        n_expand = n[:, None].expand(-1, z_vals.shape[-1])

        sigma, color = self.model(xyzs, dirs, n_expand)
        sigmas, colors = sigma[None, ...], color[None, ...]

        pixel, invdepth, weight = render_rays_log(sigmas, colors, z_vals, z_vals_log)

        xyzs_eik, dirs_eik = get_sample_points(rays_o, rays_d, z_vals_eik)

        # Sample points for the eikonal loss
        n_eik_points = z_vals.shape[0] * z_vals.shape[1]
        eikonal_points = torch.empty(n_eik_points, 3, device='cuda').uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere)
        # add some of the near surface points
        # eik_near_points = (rays_o.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
        eikonal_points = torch.cat([eikonal_points, xyzs_eik.reshape(-1, 3)], 0)

        grad_theta = self.model.gradient(eikonal_points)
        # grad_theta = self.model.gradient(xyzs)

        return pixel, weight, grad_theta