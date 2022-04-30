import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m

import helpers

from config import cfg


# https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py
def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


class NerfRenderer(nn.Module):
    def __init__(self, intrinsics, extrinsics):
        super().__init__()

        self.intrinsics = nn.Parameter(intrinsics, requires_grad=False)
        self.extrinsics = nn.Parameter(extrinsics, requires_grad=False)

        # self.T = nn.Parameter(torch.zeros(3, dtype=torch.float32, device='cuda'), requires_grad=True)
        # self.R = nn.Parameter(torch.zeros(3, dtype=torch.float32, device='cuda'), requires_grad=True)

        self.inner_bound = cfg.scene.inner_bound
        self.outer_bound = cfg.scene.outer_bound
        self.z_bounds = cfg.renderer.z_bounds
        self.steps = cfg.renderer.steps

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()


    @torch.no_grad()
    def get_uniform_z_vals(self, rays_o, rays_d, n_samples):
        z_vals_log = torch.empty((0), device=rays_o.device)
        for i in range(len(self.steps)):
            z_vals_log = torch.cat((z_vals_log, torch.linspace(m.log10(self.z_bounds[i]), m.log10(self.z_bounds[i+1])-(m.log10(self.z_bounds[i+1])-m.log10(self.z_bounds[i]))/self.steps[i], self.steps[i], device=rays_o.device)))
        z_vals_log = z_vals_log.expand(rays_o.shape[0], -1)
        z_vals = torch.pow(10, z_vals_log)
        return z_vals_log, z_vals

    
    @torch.no_grad()
    def reweight_the_weights(self, weights, alpha, steps):
        weights = torch.cat([weights.new_zeros((weights.shape[0], 1)), weights, weights.new_zeros((weights.shape[0], 1))], dim=-1)
        weights = 1/2 * (torch.maximum(weights[..., :-2], weights[..., 1:-1]) + torch.maximum(weights[..., 1:-1], weights[..., 2:]))
        weights = weights + alpha / weights.shape[-1]
        c = 0
        for i in range(len(steps)):
            weights[:, c:c+steps[i]] *= steps[i] / sum(steps)
            c += steps[i]
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, N_samples-1]
        return weights

    
    @torch.no_grad()
    def efficient_sampling(self, rays_o, rays_d, n_samples):
        z_vals_log, z_vals = self.get_uniform_z_vals(rays_o, rays_d, self.steps)

        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, self.inner_bound, self.outer_bound)

        sigmas = self.density(s_xyzs)

        delta = z_vals_log.new_zeros(sigmas.shape)  # [N_rays, N_samples]
        delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

        alpha = 1 - torch.exp(-sigmas * delta)  # [N_rays, N_samples]

        alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

        weights_re = self.reweight_the_weights(weights, cfg.renderer.alpha, self.steps)

        z_vals_log_fine = helpers.sample_pdf(z_vals_log, weights_re, n_samples)
        z_vals_fine = torch.pow(10, z_vals_log_fine)

        return z_vals_log_fine, z_vals_fine



    def render(self, n, h, w, K, E, bg_color):

        # E_c = torch.clone(E)
        # with torch.cuda.amp.autocast(enabled=False):
        #     transform = torch.eye(4, dtype=torch.float32, device='cuda')
        #     transform[:3, :3] = Exp(self.R)
        #     transform[:3, 3] = self.T
        #     # transform[:3, 3] = self.T
        #     # a = E[n < 1176]
        #     E_c[n < 1176] = transform @ E_c[n < 1176]

        rays_o, rays_d = helpers.get_rays(h, w, K, E)
        z_vals_log, z_vals = self.efficient_sampling(rays_o, rays_d, cfg.renderer.importance_steps)
        xyzs, dirs = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs = helpers.mipnerf360_scale(xyzs, self.inner_bound, self.outer_bound)
        n_expand = n[:, None].expand(-1, z_vals.shape[-1])

        sigmas, rgbs = self(s_xyzs, dirs, n_expand)

        image, invdepth, weights = helpers.render_rays_log(sigmas, rgbs, z_vals, z_vals_log)
        image = image + (1 - torch.sum(weights, dim=-1)[..., None]) * bg_color
        z_vals_log_s = (z_vals_log - m.log10(self.z_bounds[0])) / (m.log10(self.z_bounds[-1]) - m.log10(self.z_bounds[0]))

        # auxilary outputs that may be useful for inference of analysis, but not used in training
        aux_outputs = {}
        aux_outputs['invdepth'] = invdepth
        aux_outputs['z_vals'] = z_vals
        aux_outputs['z_vals_log'] = z_vals_log
        aux_outputs['sigmas'] = sigmas
        aux_outputs['rgbs'] = rgbs
        aux_outputs['xyzs'] = xyzs
        aux_outputs['R'] = torch.clone(self.R)
        aux_outputs['T'] = torch.clone(self.T)

        return image, weights, z_vals_log_s, aux_outputs


def log_base(x, base):
    return torch.log(x) / m.log(base)