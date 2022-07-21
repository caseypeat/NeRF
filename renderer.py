import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m

import helpers


class NerfRenderer(nn.Module):
    def __init__(
        self,
        model,
        
        inner_bound,
        outer_bound,

        z_bounds,
        steps_firstpass,
        steps_importance,
        alpha_importance,
        
        translation_center):
        
        super().__init__()

        self.model = model

        self.inner_bound = inner_bound
        self.outer_bound = outer_bound

        self.z_bounds = z_bounds
        self.steps_firstpass = steps_firstpass
        self.steps_importance = steps_importance
        self.alpha_importance = alpha_importance

        self.translation_center = translation_center

        
    ## Rendering Pipeline
    
    def get_rays(self, h, w, K, E):
        """ Ray generation for Oliver's calibration format """
        dirs = torch.stack([w+0.5, h+0.5, torch.ones_like(w)], -1)  # [N_rays, 3]
        dirs = torch.inverse(K) @ dirs[:, :, None]  # [N_rays, 3, 1]

        rays_d = E[:, :3, :3] @ dirs  # [N_rays, 3, 1]
        rays_d = torch.squeeze(rays_d, dim=2)  # [N_rays, 3]

        rays_o = E[:, :3, -1].expand(rays_d.shape[0], -1)

        return rays_o, rays_d

    
    def get_sample_points(self, rays_o, rays_d, z_vals):
        N_rays, N_samples = z_vals.shape

        query_points = rays_d.new_zeros((N_rays, N_samples, 3))  # [N_rays, N_samples, 3]
        query_points = rays_o[:, None, :].expand(-1, N_samples, -1) + rays_d[:, None, :].expand(-1, N_samples, -1) * z_vals[..., None]  # [N_rays, N_samples, 3]

        norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True)
        query_dirs = rays_d / norm
        query_dirs = query_dirs[:, None, :].expand(-1, N_samples, -1)

        return query_points, query_dirs

    
    def mipnerf360_scale(self, xyzs, bound_inner, bound_outer):
        d = torch.linalg.norm(xyzs, dim=-1)[..., None].expand(-1, -1, 3)
        s_xyzs = torch.clone(xyzs)
        s_xyzs[d > bound_inner] = s_xyzs[d > bound_inner] * ((bound_outer - (bound_outer - bound_inner) / d[d > bound_inner]) / d[d > bound_inner])
        return s_xyzs

    
    def render_rays_log(self, sigmas, rgbs, z_vals, z_vals_log):
        N_rays, N_samples = z_vals.shape[:2]

        delta = z_vals_log.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
        delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

        alpha = 1 - torch.exp(-sigmas * delta)  # [N_rays, N_samples]

        alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

        rgb = torch.sum(weights[..., None] * rgbs, -2)  # [N_rays, 3]
        invdepth = torch.sum(weights / z_vals, -1)  # [N_rays]

        return rgb, invdepth, weights

    @torch.no_grad()
    def sample_pdf(self, z_vals, weights, N_importance):

        N_rays, N_samples = z_vals.shape

        pdf = weights  # [N_rays, N_samples]
        cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, N_samples]

        # print(cdf.shape)

        z_vals_mid = (z_vals[:, :-1] + z_vals[:, 1:]) / 2  # [N_rays, N_samples]

        # Take uniform samples
        u = torch.linspace(0, 1-1/N_importance, N_importance, device=weights.device)  # [N_rays, N_importance]
        u = u[None, :].expand(N_rays, -1)
        u = u + torch.rand([N_rays, N_importance], device=weights.device)/N_importance  # [N_rays, N_samples]
        u = u * (cdf[:, -2, None] - cdf[:, 1, None])
        u = u + cdf[:, 1, None]

        inds = torch.searchsorted(cdf[:, 1:-2], u, right=True) + 1  # [N_rays, N_importance]

        cdf_below = torch.gather(input=cdf, dim=1, index=inds-1)
        cdf_above = torch.gather(input=cdf, dim=1, index=inds)
        t = (u - cdf_below) / (cdf_above - cdf_below)

        z_vals_mid_below = torch.gather(input=z_vals_mid, dim=1, index=inds-1)
        z_vals_mid_above = torch.gather(input=z_vals_mid, dim=1, index=inds)
        z_vals_im = z_vals_mid_below + (z_vals_mid_above - z_vals_mid_below) * t

        z_vals_fine = z_vals_im

        return z_vals_fine


    ## Calculate density distribiution along ray using forward only first pass to inform second pass
    @torch.no_grad()
    def get_uniform_z_vals(self, rays_o, rays_d, n_samples):
        z_vals_log = torch.empty((0), device=rays_o.device)
        for i in range(len(self.steps_firstpass)):
            z_vals_log = torch.cat((z_vals_log, torch.linspace(m.log10(self.z_bounds[i]), m.log10(self.z_bounds[i+1])-(m.log10(self.z_bounds[i+1])-m.log10(self.z_bounds[i]))/self.steps_firstpass[i], self.steps_firstpass[i], device=rays_o.device)))
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
    def efficient_sampling(self, rays_o, rays_d, n_samples, alpha_const):
        z_vals_log, z_vals = self.get_uniform_z_vals(rays_o, rays_d, self.steps_firstpass)

        xyzs, dirs = self.get_sample_points(rays_o, rays_d, z_vals)
        xyzs_centered = xyzs - self.translation_center.to(xyzs.device)
        s_xyzs = self.mipnerf360_scale(xyzs_centered, self.inner_bound, self.outer_bound)

        sigmas, _ = self.model.density(s_xyzs, self.outer_bound)

        delta = z_vals_log.new_zeros(sigmas.shape)  # [N_rays, N_samples]
        delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

        alpha = 1 - torch.exp(-sigmas * delta)  # [N_rays, N_samples]

        alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

        weights_re = self.reweight_the_weights(weights, alpha_const, self.steps_firstpass)

        z_vals_log_fine = self.sample_pdf(z_vals_log, weights_re, n_samples)
        z_vals_fine = torch.pow(10, z_vals_log_fine)

        return z_vals_log_fine, z_vals_fine


    def render(self, n, h, w, K, E, bg_color):

        rays_o, rays_d = self.get_rays(h, w, K, E)
        z_vals_log, z_vals = self.efficient_sampling(rays_o, rays_d, self.steps_importance, self.alpha_importance)
        xyzs, dirs = self.get_sample_points(rays_o, rays_d, z_vals)
        xyzs_centered = xyzs - self.translation_center.to(xyzs.device)
        s_xyzs = self.mipnerf360_scale(xyzs_centered, self.inner_bound, self.outer_bound)
        n_expand = n[:, None].expand(-1, z_vals.shape[-1])

        sigmas, rgbs, aux_outputs_net = self.model(s_xyzs, dirs, n_expand, self.outer_bound)

        image, invdepth, weights = self.render_rays_log(sigmas, rgbs, z_vals, z_vals_log)
        image = image + (1 - torch.sum(weights, dim=-1)[..., None]) * bg_color
        z_vals_log_s = (z_vals_log - m.log10(self.z_bounds[0])) / (m.log10(self.z_bounds[-1]) - m.log10(self.z_bounds[0]))

        # auxilary outputs that may be useful for inference of analysis, but not used in training
        aux_outputs = {}
        aux_outputs['invdepth'] = invdepth.detach()
        aux_outputs['z_vals'] = z_vals.detach()
        aux_outputs['z_vals_log'] = z_vals_log.detach()
        aux_outputs['sigmas'] = sigmas.detach()
        aux_outputs['rgbs'] = rgbs.detach()
        aux_outputs['xyzs'] = xyzs.detach()
        aux_outputs['x_hashtable'] = aux_outputs_net['x_hashtable']

        return image, weights, z_vals_log_s, aux_outputs