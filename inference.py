from turtle import bgcolor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

import helpers

from tqdm import tqdm

from config import cfg


class Inferencer(object):
    def __init__(self, model):
        
        self.model = model

        self.n_rays = cfg.inference.n_rays
        self.voxel_res = cfg.inference.voxel_res
        self.thresh = cfg.inference.thresh
        self.batch_size = cfg.inference.batch_size
    
    @torch.no_grad()
    def render_image(self, H, W, K, E):

        # _f : flattened
        # _b : batched

        K = K.to('cuda')
        E = E.to('cuda')

        h = torch.arange(0, H, device='cuda')
        w = torch.arange(0, W, device='cuda')

        h, w = torch.meshgrid(h, w, indexing='ij')

        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        image_f = torch.zeros((*h_f.shape, 3))
        invdepth_f = torch.zeros(h_f.shape)
        # weights_f = torch.zeros((*h_f.shape, cfg.renderer.importance_steps))
        # z_vals_s_f = torch.zeros((*h_f.shape, cfg.renderer.importance_steps))

        for a in tqdm(range(0, len(h_f), self.n_rays)):
            b = min(len(h_f), a+self.n_rays)

            h_fb = h_f[a:b]
            w_fb = w_f[a:b]
            n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            image_fb, _, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], bg_color=color_bg)

            image_f[a:b] = image_fb
            invdepth_f[a:b] = aux_outputs_fb['invdepth']

        image = torch.reshape(image_f, (*h.shape, 3))
        invdepth = torch.reshape(invdepth_f, h.shape)

        return image, invdepth


    @torch.no_grad()
    def render_invdepth_thresh(self, H, W, K, E, thresh=100):

        # _f : flattened
        # _b : batched

        K = K.to('cuda')
        E = E.to('cuda')

        h = torch.arange(0, H, device='cuda')
        w = torch.arange(0, W, device='cuda')

        h, w = torch.meshgrid(h, w, indexing='ij')

        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        invdepth_thresh_f = torch.zeros(h_f.shape)

        for a in tqdm(range(0, len(h_f), self.n_rays)):
            b = min(len(h_f), a+self.n_rays)

            h_fb = h_f[a:b]
            w_fb = w_f[a:b]
            n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            _, _, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], bg_color=color_bg)

            z_vals_fb = aux_outputs_fb['z_vals']
            sigmas_thresh_fb = torch.clone(aux_outputs_fb['sigmas'])

            sigmas_thresh_fb[sigmas_thresh_fb < thresh] = 0
            sigmas_thresh_fb[sigmas_thresh_fb >= thresh] = 1
            sigmas_thresh_s_fb = (1 - torch.cumsum(sigmas_thresh_fb[..., :-1], dim=-1))
            sigmas_thresh_s_fb[sigmas_thresh_s_fb < 0] = 0
            sigmas_thresh_fb[..., 1:] = sigmas_thresh_fb[..., 1:] * sigmas_thresh_s_fb

            invdepth_thresh_fb = 1 / torch.sum(sigmas_thresh_fb * z_vals_fb, dim=-1)
            invdepth_thresh_fb[torch.sum(sigmas_thresh_fb, dim=-1) == 0] = 0

            invdepth_thresh_f[a:b] = invdepth_thresh_fb

        invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape)

        return invdepth_thresh


    @torch.no_grad()
    def render_invdepth_thresh_weights(self, H, W, K, E):

        # _f : flattened
        # _b : batched

        K = K.to('cuda')
        E = E.to('cuda')

        h = torch.arange(0, H, device='cuda')
        w = torch.arange(0, W, device='cuda')

        h, w = torch.meshgrid(h, w, indexing='ij')

        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        invdepth_thresh_f = torch.zeros(h_f.shape)

        for a in tqdm(range(0, len(h_f), self.n_rays)):
            b = min(len(h_f), a+self.n_rays)

            h_fb = h_f[a:b]
            w_fb = w_f[a:b]
            n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            _, weights_fb, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], bg_color=color_bg)

            z_vals_fb = aux_outputs_fb['z_vals']

            weights_cdf_fb = torch.cumsum(weights_fb, dim=-1)

            thresh = 0.5
            weights_cdf_thresh_fb = torch.clone(weights_cdf_fb)
            weights_cdf_thresh_fb[weights_cdf_thresh_fb < thresh] = 0
            weights_cdf_thresh_fb[weights_cdf_thresh_fb >= thresh] = 1
            weights_cdf_thresh_s_fb = (1 - torch.cumsum(weights_cdf_thresh_fb[..., :-1], dim=-1))
            weights_cdf_thresh_s_fb[weights_cdf_thresh_s_fb < 0] = 0
            weights_cdf_thresh_fb[..., 1:] = weights_cdf_thresh_fb[..., 1:] * weights_cdf_thresh_s_fb

            invdepth_thresh_fb = 1 / torch.sum(weights_cdf_thresh_fb * z_vals_fb, dim=-1)
            invdepth_thresh_fb[torch.sum(weights_cdf_thresh_fb, dim=-1) == 0] = 0

            invdepth_thresh_f[a:b] = invdepth_thresh_fb

        invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape)

        return invdepth_thresh


    @torch.no_grad()
    def extract_geometry(self, N, H, W, K, E):

        mask = helpers.get_valid_positions(N, H, W, K.to('cuda'), E.to('cuda'), res=128)

        voxels = torch.linspace(-1+1/self.voxel_res, 1-1/self.voxel_res, self.voxel_res, device='cuda')

        num_samples = self.voxel_res**3

        points = torch.zeros((0, 3), device='cuda')
        colors = torch.zeros((0, 3), device='cuda')

        for a in tqdm(range(0, num_samples, self.batch_size)):
            b = min(num_samples, a+self.batch_size)

            n = torch.arange(a, b)

            x = voxels[torch.div(n, self.voxel_res**2, rounding_mode='floor')]
            y = voxels[torch.div(n, self.voxel_res, rounding_mode='floor') % self.voxel_res]
            z = voxels[n % self.voxel_res]

            xyz = torch.stack((x, y, z), dim=-1).cuda()

            x_i = ((x+1)/2*mask.shape[0]).to(int)
            y_i = ((y+1)/2*mask.shape[1]).to(int)
            z_i = ((z+1)/2*mask.shape[2]).to(int)

            xyz = xyz[mask[x_i, y_i, z_i]].view(-1, 3)
            
            dirs = torch.Tensor(np.array([0, 0, 1]))[None, ...].expand(xyz.shape[0], 3).cuda()
            n_i = torch.zeros((xyz.shape[0]), dtype=int).cuda()

            if xyz.shape[0] > 0:
                sigmas, rgbs = self.model(xyz, dirs, n_i)
                # new_points = torch.cat((xyz[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3), sigmas[sigmas > self.thresh][..., None]), dim=-1)
                new_points = xyz[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3)
                points = torch.cat((points, new_points))
                new_colors = rgbs[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3)
                colors = torch.cat((colors, new_colors))

        return points, colors


    @torch.no_grad()
    def extract_geometry_rays(self, H, W, Ks, Es, thresh=100):
        N = Ks.shape[0]

        points = torch.zeros((0, 3), device='cuda')
        colors = torch.zeros((0, 3), device='cuda')

        for K, E in zip(Ks, Es):

            K = K.to('cuda')
            E = E.to('cuda')

            h = torch.arange(0, H, device='cuda')
            w = torch.arange(0, W, device='cuda')
            

            h, w = torch.meshgrid(h, w, indexing='ij')

            h_f = torch.reshape(h, (-1,))
            w_f = torch.reshape(w, (-1,))

            for a in tqdm(range(0, len(h_f), self.n_rays)):
                b = min(len(h_f), a+self.n_rays)

                h_fb = h_f[a:b]
                w_fb = w_f[a:b]
                n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

                color_bg = torch.ones(3, device='cuda') # [3], fixed white background

                image_fb, _, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], color_bg)

                xyzs_fb = aux_outputs_fb['xyzs']
                rgbs_fb = aux_outputs_fb['rgbs']
                sigmas_thresh_fb = torch.clone(aux_outputs_fb['sigmas'])

                sigmas_thresh_fb[sigmas_thresh_fb < thresh] = 0
                sigmas_thresh_fb[sigmas_thresh_fb >= thresh] = 1
                sigmas_thresh_s_fb = (1 - torch.cumsum(sigmas_thresh_fb[..., :-1], dim=-1))
                sigmas_thresh_s_fb[sigmas_thresh_s_fb < 0] = 0
                sigmas_thresh_fb[..., 1:] = sigmas_thresh_fb[..., 1:] * sigmas_thresh_s_fb

                points_b = xyzs_fb[sigmas_thresh_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound)].reshape(-1, 3)
                colors_b = rgbs_fb[sigmas_thresh_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound)].reshape(-1, 3)

                points = torch.cat([points, points_b], dim=0)
                colors = torch.cat([colors, colors_b], dim=0)

        return points.cpu().numpy(), colors.cpu().numpy()


    @torch.no_grad()
    def extract_geometry_rays_weights(self, H, W, Ks, Es, thresh=0.5):
        N = Ks.shape[0]

        points = torch.zeros((0, 3), device='cuda')
        colors = torch.zeros((0, 3), device='cuda')

        for K, E in zip(Ks, Es):

            K = K.to('cuda')
            E = E.to('cuda')

            h = torch.arange(0, H, device='cuda')
            w = torch.arange(0, W, device='cuda')
            

            h, w = torch.meshgrid(h, w, indexing='ij')

            h_f = torch.reshape(h, (-1,))
            w_f = torch.reshape(w, (-1,))

            for a in tqdm(range(0, len(h_f), self.n_rays)):
                b = min(len(h_f), a+self.n_rays)

                h_fb = h_f[a:b]
                w_fb = w_f[a:b]
                n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

                color_bg = torch.ones(3, device='cuda') # [3], fixed white background

                image_fb, weights_fb, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], color_bg)

                xyzs_fb = aux_outputs_fb['xyzs']
                rgbs_fb = aux_outputs_fb['rgbs']
                z_vals_fb = aux_outputs_fb['z_vals']

                weights_cdf_fb = torch.cumsum(weights_fb, dim=-1)

                weights_cdf_thresh_fb = torch.clone(weights_cdf_fb)
                weights_cdf_thresh_fb[weights_cdf_thresh_fb < thresh] = 0
                weights_cdf_thresh_fb[weights_cdf_thresh_fb >= thresh] = 1
                weights_cdf_thresh_s_fb = (1 - torch.cumsum(weights_cdf_thresh_fb[..., :-1], dim=-1))
                weights_cdf_thresh_s_fb[weights_cdf_thresh_s_fb < 0] = 0
                weights_cdf_thresh_fb[..., 1:] = weights_cdf_thresh_fb[..., 1:] * weights_cdf_thresh_s_fb

                points_b = xyzs_fb[weights_cdf_thresh_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound)].reshape(-1, 3)
                colors_b = rgbs_fb[weights_cdf_thresh_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound)].reshape(-1, 3)

                points = torch.cat([points, points_b], dim=0)
                colors = torch.cat([colors, colors_b], dim=0)

        return points.cpu().numpy(), colors.cpu().numpy()


    @torch.no_grad()
    def extract_geometry_rays_weights2(self, H, W, Ks, Es, max_sigma):
        N = Ks.shape[0]

        points = torch.zeros((0, 3), device='cuda')
        colors = torch.zeros((0, 3), device='cuda')

        for K, E in zip(Ks, Es):

            K = K.to('cuda')
            E = E.to('cuda')

            h = torch.arange(0, H, device='cuda')
            w = torch.arange(0, W, device='cuda')
            

            h, w = torch.meshgrid(h, w, indexing='ij')

            h_f = torch.reshape(h, (-1,))
            w_f = torch.reshape(w, (-1,))

            for a in tqdm(range(0, len(h_f), self.n_rays)):
                b = min(len(h_f), a+self.n_rays)

                h_fb = h_f[a:b]
                w_fb = w_f[a:b]
                n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

                color_bg = torch.ones(3, device='cuda') # [3], fixed white background

                image_fb, weights_fb, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], color_bg)

                xyzs_fb = aux_outputs_fb['xyzs']
                rgbs_fb = aux_outputs_fb['rgbs']
                z_vals_fb = aux_outputs_fb['z_vals']
                z_vals_log_fb = aux_outputs_fb['z_vals_log']

                delta = z_vals_fb.new_zeros(z_vals_fb.shape)
                delta[:-1] = z_vals_fb[1:] - z_vals_fb[:-1]

                weights_cdf_fb = torch.cumsum(weights_fb*delta, dim=-1)
                # print(weights_cdf_fb.shape)
                # for i in range(weights_fb.shape[0]):
                #     if weights_cdf_fb[i, -1] > 0.9:
                #         pdf = weights_fb[i].cpu().numpy()
                #         cdf = weights_cdf_fb[i].cpu().numpy()
                #         z = z_vals_fb[i].cpu().numpy()

                #         mask = z < 1

                #         if cdf[mask][-1] > 0.9:

                #             fig, ax = plt.subplots(1, 2)
                #             ax[0].plot(z[mask], pdf[mask])
                #             ax[1].plot(z[mask], cdf[mask])
                #             plt.show()

                thresh_a = 0.2
                weights_cdf_thresh_a_fb = torch.clone(weights_cdf_fb)
                weights_cdf_thresh_a_fb[weights_cdf_thresh_a_fb < thresh_a] = 0
                weights_cdf_thresh_a_fb[weights_cdf_thresh_a_fb >= thresh_a] = 1
                weights_cdf_thresh_a_s_fb = (1 - torch.cumsum(weights_cdf_thresh_a_fb[..., :-1], dim=-1))
                weights_cdf_thresh_a_s_fb[weights_cdf_thresh_a_s_fb < 0] = 0
                weights_cdf_thresh_a_fb[..., 1:] = weights_cdf_thresh_a_fb[..., 1:] * weights_cdf_thresh_a_s_fb
                
                thresh_m = 0.5
                weights_cdf_thresh_m_fb = torch.clone(weights_cdf_fb)
                weights_cdf_thresh_m_fb[weights_cdf_thresh_m_fb < thresh_m] = 0
                weights_cdf_thresh_m_fb[weights_cdf_thresh_m_fb >= thresh_m] = 1
                weights_cdf_thresh_m_s_fb = (1 - torch.cumsum(weights_cdf_thresh_m_fb[..., :-1], dim=-1))
                weights_cdf_thresh_m_s_fb[weights_cdf_thresh_m_s_fb < 0] = 0
                weights_cdf_thresh_m_fb[..., 1:] = weights_cdf_thresh_m_fb[..., 1:] * weights_cdf_thresh_m_s_fb
                
                thresh_b = 0.8
                weights_cdf_thresh_b_fb = torch.clone(weights_cdf_fb)
                weights_cdf_thresh_b_fb[weights_cdf_thresh_b_fb < thresh_b] = 0
                weights_cdf_thresh_b_fb[weights_cdf_thresh_b_fb >= thresh_b] = 1
                weights_cdf_thresh_b_s_fb = (1 - torch.cumsum(weights_cdf_thresh_b_fb[..., :-1], dim=-1))
                weights_cdf_thresh_b_s_fb[weights_cdf_thresh_b_s_fb < 0] = 0
                weights_cdf_thresh_b_fb[..., 1:] = weights_cdf_thresh_b_fb[..., 1:] * weights_cdf_thresh_b_s_fb

                depth_a = torch.sum(weights_cdf_thresh_a_fb * z_vals_fb, dim=-1)
                depth_b = torch.sum(weights_cdf_thresh_b_fb * z_vals_fb, dim=-1)
                mask = torch.abs(depth_a - depth_b) < max_sigma


                points_b = xyzs_fb[weights_cdf_thresh_m_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound) & mask[:, None, None].expand(-1, weights_cdf_fb.shape[1], 3)].reshape(-1, 3)
                colors_b = rgbs_fb[weights_cdf_thresh_m_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound) & mask[:, None, None].expand(-1, weights_cdf_fb.shape[1], 3)].reshape(-1, 3)

                points = torch.cat([points, points_b], dim=0)
                colors = torch.cat([colors, colors_b], dim=0)

        return points.cpu().numpy(), colors.cpu().numpy()

