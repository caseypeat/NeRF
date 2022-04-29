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
    def extract_geometry(self, N, H, W, K, E):

        mask = helpers.get_valid_positions(N, H, W, K.to('cuda'), E.to('cuda'), res=128)

        voxels = torch.linspace(-1+1/self.voxel_res, 1-1/self.voxel_res, self.voxel_res, device='cuda')

        num_samples = self.voxel_res**3

        points = torch.zeros((0, 4), device='cuda')

        for a in tqdm(range(0, num_samples, self.batch_size)):
            b = min(num_samples, a+self.batch_size)

            n = torch.arange(a, b)

            x = voxels[torch.div(n, self.voxel_res**2, rounding_mode='floor')]
            y = voxels[torch.div(n, self.voxel_res, rounding_mode='floor') % self.voxel_res]
            z = voxels[n % self.voxel_res]

            xyz = torch.stack((x, y, z), dim=-1)

            x_i = ((x+1)/2*mask.shape[0]).to(int)
            y_i = ((y+1)/2*mask.shape[1]).to(int)
            z_i = ((z+1)/2*mask.shape[2]).to(int)

            xyz = xyz[mask[x_i, y_i, z_i]].view(-1, 3)

            if xyz.shape[0] > 0:
                sigmas = self.model.density(xyz).to(torch.float32)
                new_points = torch.cat((xyz[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3), sigmas[sigmas > self.thresh][..., None]), dim=-1)
                points = torch.cat((points, new_points))

        return points


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

