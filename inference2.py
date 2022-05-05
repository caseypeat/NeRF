
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

from tqdm import tqdm


class Inferencer(object):
    def __init__(
        self,
        renderer,
        n_rays,
        image_num,
        ):
        
        self.renderer = renderer
        self.n_rays = n_rays
        self.image_num = image_num


    @torch.no_grad()
    def render_image(self, n, h, w, K, E):

        # _f : flattened
        # _b : batched

        n_f = torch.reshape(n, (-1,))
        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        K_f = torch.reshape(K, (-1, 3, 3))
        E_f = torch.reshape(E, (-1, 4, 4))

        image_f = torch.zeros((*h_f.shape, 3))
        invdepth_f = torch.zeros(h_f.shape)

        for a in tqdm(range(0, len(n_f), self.n_rays)):
            b = min(len(n_f), a+self.n_rays)

            n_fb = n_f[a:b]
            h_fb = h_f[a:b]
            w_fb = w_f[a:b]

            K_fb = K_f[a:b]
            E_fb = E_f[a:b]

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            image_fb, _, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

            image_f[a:b] = image_fb
            invdepth_f[a:b] = aux_outputs_fb['invdepth']

        image = torch.reshape(image_f, (*h.shape, 3))
        invdepth = torch.reshape(invdepth_f, h.shape)

        return image, invdepth
    

    # @torch.no_grad()
    # def render_image(self, H, W, K, E):

    #     # _f : flattened
    #     # _b : batched

    #     K = K.to('cuda')
    #     E = E.to('cuda')

    #     h = torch.arange(0, H, device='cuda')
    #     w = torch.arange(0, W, device='cuda')

    #     h, w = torch.meshgrid(h, w, indexing='ij')

    #     h_f = torch.reshape(h, (-1,))
    #     w_f = torch.reshape(w, (-1,))

    #     image_f = torch.zeros((*h_f.shape, 3))
    #     invdepth_f = torch.zeros(h_f.shape)

    #     for a in tqdm(range(0, len(h_f), self.n_rays)):
    #         b = min(len(h_f), a+self.n_rays)

    #         h_fb = h_f[a:b]
    #         w_fb = w_f[a:b]
    #         n_fb = torch.full(h_fb.shape, fill_value=self.image_num, device='cuda')

    #         color_bg = torch.ones(3, device='cuda') # [3], fixed white background

    #         image_fb, _, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], bg_color=color_bg)

    #         image_f[a:b] = image_fb
    #         invdepth_f[a:b] = aux_outputs_fb['invdepth']

    #     image = torch.reshape(image_f, (*h.shape, 3))
    #     invdepth = torch.reshape(invdepth_f, h.shape)

    #     return image, invdepth


    @torch.no_grad()
    def calculate_cumlative_weights_thresh(self, weights, thresh):
        weights_cdf = torch.cumsum(torch.clone(weights), dim=-1)
        weights_cdf[weights_cdf < thresh] = 0
        weights_cdf[weights_cdf >= thresh] = 1
        weights_cdf_s = (1 - torch.cumsum(weights_cdf[..., :-1], dim=-1))
        weights_cdf_s[weights_cdf_s < 0] = 0
        weights_cdf[..., 1:] = weights_cdf[..., 1:] * weights_cdf_s
        return weights_cdf

    
    @torch.no_grad()
    def render_invdepth_thresh(self, n, h, w, K, E, thresh=0.05):

        n_f = torch.reshape(n, (-1,))
        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        K_f = torch.reshape(K, (-1, 3, 3))
        E_f = torch.reshape(E, (-1, 4, 4))

        invdepth_thresh_f = torch.zeros(h_f.shape)

        for a in tqdm(range(0, len(n_f), self.n_rays)):
            b = min(len(n_f), a+self.n_rays)

            n_fb = n_f[a:b]
            h_fb = h_f[a:b]
            w_fb = w_f[a:b]

            K_fb = K_f[a:b]
            E_fb = E_f[a:b]

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            _, weights_fb, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

            z_vals_fb = aux_outputs_fb['z_vals']

            weights_thresh_start_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.2)
            weights_thresh_mid_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5)
            weights_thresh_end_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.8)

            depth_start = torch.sum(weights_thresh_start_fb * z_vals_fb, dim=-1)
            depth_mid = torch.sum(weights_thresh_mid_fb * z_vals_fb, dim=-1)
            depth_end = torch.sum(weights_thresh_end_fb * z_vals_fb, dim=-1)

            impulse_mask = torch.abs(depth_start - depth_end) > thresh

            invdepth_thresh_fb = 1 / depth_mid
            invdepth_thresh_fb[impulse_mask] = 0
            invdepth_thresh_fb[depth_mid == 0] = 0

            invdepth_thresh_f[a:b] = invdepth_thresh_fb

        invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape)

        return invdepth_thresh
        

    @torch.no_grad()
    def extract_surface_geometry(self, n, h, w, K, E):

        n_f = torch.reshape(n, (-1,))
        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        K_f = torch.reshape(K, (-1, 3, 3))
        E_f = torch.reshape(E, (-1, 4, 4))

        points = torch.zeros((0, 3), device='cuda')
        colors = torch.zeros((0, 3), device='cuda')
        depth_variance = torch.zeros((0, 1), device='cuda')
        x_hashtable = torch.zeros((0, 40), device='cuda')

        for a in tqdm(range(0, len(n_f), self.n_rays)):
            b = min(len(n_f), a+self.n_rays)

            n_fb = n_f[a:b]
            h_fb = h_f[a:b]
            w_fb = w_f[a:b]

            K_fb = K_f[a:b]
            E_fb = E_f[a:b]

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            _, weights_fb, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

            xyzs_fb = aux_outputs_fb['xyzs']
            rgbs_fb = aux_outputs_fb['rgbs']
            z_vals_fb = aux_outputs_fb['z_vals']
            x_hashtable_fb = aux_outputs_fb['x_hashtable']

            weights_thresh_start_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.2)
            weights_thresh_mid_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5)
            weights_thresh_end_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.8)

            depth_start = torch.sum(weights_thresh_start_fb * z_vals_fb, dim=-1)
            depth_end = torch.sum(weights_thresh_end_fb * z_vals_fb, dim=-1)
            depth_variance_fb = torch.abs(depth_start - depth_end)[:, None, None].expand(-1, weights_fb.shape[1], 1)
            
            impulse_mask = depth_variance_fb < 0.1
            mid_transmittance_mask = weights_thresh_mid_fb.to(bool)[..., None]
            inside_inner_bound_mask = torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True) < self.renderer.inner_bound
            mask = (impulse_mask & mid_transmittance_mask & inside_inner_bound_mask)

            points_b = xyzs_fb[mask.expand(*xyzs_fb.shape)].reshape(-1, 3)
            colors_b = rgbs_fb[mask.expand(*rgbs_fb.shape)].reshape(-1, 3)
            depth_variance_b = depth_variance_fb[mask].reshape(-1, 1)
            x_hashtable_b = x_hashtable_fb[mask.expand(*x_hashtable_fb.shape)].reshape(-1, 40)

            points = torch.cat([points, points_b], dim=0)
            colors = torch.cat([colors, colors_b], dim=0)
            depth_variance = torch.cat([depth_variance, depth_variance_b], dim=0)
            x_hashtable = torch.cat([x_hashtable, x_hashtable_b], dim=0)

        pointcloud = {}
        pointcloud['points'] = points.cpu().numpy()
        pointcloud['colors'] = colors.cpu().numpy()
        pointcloud['depth_variance'] = depth_variance.cpu().numpy()
        pointcloud['x_hashtable'] = x_hashtable.cpu().numpy()

        return pointcloud



    # @torch.no_grad()
    # def render_invdepth_thresh(self, H, W, K, E, thresh=100):

    #     # _f : flattened
    #     # _b : batched

    #     K = K.to('cuda')
    #     E = E.to('cuda')

    #     h = torch.arange(0, H, device='cuda')
    #     w = torch.arange(0, W, device='cuda')

    #     h, w = torch.meshgrid(h, w, indexing='ij')

    #     h_f = torch.reshape(h, (-1,))
    #     w_f = torch.reshape(w, (-1,))

    #     invdepth_thresh_f = torch.zeros(h_f.shape)

    #     for a in tqdm(range(0, len(h_f), self.n_rays)):
    #         b = min(len(h_f), a+self.n_rays)

    #         h_fb = h_f[a:b]
    #         w_fb = w_f[a:b]
    #         n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

    #         color_bg = torch.ones(3, device='cuda') # [3], fixed white background

    #         _, _, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], bg_color=color_bg)

    #         z_vals_fb = aux_outputs_fb['z_vals']
    #         sigmas_thresh_fb = torch.clone(aux_outputs_fb['sigmas'])

    #         sigmas_thresh_fb[sigmas_thresh_fb < thresh] = 0
    #         sigmas_thresh_fb[sigmas_thresh_fb >= thresh] = 1
    #         sigmas_thresh_s_fb = (1 - torch.cumsum(sigmas_thresh_fb[..., :-1], dim=-1))
    #         sigmas_thresh_s_fb[sigmas_thresh_s_fb < 0] = 0
    #         sigmas_thresh_fb[..., 1:] = sigmas_thresh_fb[..., 1:] * sigmas_thresh_s_fb

    #         invdepth_thresh_fb = 1 / torch.sum(sigmas_thresh_fb * z_vals_fb, dim=-1)
    #         invdepth_thresh_fb[torch.sum(sigmas_thresh_fb, dim=-1) == 0] = 0

    #         invdepth_thresh_f[a:b] = invdepth_thresh_fb

    #     invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape)

    #     return invdepth_thresh


    # @torch.no_grad()
    # def render_invdepth_thresh_weights(self, H, W, K, E):

    #     # _f : flattened
    #     # _b : batched

    #     K = K.to('cuda')
    #     E = E.to('cuda')

    #     h = torch.arange(0, H, device='cuda')
    #     w = torch.arange(0, W, device='cuda')

    #     h, w = torch.meshgrid(h, w, indexing='ij')

    #     h_f = torch.reshape(h, (-1,))
    #     w_f = torch.reshape(w, (-1,))

    #     invdepth_thresh_f = torch.zeros(h_f.shape)

    #     for a in tqdm(range(0, len(h_f), self.n_rays)):
    #         b = min(len(h_f), a+self.n_rays)

    #         h_fb = h_f[a:b]
    #         w_fb = w_f[a:b]
    #         n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

    #         color_bg = torch.ones(3, device='cuda') # [3], fixed white background

    #         _, weights_fb, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], bg_color=color_bg)

    #         z_vals_fb = aux_outputs_fb['z_vals']

    #         weights_cdf_fb = torch.cumsum(weights_fb, dim=-1)

    #         thresh = 0.5
    #         weights_cdf_thresh_fb = torch.clone(weights_cdf_fb)
    #         weights_cdf_thresh_fb[weights_cdf_thresh_fb < thresh] = 0
    #         weights_cdf_thresh_fb[weights_cdf_thresh_fb >= thresh] = 1
    #         weights_cdf_thresh_s_fb = (1 - torch.cumsum(weights_cdf_thresh_fb[..., :-1], dim=-1))
    #         weights_cdf_thresh_s_fb[weights_cdf_thresh_s_fb < 0] = 0
    #         weights_cdf_thresh_fb[..., 1:] = weights_cdf_thresh_fb[..., 1:] * weights_cdf_thresh_s_fb

    #         invdepth_thresh_fb = 1 / torch.sum(weights_cdf_thresh_fb * z_vals_fb, dim=-1)
    #         invdepth_thresh_fb[torch.sum(weights_cdf_thresh_fb, dim=-1) == 0] = 0

    #         invdepth_thresh_f[a:b] = invdepth_thresh_fb

    #     invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape)

    #     return invdepth_thresh


    # @torch.no_grad()
    # def extract_geometry_rays(self, H, W, Ks, Es, max_sigma):
    #     N = Ks.shape[0]

    #     points = torch.zeros((0, 3), device='cuda')
    #     colors = torch.zeros((0, 3), device='cuda')

    #     for K, E in zip(Ks, Es):

    #         K = K.to('cuda')
    #         E = E.to('cuda')

    #         h = torch.arange(0, H, device='cuda')
    #         w = torch.arange(0, W, device='cuda')
            

    #         h, w = torch.meshgrid(h, w, indexing='ij')

    #         h_f = torch.reshape(h, (-1,))
    #         w_f = torch.reshape(w, (-1,))

    #         for a in tqdm(range(0, len(h_f), self.n_rays)):
    #             b = min(len(h_f), a+self.n_rays)

    #             h_fb = h_f[a:b]
    #             w_fb = w_f[a:b]
    #             n_fb = torch.full(h_fb.shape, fill_value=cfg.inference.image_num, device='cuda')

    #             color_bg = torch.ones(3, device='cuda') # [3], fixed white background

    #             image_fb, weights_fb, _, aux_outputs_fb = self.model.render(n_fb, h_fb, w_fb, K[None, ...], E[None, ...], color_bg)

    #             xyzs_fb = aux_outputs_fb['xyzs']
    #             rgbs_fb = aux_outputs_fb['rgbs']
    #             z_vals_fb = aux_outputs_fb['z_vals']
    #             z_vals_log_fb = aux_outputs_fb['z_vals_log']

    #             # delta = z_vals_fb.new_zeros(z_vals_fb.shape)
    #             # delta[:-1] = z_vals_fb[1:] - z_vals_fb[:-1]

    #             weights_cdf_fb = torch.cumsum(weights_fb, dim=-1)

    #             thresh_a = 0.2
    #             weights_cdf_thresh_a_fb = torch.clone(weights_cdf_fb)
    #             weights_cdf_thresh_a_fb[weights_cdf_thresh_a_fb < thresh_a] = 0
    #             weights_cdf_thresh_a_fb[weights_cdf_thresh_a_fb >= thresh_a] = 1
    #             weights_cdf_thresh_a_s_fb = (1 - torch.cumsum(weights_cdf_thresh_a_fb[..., :-1], dim=-1))
    #             weights_cdf_thresh_a_s_fb[weights_cdf_thresh_a_s_fb < 0] = 0
    #             weights_cdf_thresh_a_fb[..., 1:] = weights_cdf_thresh_a_fb[..., 1:] * weights_cdf_thresh_a_s_fb
                
    #             thresh_m = 0.5
    #             weights_cdf_thresh_m_fb = torch.clone(weights_cdf_fb)
    #             weights_cdf_thresh_m_fb[weights_cdf_thresh_m_fb < thresh_m] = 0
    #             weights_cdf_thresh_m_fb[weights_cdf_thresh_m_fb >= thresh_m] = 1
    #             weights_cdf_thresh_m_s_fb = (1 - torch.cumsum(weights_cdf_thresh_m_fb[..., :-1], dim=-1))
    #             weights_cdf_thresh_m_s_fb[weights_cdf_thresh_m_s_fb < 0] = 0
    #             weights_cdf_thresh_m_fb[..., 1:] = weights_cdf_thresh_m_fb[..., 1:] * weights_cdf_thresh_m_s_fb
                
    #             thresh_b = 0.8
    #             weights_cdf_thresh_b_fb = torch.clone(weights_cdf_fb)
    #             weights_cdf_thresh_b_fb[weights_cdf_thresh_b_fb < thresh_b] = 0
    #             weights_cdf_thresh_b_fb[weights_cdf_thresh_b_fb >= thresh_b] = 1
    #             weights_cdf_thresh_b_s_fb = (1 - torch.cumsum(weights_cdf_thresh_b_fb[..., :-1], dim=-1))
    #             weights_cdf_thresh_b_s_fb[weights_cdf_thresh_b_s_fb < 0] = 0
    #             weights_cdf_thresh_b_fb[..., 1:] = weights_cdf_thresh_b_fb[..., 1:] * weights_cdf_thresh_b_s_fb

    #             depth_a = torch.sum(weights_cdf_thresh_a_fb * z_vals_fb, dim=-1)
    #             depth_b = torch.sum(weights_cdf_thresh_b_fb * z_vals_fb, dim=-1)
    #             mask = torch.abs(depth_a - depth_b) < max_sigma


    #             points_b = xyzs_fb[weights_cdf_thresh_m_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound) & mask[:, None, None].expand(-1, weights_cdf_fb.shape[1], 3)].reshape(-1, 3)
    #             colors_b = rgbs_fb[weights_cdf_thresh_m_fb.to(bool)[..., None].expand(-1, -1, 3) & (torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True).expand(-1, -1, 3) < cfg.scene.inner_bound) & mask[:, None, None].expand(-1, weights_cdf_fb.shape[1], 3)].reshape(-1, 3)

    #             points = torch.cat([points, points_b], dim=0)
    #             colors = torch.cat([colors, colors_b], dim=0)

    #     return points.cpu().numpy(), colors.cpu().numpy()

