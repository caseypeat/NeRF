
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

from tqdm import tqdm


def rot90(image):
    if isinstance(image, torch.Tensor):
        if len(image.shape) == 2:
            return image.permute(1, 0)[:, ::-1]
        elif len(image.shape) == 3:
            return image.permute(1, 0, 2)[:, ::-1, :]
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            return image.transpose(1, 0)[:, ::-1]
        elif len(image.shape) == 3:
            return image.transpose(1, 0, 2)[:, ::-1, :]


class Inferencer(object):
    def __init__(
        self,
        dataloader,
        renderer,
        n_rays,
        image_num,
        # rotate,
        max_variance_npy,
        max_variance_pcd,
        distribution_area,
        
        cams,
        freq,
        side_buffer,
        ):
        
        self.dataloader = dataloader
        self.renderer = renderer
        self.n_rays = n_rays

        self.image_num = image_num
        # self.rotate = rotate

        self.max_variance_npy = max_variance_npy
        self.max_variance_pcd = max_variance_pcd
        self.distribution_area = distribution_area

        self.cams = cams
        self.freq = freq
        self.side_buffer = side_buffer


    @torch.no_grad()
    def render_image(self):
        n, h, w, K, E, _, _, _ = self.dataloader.get_image_batch(self.image_num, device='cuda')
        return self.render_image_(n, h, w, K, E)

    @torch.no_grad()
    def render_image_(self, n, h, w, K, E):

        # _f : flattened
        # _b : batched

        n_f = torch.reshape(n, (-1,))
        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        K_f = torch.reshape(K, (-1, 3, 3))
        E_f = torch.reshape(E, (-1, 4, 4))

        image_f = torch.zeros((*h_f.shape, 3), device='cpu')
        invdepth_f = torch.zeros(h_f.shape, device='cpu')

        for a in tqdm(range(0, len(n_f), self.n_rays)):
            b = min(len(n_f), a+self.n_rays)

            n_fb = n_f[a:b]
            h_fb = h_f[a:b]
            w_fb = w_f[a:b]

            K_fb = K_f[a:b]
            E_fb = E_f[a:b]

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            image_fb, _, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

            image_f[a:b] = image_fb.cpu()
            invdepth_f[a:b] = aux_outputs_fb['invdepth'].cpu()

        image = torch.reshape(image_f, (*h.shape, 3)).numpy()
        invdepth = torch.reshape(invdepth_f, h.shape).numpy()

        # for i in range(self.rotate // 90):
        #     image = rot90(image)
        #     invdepth = rot90(invdepth)

        return image, invdepth
        

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
    def render_invdepth_thresh(self):
        n, h, w, K, E, _, _, _ = self.dataloader.get_image_batch(self.image_num, device='cuda')
        return self.render_invdepth_thresh_(n, h, w, K, E)
    
    @torch.no_grad()
    def render_invdepth_thresh_(self, n, h, w, K, E):

        n_f = torch.reshape(n, (-1,))
        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        K_f = torch.reshape(K, (-1, 3, 3))
        E_f = torch.reshape(E, (-1, 4, 4))

        invdepth_thresh_f = torch.zeros(h_f.shape, device='cpu')

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

            weights_thresh_start_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 - self.distribution_area / 2)
            weights_thresh_mid_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5)
            weights_thresh_end_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 + self.distribution_area / 2)

            depth_start = torch.sum(weights_thresh_start_fb * z_vals_fb, dim=-1)
            depth_mid = torch.sum(weights_thresh_mid_fb * z_vals_fb, dim=-1)
            depth_end = torch.sum(weights_thresh_end_fb * z_vals_fb, dim=-1)

            impulse_mask = torch.abs(depth_start - depth_end) > self.max_variance_pcd

            invdepth_thresh_fb = 1 / depth_mid
            invdepth_thresh_fb[impulse_mask] = 0
            invdepth_thresh_fb[depth_mid == 0] = 0

            invdepth_thresh_f[a:b] = invdepth_thresh_fb.cpu()

        invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape).numpy()

        # for i in range(self.rotate // 90):
        #     invdepth_thresh = rot90(invdepth_thresh)

        return invdepth_thresh
        
    
    @torch.no_grad()
    def extract_surface_geometry(self):
        n, h, w, K, E, _, _, _ = self.dataloader.get_pointcloud_batch(cams=self.cams, freq=self.freq, side_buffer=self.side_buffer, device='cpu')
        return self.extract_surface_geometry_(n, h, w, K, E)

    @torch.no_grad()
    def extract_surface_geometry_(self, n, h, w, K, E):

        n_f = torch.reshape(n, (-1,))
        h_f = torch.reshape(h, (-1,))
        w_f = torch.reshape(w, (-1,))

        K_f = torch.reshape(K, (-1, 3, 3))
        E_f = torch.reshape(E, (-1, 4, 4))

        points = torch.zeros((0, 3), device='cpu')
        colors = torch.zeros((0, 3), device='cpu')
        depth_variance = torch.zeros((0, 1), device='cpu')
        # x_hashtable = torch.zeros((0, 40), device='cpu')

        for a in tqdm(range(0, len(n_f), self.n_rays)):
            b = min(len(n_f), a+self.n_rays)

            n_fb = n_f[a:b].to('cuda')
            h_fb = h_f[a:b].to('cuda')
            w_fb = w_f[a:b].to('cuda')

            K_fb = K_f[a:b].to('cuda')
            E_fb = E_f[a:b].to('cuda')

            color_bg = torch.ones(3, device='cuda') # [3], fixed white background

            _, weights_fb, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

            xyzs_fb = aux_outputs_fb['xyzs']
            xyzs_centered_fb = aux_outputs_fb['xyzs_centered']
            rgbs_fb = aux_outputs_fb['rgbs']
            z_vals_fb = aux_outputs_fb['z_vals']
            # x_hashtable_fb = aux_outputs_fb['x_hashtable']

            weights_thresh_start_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 - self.distribution_area / 2)
            weights_thresh_mid_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5)
            weights_thresh_end_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 + self.distribution_area / 2)

            depth_start = torch.sum(weights_thresh_start_fb * z_vals_fb, dim=-1)
            depth_end = torch.sum(weights_thresh_end_fb * z_vals_fb, dim=-1)
            depth_variance_fb = torch.abs(depth_start - depth_end)[:, None, None].expand(-1, weights_fb.shape[1], 1)
            
            impulse_mask = depth_variance_fb < self.max_variance_npy
            mid_transmittance_mask = weights_thresh_mid_fb.to(bool)[..., None]
            inside_inner_bound_mask = torch.linalg.norm(xyzs_centered_fb, dim=-1, keepdim=True) < self.renderer.inner_bound
            mask = (impulse_mask & mid_transmittance_mask & inside_inner_bound_mask)
            
            points_b = xyzs_fb[mask.expand(*xyzs_fb.shape)].reshape(-1, 3)
            colors_b = rgbs_fb[mask.expand(*rgbs_fb.shape)].reshape(-1, 3)
            depth_variance_b = depth_variance_fb[mask].reshape(-1, 1)
            # x_hashtable_b = x_hashtable_fb[mask.expand(*x_hashtable_fb.shape)].reshape(-1, 40)

            points = torch.cat([points, points_b.cpu()], dim=0)
            colors = torch.cat([colors, colors_b.cpu()], dim=0)
            depth_variance = torch.cat([depth_variance, depth_variance_b.cpu()], dim=0)
            # x_hashtable = torch.cat([x_hashtable, x_hashtable_b.cpu()], dim=0)

            # print(depth_variance_fb)

        pointcloud = {}
        pointcloud['points'] = points.numpy()
        pointcloud['colors'] = colors.numpy()
        pointcloud['depth_variance'] = depth_variance.numpy()
        # pointcloud['x_hashtable'] = x_hashtable.numpy()

        return pointcloud