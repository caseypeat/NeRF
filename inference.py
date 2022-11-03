
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

from tqdm import tqdm

from render import render_nerf


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


# class Inferencer(object):
#     def __init__(
#         self,
#         renderer,
#         n_rays,
#         image_num,
#         rotate,
#         max_variance_npy,
#         max_variance_pcd,
#         distribution_area,
#         cams,
#         freq,
#         ):
        
#         self.renderer = renderer
#         self.n_rays = n_rays

#         self.image_num = image_num
#         self.rotate = rotate

#         self.max_variance_npy = max_variance_npy
#         self.max_variance_pcd = max_variance_pcd
#         self.distribution_area = distribution_area

#         self.cams = cams
#         self.freq = freq

#     @torch.no_grad()
#     def render_image(self, n, h, w, K, E):

#         # _f : flattened
#         # _b : batched

#         n_f = torch.reshape(n, (-1,))
#         h_f = torch.reshape(h, (-1,))
#         w_f = torch.reshape(w, (-1,))

#         K_f = torch.reshape(K, (-1, 3, 3))
#         E_f = torch.reshape(E, (-1, 4, 4))

#         image_f = torch.zeros((*h_f.shape, 3), device='cpu')
#         invdepth_f = torch.zeros(h_f.shape, device='cpu')

#         for a in tqdm(range(0, len(n_f), self.n_rays)):
#             b = min(len(n_f), a+self.n_rays)

#             n_fb = n_f[a:b]
#             h_fb = h_f[a:b]
#             w_fb = w_f[a:b]

#             K_fb = K_f[a:b]
#             E_fb = E_f[a:b]

#             color_bg = torch.ones(3, device='cuda') # [3], fixed white background

#             image_fb, _, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

#             image_f[a:b] = image_fb.cpu()
#             invdepth_f[a:b] = aux_outputs_fb['invdepth'].cpu()

#         image = torch.reshape(image_f, (*h.shape, 3)).numpy()
#         invdepth = torch.reshape(invdepth_f, h.shape).numpy()

#         # for i in range(self.rotate // 90):
#         #     image = rot90(image)
#         #     invdepth = rot90(invdepth)

#         return image, invdepth
        

#     @torch.no_grad()
#     def calculate_cumlative_weights_thresh(self, weights, thresh):
#         weights_cdf = torch.cumsum(torch.clone(weights), dim=-1)
#         weights_cdf[weights_cdf < thresh] = 0
#         weights_cdf[weights_cdf >= thresh] = 1
#         weights_cdf_s = (1 - torch.cumsum(weights_cdf[..., :-1], dim=-1))
#         weights_cdf_s[weights_cdf_s < 0] = 0
#         weights_cdf[..., 1:] = weights_cdf[..., 1:] * weights_cdf_s
#         return weights_cdf

    
#     @torch.no_grad()
#     def render_invdepth_thresh(self, n, h, w, K, E):

#         n_f = torch.reshape(n, (-1,))
#         h_f = torch.reshape(h, (-1,))
#         w_f = torch.reshape(w, (-1,))

#         K_f = torch.reshape(K, (-1, 3, 3))
#         E_f = torch.reshape(E, (-1, 4, 4))

#         invdepth_thresh_f = torch.zeros(h_f.shape, device='cpu')

#         for a in tqdm(range(0, len(n_f), self.n_rays)):
#             b = min(len(n_f), a+self.n_rays)

#             n_fb = n_f[a:b]
#             h_fb = h_f[a:b]
#             w_fb = w_f[a:b]

#             K_fb = K_f[a:b]
#             E_fb = E_f[a:b]

#             color_bg = torch.ones(3, device='cuda') # [3], fixed white background

#             _, weights_fb, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

#             z_vals_fb = aux_outputs_fb['z_vals']

#             weights_thresh_start_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 - self.distribution_area / 2)
#             weights_thresh_mid_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5)
#             weights_thresh_end_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 + self.distribution_area / 2)

#             depth_start = torch.sum(weights_thresh_start_fb * z_vals_fb, dim=-1)
#             depth_mid = torch.sum(weights_thresh_mid_fb * z_vals_fb, dim=-1)
#             depth_end = torch.sum(weights_thresh_end_fb * z_vals_fb, dim=-1)

#             impulse_mask = torch.abs(depth_start - depth_end) > self.max_variance_pcd

#             invdepth_thresh_fb = 1 / depth_mid
#             invdepth_thresh_fb[impulse_mask] = 0
#             invdepth_thresh_fb[depth_mid == 0] = 0

#             invdepth_thresh_f[a:b] = invdepth_thresh_fb.cpu()

#         invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape).numpy()

#         # for i in range(self.rotate // 90):
#         #     invdepth_thresh = rot90(invdepth_thresh)

#         return invdepth_thresh
        

#     @torch.no_grad()
#     def extract_surface_geometry(self, n, h, w, K, E):

#         n_f = torch.reshape(n, (-1,))
#         h_f = torch.reshape(h, (-1,))
#         w_f = torch.reshape(w, (-1,))

#         K_f = torch.reshape(K, (-1, 3, 3))
#         E_f = torch.reshape(E, (-1, 4, 4))

#         points = torch.zeros((0, 3), device='cpu')
#         colors = torch.zeros((0, 3), device='cpu')
#         depth_variance = torch.zeros((0, 1), device='cpu')
#         # x_hashtable = torch.zeros((0, 40), device='cpu')

#         for a in tqdm(range(0, len(n_f), self.n_rays)):
#             b = min(len(n_f), a+self.n_rays)

#             n_fb = n_f[a:b].to('cuda')
#             h_fb = h_f[a:b].to('cuda')
#             w_fb = w_f[a:b].to('cuda')

#             K_fb = K_f[a:b].to('cuda')
#             E_fb = E_f[a:b].to('cuda')

#             color_bg = torch.ones(3, device='cuda') # [3], fixed white background

#             _, weights_fb, _, aux_outputs_fb = self.renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

#             xyzs_fb = aux_outputs_fb['xyzs']
#             rgbs_fb = aux_outputs_fb['rgbs']
#             z_vals_fb = aux_outputs_fb['z_vals']
#             # x_hashtable_fb = aux_outputs_fb['x_hashtable']

#             weights_thresh_start_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 - self.distribution_area / 2)
#             weights_thresh_mid_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5)
#             weights_thresh_end_fb = self.calculate_cumlative_weights_thresh(weights_fb, 0.5 + self.distribution_area / 2)

#             depth_start = torch.sum(weights_thresh_start_fb * z_vals_fb, dim=-1)
#             depth_end = torch.sum(weights_thresh_end_fb * z_vals_fb, dim=-1)
#             depth_variance_fb = torch.abs(depth_start - depth_end)[:, None, None].expand(-1, weights_fb.shape[1], 1)
            
#             impulse_mask = depth_variance_fb < self.max_variance_npy
#             mid_transmittance_mask = weights_thresh_mid_fb.to(bool)[..., None]
#             inside_inner_bound_mask = torch.linalg.norm(xyzs_fb, dim=-1, keepdim=True) < self.renderer.inner_bound
#             mask = (impulse_mask & mid_transmittance_mask & inside_inner_bound_mask)

#             points_b = xyzs_fb[mask.expand(*xyzs_fb.shape)].reshape(-1, 3)
#             colors_b = rgbs_fb[mask.expand(*rgbs_fb.shape)].reshape(-1, 3)
#             depth_variance_b = depth_variance_fb[mask].reshape(-1, 1)
#             # x_hashtable_b = x_hashtable_fb[mask.expand(*x_hashtable_fb.shape)].reshape(-1, 40)

#             points = torch.cat([points, points_b.cpu()], dim=0)
#             colors = torch.cat([colors, colors_b.cpu()], dim=0)
#             depth_variance = torch.cat([depth_variance, depth_variance_b.cpu()], dim=0)
#             # x_hashtable = torch.cat([x_hashtable, x_hashtable_b.cpu()], dim=0)

#         pointcloud = {}
#         pointcloud['points'] = points.numpy()
#         pointcloud['colors'] = colors.numpy()
#         pointcloud['depth_variance'] = depth_variance.numpy()
#         # pointcloud['x_hashtable'] = x_hashtable.numpy()

#         return pointcloud


@torch.no_grad()
def render_image(renderer, n, h, w, K, E, n_rays):

    # _f : flattened
    # _b : batched

    # renderer.eval()

    n_f = torch.reshape(n, (-1,))
    h_f = torch.reshape(h, (-1,))
    w_f = torch.reshape(w, (-1,))

    K_f = torch.reshape(K, (-1, 3, 3))
    E_f = torch.reshape(E, (-1, 4, 4))

    image_f = torch.zeros((*h_f.shape, 3), device='cpu')
    invdepth_f = torch.zeros(h_f.shape, device='cpu')

    for a in tqdm(range(0, len(n_f), n_rays)):
        b = min(len(n_f), a+n_rays)

        n_fb = n_f[a:b]
        h_fb = h_f[a:b]
        w_fb = w_f[a:b]

        K_fb = K_f[a:b]
        E_fb = E_f[a:b]

        color_bg = torch.ones(3, device='cuda') # [3], fixed white background

        image_fb, _, _, aux_outputs_fb = renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, bg_color=color_bg)

        image_f[a:b] = image_fb.detach().cpu()
        invdepth_f[a:b] = aux_outputs_fb['invdepth'].cpu()

    image = torch.reshape(image_f, (*h.shape, 3))
    invdepth = torch.reshape(invdepth_f, h.shape)

    return image, invdepth


@torch.no_grad()
def calculate_cumlative_weights_thresh(weights, thresh):
    weights_cdf = torch.cumsum(torch.clone(weights), dim=-1)
    weights_cdf[weights_cdf < thresh] = 0
    weights_cdf[weights_cdf >= thresh] = 1
    weights_cdf_s = (1 - torch.cumsum(weights_cdf[..., :-1], dim=-1))
    weights_cdf_s[weights_cdf_s < 0] = 0
    weights_cdf[..., 1:] = weights_cdf[..., 1:] * weights_cdf_s
    return weights_cdf


@torch.no_grad()
def calculate_dist_mask(weights, z_vals, dist_area, max_var):
    weights_thresh_start = calculate_cumlative_weights_thresh(weights, 0.5 - dist_area / 2)
    weights_thresh_mid = calculate_cumlative_weights_thresh(weights, 0.5)
    weights_thresh_end = calculate_cumlative_weights_thresh(weights, 0.5 + dist_area / 2)

    depth_start = torch.sum(weights_thresh_start * z_vals, dim=-1)
    depth_end = torch.sum(weights_thresh_end * z_vals, dim=-1)

    dist_mask = torch.abs(depth_start - depth_end) > max_var

    return dist_mask


@torch.no_grad()
def extract_surface_geometry_map(weights, z_vals, dist_area, max_var):
    dist_mask = calculate_dist_mask(weights, z_vals, dist_area=0.5, max_var=0.05)
    weights_thresh_mid = calculate_cumlative_weights_thresh(weights, 0.5)
    depth_median = torch.sum(weights_thresh_mid * z_vals, dim=-1)

    invdepth = 1 / depth_median
    invdepth[dist_mask] = 0
    invdepth[depth_median == 0] = 0
    return invdepth



# @torch.no_grad()
# def extract_surface_geometry_points(xyzs, rgbs, weights, z_vals, dist_area, max_var, max_radius):
#     dist_mask = calculate_dist_mask(weights, z_vals, dist_area, max_var)
#     median_mask = calculate_cumlative_weights_thresh(weights, 0.5).to(bool)[..., None]
#     inside_radius_mask = torch.linalg.norm(xyzs, dim=-1, keepdim=True) < max_radius

#     mask = (dist_mask & median_mask & inside_radius_mask)

#     points = xyzs[mask.expand(*xyzs.shape)].reshape(-1, 3)
#     colors = rgbs[mask.expand(*rgbs.shape)].reshape(-1, 3)

#     return points, colors


@torch.no_grad()
def extract_surface_geometry_points(xyzs, colors, weights, z_vals, dist_area, max_var):
    dist_mask = calculate_dist_mask(weights, z_vals, dist_area, max_var).to(bool)[..., None, None]
    median_mask = calculate_cumlative_weights_thresh(weights, 0.5).to(bool)[..., None]

    mask = (dist_mask & median_mask)

    points = xyzs[mask.expand(*xyzs.shape)].reshape(-1, 3)
    colors_ = colors[mask.expand(*colors.shape)].reshape(-1, 3)

    return points, colors_


@torch.no_grad()
def render_invdepth_thresh(renderer, n, h, w, K, E, n_rays):

    n_f = torch.reshape(n, (-1,))
    h_f = torch.reshape(h, (-1,))
    w_f = torch.reshape(w, (-1,))

    K_f = torch.reshape(K, (-1, 3, 3))
    E_f = torch.reshape(E, (-1, 4, 4))

    invdepth_thresh_f = torch.zeros(h_f.shape, device='cpu')

    for a in tqdm(range(0, len(n_f), n_rays)):
        b = min(len(n_f), a+n_rays)

        n_fb = n_f[a:b]
        h_fb = h_f[a:b]
        w_fb = w_f[a:b]

        K_fb = K_f[a:b]
        E_fb = E_f[a:b]

        _, weights_fb, _, aux_outputs_fb = renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb)
        z_vals_fb = aux_outputs_fb['z_vals']

        invdepth_thresh_fb = extract_surface_geometry_map(
                                weights_fb, z_vals_fb, dist_area=0.5, max_var=0.05)
        invdepth_thresh_f[a:b] = invdepth_thresh_fb.cpu()

    invdepth_thresh = torch.reshape(invdepth_thresh_f, h.shape)

    return invdepth_thresh


@torch.no_grad()
def generate_pointcloud(renderer, n, h, w, K, E, n_rays, max_varience, distribution_area):

    n_f = torch.reshape(n, (-1,))
    h_f = torch.reshape(h, (-1,))
    w_f = torch.reshape(w, (-1,))

    K_f = torch.reshape(K, (-1, 3, 3))
    E_f = torch.reshape(E, (-1, 4, 4))

    points = torch.zeros((0, 3), device='cpu')
    colors = torch.zeros((0, 3), device='cpu')

    for a in tqdm(range(0, len(n_f), n_rays)):
        b = min(len(n_f), a+n_rays)

        n_fb = n_f[a:b].to('cuda')
        h_fb = h_f[a:b].to('cuda')
        w_fb = w_f[a:b].to('cuda')

        K_fb = K_f[a:b].to('cuda')
        E_fb = E_f[a:b].to('cuda')

        _, weights_fb, _, aux_outputs_fb = renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb)
        xyzs_fb = aux_outputs_fb['xyzs']
        colors_fb = aux_outputs_fb['colors']
        z_vals_fb = aux_outputs_fb['z_vals']

        points_b, colors_b = extract_surface_geometry_points(
                                xyzs_fb, colors_fb, weights_fb, z_vals_fb,
                                distribution_area, max_varience)

        points = torch.cat([points, points_b.cpu()], dim=0)
        colors = torch.cat([colors, colors_b.cpu()], dim=0)

    pointcloud = {}
    pointcloud['points'] = points
    pointcloud['colors'] = colors
    return pointcloud


class ImageInference(object):
    def __init__(self, renderer, dataloader, n_rays, image_num):
        self.renderer = renderer
        self.dataloader = dataloader

        self.n_rays = n_rays
        self.image_num = image_num

    def __call__(self, image_num=None):
        image_num = self.image_num if image_num is None else image_num 
        n, h, w, K, E, _, _, _ = self.dataloader.get_image_batch(image_num, device='cuda')
        return render_image(self.renderer, n, h, w, K, E, self.n_rays)


class InvdepthThreshInference(object):
    def __init__(self, renderer, dataloader, n_rays, image_num):
        self.renderer = renderer
        self.dataloader = dataloader

        self.n_rays = n_rays
        self.image_num = image_num

    def __call__(self, image_num=None):
        image_num = self.image_num if image_num is None else image_num 
        n, h, w, K, E, _, _, _ = self.dataloader.get_image_batch(image_num, device='cuda')
        return render_invdepth_thresh(self.renderer, n, h, w, K, E, self.n_rays)


class PointcloudInference(object):
    def __init__(self,
        renderer,
        dataloader,

        max_variance,
        distribution_area,

        n_rays,
        cams,
        freq,
        side_margin,):

        self.renderer = renderer
        self.dataloader = dataloader

        self.max_variance = max_variance
        self.distribution_area = distribution_area

        self.n_rays = n_rays
        self.cams = cams
        self.freq = freq
        self.side_margin = side_margin

    def __call__(self):
        n, h, w, K, E, _, _, _ = self.dataloader.get_pointcloud_batch(
            self.cams, self.freq, self.side_margin, device='cpu')
        return generate_pointcloud(
            self.renderer,
            n, h, w, K, E,
            self.n_rays,
            self.max_variance,
            self.distribution_area)