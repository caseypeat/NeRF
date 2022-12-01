import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m

from tqdm import tqdm

from nets import NeRFNetwork, NeRFCoordinateWrapper
from render import get_rays, get_uniform_z_vals, get_sample_points


@torch.no_grad()
def test_allocate():
    total_samples = 8192
    n_rays, n_samples = 16384, 512
    steps = 64
    weights = torch.rand([n_rays, n_samples], device="cuda") / n_samples
    z_vals_log, z_vals = get_uniform_z_vals([n_samples], [1, 10], n_rays)

    # weights_f = weights.reshape[-1]

    cdf = torch.cumsum(weights, dim=-1)  # [n_rays, n_samples] | 0 <= cdf <= 1

    u = torch.linspace(0, 1-1/steps, steps, device="cuda")  # [N_rays, steps]
    u = u + torch.rand([n_rays, steps], device="cuda")/steps  # [N_rays, N_samples]
    # u = u[None, :].expand(n_rays, -1)

    # print(u.shape, cdf.shape)

    inds = torch.searchsorted(cdf, u)  # [N_rays, steps]

    cdf_zero = torch.cat([cdf.new_zeros((cdf.shape[0], 1)), cdf], dim=-1)[:, :-1]  # [N_rays, N_samples]

    # mask = u >= cdf[:, -1, None].expand(512, 64)
    # mask = u < cdf[:, -1, None]
    mask = u >= cdf[:, -1, None]
    # print(mask[1], mask.shape)

    inds[mask] = 0

    cdf_below = torch.gather(input=cdf_zero, dim=1, index=inds)
    cdf_above = torch.gather(input=cdf, dim=1, index=inds)
    t = (u - cdf_below) / (cdf_above - cdf_below)

    min_distance = 1
    z_vals_zero = torch.cat([z_vals.new_full((z_vals.shape[0], 1), fill_value=min_distance), cdf], dim=-1)[:, :-1]  # [N_rays, N_samples]
    z_vals_below = torch.gather(input=z_vals_zero, dim=1, index=inds)
    z_vals_above = torch.gather(input=z_vals, dim=1, index=inds)
    z_vals_im = z_vals_above * t + z_vals_below * (1-t)

    t[mask] = 0
    z_vals_im[mask] = 0

    # print(z_vals_im[~mask], z_vals_im[~mask].shape)

    # print(z_vals_im.shape)





@torch.no_grad()
def efficient_sampling_3d(
    model,
    rays_o,
    rays_d,
    steps_firstpass,
    z_bounds,
    steps_importance,
    alpha_importance):

    total_samples = int(16384 * 128)

    n_rays, n_samples = rays_o.shape[:2]
    z_vals_log, z_vals = get_uniform_z_vals(steps_firstpass, z_bounds, n_rays)

    xyzs, _ = get_sample_points(rays_o, rays_d, z_vals)

    sigmas = model.density(xyzs)

    deltas = z_vals_log.new_zeros(sigmas.shape)  # [N_rays, N_samples]
    deltas[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

    alphas = 1 - torch.exp(-sigmas * deltas)  # [N_rays, N_samples]
    alphas_shift = torch.cat([alphas.new_zeros((alphas.shape[0], 1)), alphas], dim=-1)[:, :-1]  # [N_rays, N_samples]
    weights = alphas * torch.cumprod(1 - alphas_shift, dim=-1)  # [N_rays, N_samples]

    # step one: allocate sample on cumlitive ray weight
    weight_cum = torch.sum(weights, dim=-1)  # [N_rays,]
    samples_per_ray = total_samples * weight_cum / torch.sum(weight_cum)  # [N_rays,]

    # step two: allocate sample allocations on density




def render_nerf(
    model,
    n,
    h,
    w,
    K,
    E,
    steps_firstpass,
    z_bounds,
    steps_importance,
    bg_color):

    rays_o, rays_d = get_rays(h, w, K, E)


if __name__ == "__main__":
    for i in tqdm(range(10000)):
        test_allocate()