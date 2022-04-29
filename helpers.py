import numpy as np
import torch
import math as m

from torch import nn
from torch.nn import functional as F

from tqdm import tqdm

## Losses and Metrics
def psnr(rgb_, rgb):
    mse = F.mse_loss(rgb_, rgb)
    return -10.0 * m.log10(mse)

def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse < 1e-5:
        mse = 1e-5
    return -10.0 * m.log10(mse)

def criterion_dist(weights, z_vals):
    # z_vals_s = (z_vals + torch.min(z_vals)) / (torch.max(z_vals) - torch.min(z_vals))
    w = torch.bmm(weights[:, :, None], weights[:, None, :])
    s = torch.abs(z_vals[:, :, None] - z_vals[:, None, :])
    loss = w * s
    loss = torch.mean(torch.sum(loss, dim=[1, 2]))
    return loss

def criterion_rgb(rgb_, rgb):
    loss = F.huber_loss(rgb_, rgb, delta=0.1)
    # loss = F.mse_loss(rgb_, rgb)
    return loss


## Calculate bounds of scene
def calculate_bounds(images, depths, intrinsics, extrinsics):

    n = np.arange(images.shape[0])
    h = np.arange(images.shape[1])
    w = np.arange(images.shape[2])

    n, h, w = np.meshgrid(n, h, w, indexing='ij')

    mask = images[..., 3].astype(bool)
    n, h, w = n[mask], h[mask], w[mask]

    K = intrinsics[n]
    E = extrinsics[n]

    rays_o, rays_d = get_rays_np(h, w, K, E)

    xyz = rays_o + rays_d * depths[mask][..., None]
    xyz = np.concatenate([xyz, extrinsics[..., :3, -1]])

    xyz_min = np.amin(xyz, axis=tuple(np.arange(len(xyz.shape[:-1]))))
    xyz_max = np.amax(xyz, axis=tuple(np.arange(len(xyz.shape[:-1]))))

    return xyz_min, xyz_max

def calculate_bounds_sphere(images, depths, intrinsics, extrinsics):

    n = np.arange(images.shape[0])
    h = np.arange(images.shape[1])
    w = np.arange(images.shape[2])

    n, h, w = np.meshgrid(n, h, w, indexing='ij')

    mask = images[..., 3].astype(bool)
    n, h, w = n[mask], h[mask], w[mask]

    K = intrinsics[n]
    E = extrinsics[n]

    rays_o, rays_d = get_rays_np(h, w, K, E)

    xyz = rays_o + rays_d * depths[mask][..., None]
    xyz = np.concatenate([xyz, extrinsics[..., :3, -1]])

    xyz_min = np.amin(np.linalg.norm(xyz, axis=-1))
    xyz_max = np.amax(np.linalg.norm(xyz, axis=-1))

    return xyz_min, xyz_max


## Transforms
def mipnerf360_scale(xyzs, bound):
    d = torch.linalg.norm(xyzs, dim=-1)[..., None].expand(-1, -1, 3)
    s_xyzs = torch.clone(xyzs)
    s_xyzs[d > 1] = s_xyzs[d > 1] * ((bound - (bound - 1) / d[d > 1]) / d[d > 1])
    return s_xyzs


## Rendering Pipeline
def get_rays_np(h, w, K, E):
    """ Ray generation for Oliver's calibration format """
    dirs = np.stack([w+0.5, h+0.5, np.ones_like(w)], -1)  # [N_rays, 3]
    dirs = np.linalg.inv(K) @ dirs[..., None]  # [N_rays, 3, 1]

    rays_d = E[..., :3, :3] @ dirs  # [N_rays, 3, 1]
    rays_d = rays_d[..., 0]  # [N_rays, 3]

    rays_o = E[..., :3, -1]

    return rays_o, rays_d


def get_rays(h, w, K, E):
    """ Ray generation for Oliver's calibration format """
    dirs = torch.stack([w+0.5, h+0.5, torch.ones_like(w)], -1)  # [N_rays, 3]
    dirs = torch.inverse(K) @ dirs[:, :, None]  # [N_rays, 3, 1]

    rays_d = E[:, :3, :3] @ dirs  # [N_rays, 3, 1]
    rays_d = torch.squeeze(rays_d, dim=2)  # [N_rays, 3]

    rays_o = E[:, :3, -1].expand(rays_d.shape[0], -1)

    return rays_o, rays_d


def get_z_vals_log(begin, end, N_rays, N_samples, device):
    """ Inversly proportional to depth sample spacing """
    z_vals = torch.linspace(0, 1-1/N_samples, N_samples, device=device)
    z_vals = z_vals[None, :].expand(N_rays, -1)
    z_vals = z_vals + torch.rand([N_rays, N_samples], device=device)/N_samples  # [N_rays, N_samples]

    z_vals = torch.exp((m.log(end)-m.log(begin))*z_vals + m.log(begin))

    return z_vals


def get_sample_points(rays_o, rays_d, z_vals):
    N_rays, N_samples = z_vals.shape

    query_points = rays_d.new_zeros((N_rays, N_samples, 3))  # [N_rays, N_samples, 3]
    query_points = rays_o[:, None, :].expand(-1, N_samples, -1) + rays_d[:, None, :].expand(-1, N_samples, -1) * z_vals[..., None]  # [N_rays, N_samples, 3]

    norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    query_dirs = rays_d / norm
    query_dirs = query_dirs[:, None, :].expand(-1, N_samples, -1)

    return query_points, query_dirs


def render_rays_log(sigmas, rgbs, z_vals, z_vals_log):
    N_rays, N_samples = z_vals.shape[:2]

    delta = z_vals_log.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
    delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

    alpha = 1 - torch.exp(-sigmas * delta)  # [N_rays, N_samples]

    alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

    rgb = torch.sum(weights[..., None] * rgbs, -2)  # [N_rays, 3]
    invdepth = torch.sum(weights / z_vals, -1)  # [N_rays]

    return rgb, invdepth, weights


def sample_pdf(z_vals, weights, N_importance):

    N_rays, N_samples = z_vals.shape

    pdf = weights  # [N_rays, N_samples-1]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, N_samples-1]

    z_vals_mid = (z_vals[:, :-1] + z_vals[:, 1:]) / 2  # [N_rays, N_samples]

    # Take uniform samples
    u = torch.linspace(0, 1-1/N_importance, N_importance, device=weights.device)  # [N_rays, N_importance]
    u = u[None, :].expand(N_rays, -1)
    u = u + torch.rand([N_rays, N_importance], device=weights.device)/N_importance  # [N_rays, N_samples]
    u = u * (cdf[:, -2, None] - cdf[:, 1, None] - 2e-5)
    u = u + cdf[:, 1, None] + 1e-5

    inds = torch.searchsorted(cdf, u, right=True)  # [N_rays, N_importance]

    cdf_below = torch.gather(input=cdf, dim=1, index=inds-1)
    cdf_above = torch.gather(input=cdf, dim=1, index=inds)
    t = (u - cdf_below) / (cdf_above - cdf_below)

    z_vals_mid_below = torch.gather(input=z_vals_mid, dim=1, index=inds-1)
    z_vals_mid_above = torch.gather(input=z_vals_mid, dim=1, index=inds)
    z_vals_im = z_vals_mid_below + (z_vals_mid_above - z_vals_mid_below) * t

    z_vals_fine = z_vals_im

    return z_vals_fine


## Misc




## get visible points in a scene (probaly rewrite)
@torch.no_grad()
def get_valid_positions(N, H, W, K, E, res):

    mask_full = torch.zeros((res, res, res), dtype=bool, device='cuda')

    for i in tqdm(range(res)):
        d = torch.linspace(-1, 1, res, device='cuda')
        D = torch.stack(torch.meshgrid(d[i], d, d, indexing='ij'), dim=-1)
        dist = torch.linalg.norm(D, dim=-1)[:, :, :, None].expand(-1, -1, -1, 3)
        mask = torch.zeros(dist.shape, dtype=bool, device='cuda')
        mask[dist < 1] = True

        # also mask out parts outside camera coverage
        rays_d = D - E[:, None, None, :3, -1]
        dirs_ = torch.inverse(E[:, None, None, :3, :3]) @ rays_d[..., None]
        dirs_ = K[:, None, None, ...] @ dirs_
        dirs = dirs_ / dirs_[:, :, :, 2, None, :]
        mask_dirs = torch.zeros((N, res, res), dtype=int, device='cuda')
        mask_dirs[((dirs[:, :, :, 0, 0] > 0) & (dirs[:, :, :, 0, 0] < H) & (dirs[:, :, :, 1, 0] > 0) & (dirs[:, :, :, 1, 0] < W) & (dirs_[:, :, :, 2, 0] > 0))] = 1
        mask_dirs = torch.sum(mask_dirs, dim=0)
        mask_dirs[mask_dirs > 0] = 1
        mask_dirs = mask_dirs.to(bool)
        mask_dirs = mask_dirs[None, :, :, None].expand(-1, -1, -1, 3)
        mask = torch.logical_and(mask, mask_dirs)

        mask_full[i, :, :] = mask[..., 0]

    return mask_full