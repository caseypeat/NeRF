import numpy as np
import torch
import math as m

from torch import nn
from torch.nn import functional as F


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse < 1e-5:
        mse = 1e-5
    return -10.0 * m.log10(mse)


def get_rays_np(h, w, K, E):
    """ Ray generation for Oliver's calibration format """
    dirs = np.stack([w+0.5, h+0.5, np.ones_like(w)], -1)  # [N_rays, 3]
    dirs = np.linalg.inv(K) @ dirs[..., None]  # [N_rays, 3, 1]

    rays_d = E[..., :3, :3] @ dirs  # [N_rays, 3, 1]
    rays_d = rays_d[..., 0]  # [N_rays, 3]

    rays_o = E[..., :3, -1]

    return rays_o, rays_d


def calculate_bounds(images, depths, intrinsics, extrinsics):

    n = np.arange(images.shape[0])
    h = np.arange(images.shape[1])
    w = np.arange(images.shape[2])

    n, h, w = np.meshgrid(n, h, w, indexing='ij')

    K = intrinsics[n]
    E = extrinsics[n]

    rays_o, rays_d = get_rays_np(h, w, K, E)

    xyz = rays_o + rays_d * depths[..., None]

    xyz[xyz == np.inf] = 0
    xyz[xyz == -np.inf] = 0

    xyz_min = np.amin(xyz, axis=tuple(np.arange(len(xyz.shape[:-1]))))
    xyz_max = np.amax(xyz, axis=tuple(np.arange(len(xyz.shape[:-1]))))

    return xyz_min, xyz_max

def calculate_bounds_sphere(images, depths, intrinsics, extrinsics):

    n = np.arange(images.shape[0])
    h = np.arange(images.shape[1])
    w = np.arange(images.shape[2])

    n, h, w = np.meshgrid(n, h, w, indexing='ij')

    K = intrinsics[n]
    E = extrinsics[n]

    rays_o, rays_d = get_rays_np(h, w, K, E)

    xyz = rays_o + rays_d * depths[..., None]

    xyz[xyz == np.inf] = 0
    xyz[xyz == -np.inf] = 0

    xyz_min = np.amin(np.linalg.norm(xyz, axis=-1))
    xyz_max = np.amax(np.linalg.norm(xyz, axis=-1))

    return xyz_min, xyz_max


def get_rays(h, w, K, E):
    """ Ray generation for Oliver's calibration format """
    dirs = torch.stack([w+0.5, h+0.5, torch.ones_like(w)], -1)  # [N_rays, 3]
    dirs = torch.inverse(K) @ dirs[:, :, None]  # [N_rays, 3, 1]

    rays_d = E[:, :3, :3] @ dirs  # [N_rays, 3, 1]
    rays_d = torch.squeeze(rays_d, dim=2)  # [N_rays, 3]

    rays_o = E[:, :3, -1]

    return rays_o, rays_d


def mipnerf360_coordinate_transform(xyz):
    xyz[xyz > 1] = 2 - 1/xyz[xyz > 1]
    xyz[xyz < -1] = -2 - 1/xyz[xyz < -1]
    return xyz


def get_z_vals_log(begin, end, N_rays, N_samples, device):
    """ Inversly proportional to depth sample spacing """
    z_vals = torch.linspace(0, 1-1/N_samples, N_samples, device=device)
    z_vals = z_vals[None, :].expand(N_rays, -1)
    z_vals = z_vals + torch.rand([N_rays, N_samples], device=device)/N_samples  # [N_rays, N_samples]

    z_vals = torch.exp((torch.log(end)-torch.log(begin))[..., None]*z_vals + torch.log(begin)[..., None])

    return z_vals


def freq_embedding(x, L, sub_fundemental=0):
    gamma = x.new_zeros((*x.shape, L*2))

    for i in range(L):
        gamma[..., 2*i] = torch.sin(2**(i-sub_fundemental) * m.pi * x)
        gamma[..., 2*i+1] = torch.cos(2**(i-sub_fundemental) * m.pi * x)

    gamma = torch.flatten(gamma, start_dim=-2)

    return gamma


def get_sample_points(rays_o, rays_d, z_vals):
    N_rays, N_samples = z_vals.shape

    query_points = rays_d.new_zeros((N_rays, N_samples, 3))  # [N_rays, N_samples, 3]
    query_points = rays_o[:, None, :].expand(-1, N_samples, -1) + rays_d[:, None, :].expand(-1, N_samples, -1) * z_vals[..., None]  # [N_rays, N_samples, 3]

    norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    query_dirs = rays_d / norm
    query_dirs = query_dirs[:, None, :].expand(-1, N_samples, -1)

    return query_points, query_dirs


def get_sample_points_ipe(origin, direction, focal, z_vals, degree, sub_fundemental, degree_dir, sub_fundenmental_dir, ipe):
    """ get intergrated positional encoding """
    N, S = z_vals.shape

    radius = 2/m.sqrt(12) / focal[:, None]

    mean, cov_diag = cast_torch(origin, direction, radius, z_vals)
    if ipe:
        ipe_xyz = IPE_torch(mean, cov_diag, degree, sub_fundemental)
    else:
        ipe_xyz = PE2_torch(mean, cov_diag, degree, sub_fundemental)

    direction_norm = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
    pe_dir = PE_torch(direction_norm, degree_dir, sub_fundenmental_dir)
    pe_dir = pe_dir[:, None, :].expand(-1, S-1, -1)

    return mean, ipe_xyz, pe_dir


def render_rays_log(samples, z_vals):
    N_rays, N_samples = z_vals.shape[:2]

    color = torch.sigmoid(samples[..., :3])  # [N_rays, N_samples, 3]
    sigma = samples[..., 3]  # [N_rays, N_samples]

    delta = z_vals.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
    delta[:, :-1] = (z_vals[:, 1:] - z_vals[:, :-1]) / (z_vals[:, 1:]/2 + z_vals[:, :-1]/2)

    alpha = 1 - torch.exp(-F.relu(sigma) * delta)  # [N_rays, N_samples]

    alpha_shift = torch.cat([samples.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

    rgb = torch.sum(weights[..., None] * color, -2)  # [N_rays, 3]
    invdepth = torch.sum(weights / z_vals, -1)  # [N_rays]

    return rgb, invdepth, weights, sigma, color


def sample_pdf(z_vals, weights, N_importance, alpha=0.0001):

    N_rays, N_samples = z_vals.shape

    weights = torch.cat([weights.new_zeros((N_rays, 1)), weights, weights.new_zeros((N_rays, 1))], dim=-1)
    weights = 1/2 * (torch.maximum(weights[..., :-2], weights[..., 1:-1]) + torch.maximum(weights[..., 1:-1], weights[..., 2:]))
    weights = weights + alpha
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, N_samples-1]
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

    z_vals_fine = torch.cat([z_vals, z_vals_im], dim=1)  # [N_rays, N_samples+N_importance]
    z_vals_fine, _ = torch.sort(z_vals_fine, dim=1)  # [N_rays, N_samples+N_importance]
    # z_vals_fine = z_vals_im

    return z_vals_fine


def theshold_sigma_render(sigma, z_vals, threshold):

    N_rays, N_samples = z_vals.shape[:2]
    
    sigma_c = torch.clone(sigma)
    sigma_c[sigma_c >= threshold] = 1e4
    sigma_c[sigma_c < threshold] = 0

    delta = z_vals.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
    delta[:, :-1] = (z_vals[:, 1:] - z_vals[:, :-1]) / (z_vals[:, 1:]/2 + z_vals[:, :-1]/2)

    alpha = 1 - torch.exp(-F.relu(sigma_c) * delta)  # [N_rays, N_samples]

    alpha_shift = torch.cat([sigma_c.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rasigma_c]
    weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

    invdepth = torch.sum(weights / z_vals, -1)  # [N_rays]

    return invdepth


## MIP NeRF
# Numpy (as written in paper: https://arxiv.org/pdf/2103.13415.pdf)
def cast(origin, direction, radius, t):
    t0, t1 = t[..., :-1], t[..., 1:]
    c, d = (t0 + t1)/2, (t1 - t0)/2
    t_mean = c + (2*c*d**2) / (3*c**2 + d**2)
    t_var = (d**2)/3 - (4/15) * ((d**4 * (12*c**2 - d**2)) / (3*c**2 + d**2)**2)
    r_var = radius**2 * ((c**2)/4 + (5/12) * d**2 - (4/15) * (d**4) / (3*c**2 + d**2))
    mean = direction[..., None, :] * t_mean[..., None]
    null_outer_diag = 1 - (direction**2) / np.sum(direction**2, axis=-1, keepdims=True)
    cov_diag = (t_var[..., None] * (direction**2)[..., None, :]
        + r_var[..., None] * null_outer_diag[..., None, :])
    return mean + origin[..., None, :], cov_diag

def IPE(mean, cov_diag, degree):
    y = np.concatenate([2**i * mean for i in range(degree)])
    w = np.concatenate([np.exp(-0.5 * 4**i * cov_diag) for i in range(degree)])
    return np.concatenate([np.sin(y) * w, np.cos(y) * w])

def PE(x, degree):
    y = np.concatenate([2**i * x for i in range(degree)])
    return np.concatenate([np.sin(y), np.cos(y)])


# Torch
def cast_torch(origin, direction, radius, t):
    t0, t1 = t[..., :-1], t[..., 1:]
    c, d = (t0 + t1)/2, (t1 - t0)/2
    t_mean = c + (2*c*d**2) / (3*c**2 + d**2)
    t_var = (d**2)/3 - (4/15) * ((d**4 * (12*c**2 - d**2)) / (3*c**2 + d**2)**2)
    r_var = radius**2 * ((c**2)/4 + (5/12) * d**2 - (4/15) * (d**4) / (3*c**2 + d**2))
    mean = direction[..., None, :] * t_mean[..., None]
    null_outer_diag = 1 - (direction**2) / torch.sum(direction**2, dim=-1, keepdims=True)
    cov_diag = (t_var[..., None] * (direction**2)[..., None, :] + r_var[..., None] * null_outer_diag[..., None, :])
    return mean + origin[..., None, :], cov_diag

def IPE_torch(mean, cov_diag, degree, sub_fundemental=0):
    o = sub_fundemental
    y = torch.cat([2**(i-o) * mean for i in range(degree)], dim=-1)
    w = torch.cat([torch.exp(-0.5 * 4**(i-o) * cov_diag) for i in range(degree)], dim=-1)
    return torch.cat([torch.sin(y) * w, torch.cos(y) * w], dim=-1)

def PE2_torch(mean, cov_diag, degree, sub_fundemental=0):
    o = sub_fundemental
    y = torch.cat([2**(i-o) * mean * m.pi for i in range(degree)], dim=-1)
    w = 1
    return torch.cat([torch.sin(y) * w, torch.cos(y) * w], dim=-1)


def PE_torch(x, degree, sub_fundemental=0):
    o = sub_fundemental
    y = torch.cat([2**(i-o) * x for i in range(degree)], dim=-1)
    return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)