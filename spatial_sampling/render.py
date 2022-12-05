import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit 
import math as m

from torch.utils.cpp_extension import load

from tqdm import tqdm

from nets import NeRFNetwork, NeRFCoordinateWrapper
from render import get_rays, get_uniform_z_vals, get_sample_points


fill = load(name="fill", sources=["./spatial_sampling/extensions/fill.cpp", "./spatial_sampling/extensions/fill_kernel.cu"])


@torch.no_grad()
def generate_dummy_weights(n_rays, n_samples):
    ray_transmittance = torch.rand([n_rays], device="cuda")

    weights = torch.rand([n_rays, n_samples], device="cuda")
    weights = weights / torch.sum(weights, dim=-1)[:, None] * ray_transmittance[:, None]

    return weights


# @torch.jit.script
# def fill(tensor, mask):
#     for i in range(1, tensor.shape[1]):
#         tensor[:, i][~mask[:, i]] = tensor[:, i-1][~mask[:, i]]


@torch.no_grad()
def test_allocate_old():
    total_samples = int(16384 * 64)
    n_rays, n_samples = 16384, 512
    steps = 256
    # weights = torch.rand([n_rays, n_samples], device="cuda") / n_samples
    weights = generate_dummy_weights(n_rays, n_samples)
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
    mask = u < cdf[:, -1, None]
    # mask = u >= cdf[:, -1, None]
    # print(mask[1], mask.shape)

    inds[~mask] = 0

    cdf_below = torch.gather(input=cdf_zero, dim=1, index=inds)
    cdf_above = torch.gather(input=cdf, dim=1, index=inds)
    t = (u - cdf_below) / (cdf_above - cdf_below)

    min_distance = 1
    z_vals_zero = torch.cat([z_vals.new_full((z_vals.shape[0], 1), fill_value=min_distance), cdf], dim=-1)[:, :-1]  # [N_rays, N_samples]
    z_vals_below = torch.gather(input=z_vals_zero, dim=1, index=inds)
    z_vals_above = torch.gather(input=z_vals, dim=1, index=inds)
    z_vals_im = z_vals_above * t + z_vals_below * (1-t)

    t[~mask] = 0
    z_vals_im[~mask] = 0

    # print(16384 * 64, z_vals_im[~mask].shape)
    
    total_samples_temp = torch.sum(mask)
    amount_culled = total_samples_temp - total_samples
    # z = torch.randperm(total_samples_temp)
    # print(amount_culled.item())
    # exit()
    # mask2 = torch.randperm(total_samples_temp.item())[amount_culled.item():]
    mask2 = torch.randperm(total_samples_temp, device="cuda")[:amount_culled]
    # print(mask[mask].shape)
    # mask_ = torch.clone(mask)
    # mask[mask][mask2] = False
    # print(mask)
    # exit()

    mask_flat = mask.reshape(-1)
    a = torch.arange(0, len(mask_flat), device="cuda")[mask_flat][mask2]
    b = torch.clone(mask_flat)
    b[a] = False
    c = b.reshape(*mask.shape)
    # print(b.sum(), b.shape, mask_flat.shape, mask.shape)
    # print(b)
    # print(b[a].sum(), b[a].shape, len(mask2))

    # print(c)



    # print(torch.sum(mask), mask.shape)

    # print(z_vals_im.shape)


@torch.no_grad()
def test_allocate():
    total_samples = int(16384 * 64)
    n_rays, n_samples = 16384, 512
    steps = 256
    # weights = torch.rand([n_rays, n_samples], device="cuda") / n_samples
    weights = generate_dummy_weights(n_rays, n_samples)
    z_vals_log, z_vals = get_uniform_z_vals([n_samples], [0.1, 1], n_rays)
    # print(torch.amin(z_vals_log))
    sample_pdf_3d(weights, steps, total_samples, z_vals_log, [0.1, 1])


@torch.no_grad()
def normalize_pdf(pdf, x):
    n_rays, n_samples = pdf.shape
    pdf_norm = pdf * (1-x) + x / n_samples
    return pdf_norm


@torch.no_grad()
def sample_pdf_3d(pdf, steps, total_samples, z_vals, z_bounds):
    n_rays, n_samples = pdf.shape

    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, n_samples] | 0 <= cdf <= 1

    u = torch.linspace(0, 1-1/steps, steps, device="cuda")  # [N_rays, steps]
    u = u + torch.rand([n_rays, steps], device="cuda")/steps  # [N_rays, N_samples]

    mask1 = u < cdf[:, -1, None]
    mask2 = torch.randperm(mask1.sum(), device="cuda")[:total_samples]
    a = torch.arange(0, len(mask1.view(-1)), device="cuda")[mask1.view(-1)][mask2]
    mask = torch.zeros(mask1.view(-1).shape, dtype=bool, device="cuda")
    mask[a] = True
    mask = mask.reshape(*mask1.shape)

    # print(u.shape, mask.shape)
    # exit()
    fill.fill(u, mask, u.shape[1])

    # print(u)


    # for i in range(1, u.shape[1]):
    #     u[:, i][~mask[:, i]] = u[:, i-1][~mask[:, i]]

    inds = torch.searchsorted(cdf, u)  # [N_rays, steps]

    # print(u)
    # print(u[0])
    # print(inds[0])
    # print(mask[0])

    # exit()

    # for i in range(1, inds.shape[1]):
    #     u[:, i][~mask[:, i]] = u[:, i-1][~mask[:, i]]
    #     inds[:, i][~mask[:, i]] = inds[:, i-1][~mask[:, i]]

    # m = u.new_zeros(u.shape)
    # # i = torch.arange(0, u.shape[1], device="cuda").expand(n_rays, -1)[mask]
    # # j = torch.arange(0, u.shape[0], device="cuda").expand(-1, n_samples)[mask]
    # i = mask.nonzero()
    # m_f = m.reshape(-1)
    # i_f = i[:, 0] * m.shape[1] + i[:, 1]  # does not repeat

    # i_f = mask.reshape(-1).nonzero()[:, 0]

    # # w = torch.arange(0, m_f.shape[0], device="cuda")  # does not repeat

    # w_f = (~mask).reshape(-1).nonzero()[:, 0]

    # x_f = torch.searchsorted(i_f, w_f)

    # print(x_f)

    # exit()

    # m[~mask] = i_f[x_f]

    # x = torch.searchsorted(w, i_f)
    # x_f = torch.searchsorted(i_f, w[~mask.view(-1)])
    # x_f = torch.searchsorted(i_f, w_f)
    # x = x_f.reshape(n_rays, u.shape[1])
    # print(x_f)
    # exit()
    # print(x_f.shape)
    # exit()
    # o = w[mask.view(-1)]
    # o = i_f.clone()
    # o_ = torch.cat([o, w[-1, None]], dim=0)
    # o_ = torch.cat([w[0, None], o], dim=0)
    # p_f = o_[x_f]
    # p = p_f.reshape(n_rays, u.shape[1])
    # p_d = torch.remainder(p, u.shape[1])
    # print(inds)
    # print(x[0])
    # print(p_d.shape)
    # print(p[0])
    # print(x[0])
    # print(o.shape)
    # exit()

    cdf_zero = torch.cat([cdf.new_full((cdf.shape[0], 1), fill_value=z_bounds[0]), cdf], dim=-1)  # [N_rays, N_samples+1]
    cdf_zero_1 = torch.cat([cdf, cdf.new_full((cdf.shape[0], 1), fill_value=z_bounds[-1])], dim=-1)  # [N_rays, N_samples+1]

    cdf_below = torch.gather(input=cdf_zero, dim=1, index=inds)
    cdf_above = torch.gather(input=cdf_zero_1, dim=1, index=inds)
    t = (u - cdf_below) / (cdf_above - cdf_below)

    z_vals_zero = torch.cat([z_vals.new_full((z_vals.shape[0], 1), fill_value=z_bounds[0]), z_vals], dim=-1)  # [N_rays, N_samples+1]
    z_vals_zero_1 = torch.cat([z_vals, z_vals.new_full((z_vals.shape[0], 1), fill_value=z_bounds[-1]), ], dim=-1)  # [N_rays, N_samples+1]

    z_vals_below = torch.gather(input=z_vals_zero, dim=1, index=inds)
    z_vals_above = torch.gather(input=z_vals_zero_1, dim=1, index=inds)
    z_vals_im = z_vals_above * t + z_vals_below * (1-t)

    return z_vals_im, mask


@torch.no_grad()
def efficient_sampling_3d(
    model,
    rays_o,
    rays_d,
    steps_firstpass,
    steps_importance,
    total_samples,
    z_bounds):

    n_rays, n_samples = rays_o.shape[:2]
    z_vals_log, z_vals = get_uniform_z_vals(steps_firstpass, z_bounds, n_rays)

    xyzs, _ = get_sample_points(rays_o, rays_d, z_vals)

    sigmas = model.density(xyzs)

    deltas = z_vals_log.new_zeros(sigmas.shape)  # [N_rays, N_samples]
    deltas[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

    alphas = 1 - torch.exp(-sigmas * deltas)  # [N_rays, N_samples]
    alphas_shift = torch.cat([alphas.new_zeros((alphas.shape[0], 1)), alphas], dim=-1)[:, :-1]  # [N_rays, N_samples]
    weights = alphas * torch.cumprod(1 - alphas_shift, dim=-1)  # [N_rays, N_samples]

    pdf = weights.clone()
    pdf_norm = normalize_pdf(pdf, 0.5)

    z_vals_log_fine, mask = sample_pdf_3d(pdf_norm, steps_importance, total_samples, z_vals_log, z_bounds)
    z_vals_fine = torch.pow(10, z_vals_log_fine)

    return z_vals_log_fine, z_vals_fine, mask


def render_rays_log(sigma, color, z_vals, z_vals_log, z_vals_log_min):
    n_rays, n_samples = z_vals.shape[:2]

    delta = z_vals_log.new_zeros([n_rays, n_samples])  # [n_rays, n_samples]
    delta[:, 0] = z_vals_log[:, 0] - z_vals_log_min
    delta[:, 1:] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [n_rays, n_samples]
    alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [n_rays, n_samples]
    weight = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [n_rays, n_samples]

    pixel = torch.sum(weight[..., None] * color, -2)  # [n_rays, 3]
    invdepth = torch.sum(weight / z_vals, -1)  # [n_rays]

    return pixel, invdepth, weight


def render_nerf(
    model,
    n,
    h,
    w,
    K,
    E,
    steps_firstpass,
    steps_importance,
    total_samples,
    z_bounds,
    bg_color):

    rays_o, rays_d = get_rays(h, w, K, E)

    z_vals_log, z_vals, mask = efficient_sampling_3d(
        model, rays_o, rays_d, steps_firstpass, steps_importance, total_samples, z_bounds)

    xyzs, dirs = get_sample_points(rays_o, rays_d, z_vals)

    n_expand = n[:, None].expand(-1, z_vals.shape[-1])

    sigma_, color_ = model(xyzs[mask], dirs[mask], n_expand[mask])

    sigma = sigma_.new_zeros(mask.shape)
    sigma[mask] = sigma_
    color = color_.new_zeros(xyzs.shape)
    color[mask] = color_

    pixel, invdepth, weight = render_rays_log(sigma, color, z_vals, z_vals_log, 0)
    if bg_color is not None:
        pixel = pixel + (1 - torch.sum(weight, dim=-1)[..., None]) * bg_color

    z_vals_log_norm = (z_vals_log - m.log10(z_bounds[0])) / (m.log10(z_bounds[-1]) - m.log10(z_bounds[0]))

    aux_outputs = {}
    aux_outputs['invdepth'] = invdepth.detach()
    aux_outputs['z_vals'] = z_vals.detach()
    aux_outputs['z_vals_log'] = z_vals_log.detach()

    return pixel, weight, z_vals_log_norm, aux_outputs


class Render(object):
    def __init__(self,
        models,
        steps_firstpass,
        steps_importance,
        total_samples,
        z_bounds,
        ):
        
        self.models = models
        self.steps_firstpass = steps_firstpass
        self.steps_importance = steps_importance
        self.total_samples = total_samples
        self.z_bounds = z_bounds

    def render(self, n, h, w, K, E, bg_color=None):
        return render_nerf(
            self.models, 
            n, h, w, K, E,
            steps_firstpass=self.steps_firstpass,
            steps_importance=self.steps_importance,
            total_samples=self.total_samples,
            z_bounds=self.z_bounds,
            bg_color=bg_color)



if __name__ == "__main__":
    for i in tqdm(range(10000)):
        test_allocate()
    # test_allocate()