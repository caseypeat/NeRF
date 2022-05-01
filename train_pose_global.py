import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d

from tqdm import tqdm

import helpers

from loaders.camera_geometry_loader import meta_camera_geometry_real
from utils.allign_pointclouds import allign_pointclouds

from nets import NeRFNetwork

from config import cfg


# https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py
def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def vec2skew_vec(v):
    """
    :param v:  (N, 3, ) torch tensor
    :return:   (N, 3, 3)
    """
    zero = torch.zeros((v.shape[0], 1), dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[:, 2:3],   v[:, 1:2]], dim=-1)  # (N, 3)
    skew_v1 = torch.cat([ v[:, 2:3],   zero,    -v[:, 0:1]], dim=-1)
    skew_v2 = torch.cat([-v[:, 1:2],   v[:, 0:1],   zero], dim=-1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (N, 3_, 3)
    return skew_v  # (N, 3, 3)


def Exp_vec(r):
    """so(3) vector to SO(3) matrix
    :param r: (N, 3, ) axis-angle, torch tensor
    :return:  (N, 3, 3)
    """
    skew_r = vec2skew_vec(r)  # (N, 3, 3)
    norm_r = torch.linalg.norm(r, dim=1) + 1e-15
    norm_r = norm_r[:, None, None]
    eye = torch.eye(3, dtype=torch.float32, device=r.device)[None, :, :].expand(r.shape[0], 3, 3)
    R = eye + ((torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r))
    return R



def sample_dist(dist, batch_size):
    pdf = dist / torch.sum(dist, dim=0, keepdim=True)
    cdf = torch.cumsum(pdf, dim=0)
    l = torch.linspace(0+1/cdf.shape[0]/2, 1-1/cdf.shape[0]/2, cdf.shape[0])
    u = torch.rand((batch_size,))
    u = u * (cdf[-2] - cdf[1] - 2e-5)
    u = u + cdf[1] + 1e-5
    inds = torch.searchsorted(cdf, u, right=True)
    cdf_below = torch.gather(input=cdf, dim=0, index=inds-1)
    cdf_above = torch.gather(input=cdf, dim=0, index=inds)
    t = (u - cdf_below) / (cdf_above - cdf_below)

    l_below = torch.gather(input=l, dim=0, index=inds-1)
    l_above = torch.gather(input=l, dim=0, index=inds)
    s = l_below + (l_above - l_below) * t
    return s


def sample_rt(r_dist, t_dist, batch_size):
    R = torch.zeros((batch_size, 3))
    R[:, 0] = sample_dist(r_dist[:, 0], batch_size) * 2 - 1
    R[:, 1] = sample_dist(r_dist[:, 1], batch_size) * 2 - 1
    R[:, 2] = sample_dist(r_dist[:, 2], batch_size) * 2 - 1

    T = torch.zeros((batch_size, 3))
    T[:, 0] = sample_dist(t_dist[:, 0], batch_size) * 4 - 2
    T[:, 1] = sample_dist(t_dist[:, 1], batch_size) * 4 - 2
    T[:, 2] = sample_dist(t_dist[:, 2], batch_size) * 4 - 2

    return R, T



if __name__ == '__main__':

    # dist = torch.arange(10)
    # s = sample_dist(dist, 10000)
    # hist, bounds = np.histogram(s.numpy())
    # print(hist)
    # print(bounds)
    # plt.plot(bounds[:-1], hist)
    # plt.show()
    # print(torch.mean(s))
    # exit()

    # R_test = torch.rand((2, 3))
    # T_test = torch.rand((2, 3))

    # a_ = Exp(R_test[0])
    # print(a_)
    # a = Exp_vec(R_test)
    # print(a[0])
    # print(a[1])


    # exit()

    thresh = 100

    N_a = 1176
    model_a = NeRFNetwork(
        # renderer
        intrinsics=torch.zeros((N_a, 3, 3)),
        extrinsics=torch.zeros((N_a, 4, 4)),

        # net
        N = N_a
    ).to('cuda')
    model_a.load_state_dict(torch.load('./logs/east_west/20220429_175050/model/6000.pth'))

    points_a = np.load('./logs/east_west/20220429_175050/pointcloud/6000.npy')
    points_a, sigmas_a = points_a[..., :3], points_a[..., 3]
    points_a = points_a[np.broadcast_to(sigmas_a[..., None], (sigmas_a.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas_a[sigmas_a > thresh][..., None]
    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(points_a)
    pcd_a.paint_uniform_color([1, 0, 0])
    
    N_b = 972
    model_b = NeRFNetwork(
        # renderer
        intrinsics=torch.zeros((N_b, 3, 3)),
        extrinsics=torch.zeros((N_b, 4, 4)),

        # net
        N = N_b
    ).to('cuda')
    model_b.load_state_dict(torch.load('./logs/east_west/20220429_183352/model/6000.pth'))

    points_b = np.load('./logs/east_west/20220429_183352/pointcloud/6000.npy')
    points_b, sigmas_b = points_b[..., :3], points_b[..., 3]
    points_b = points_b[np.broadcast_to(sigmas_b[..., None], (sigmas_b.shape[0], 3)) > thresh].reshape(-1, 3)
    sigmas = sigmas_b[sigmas_b > thresh][..., None]
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(points_b)
    pcd_b.paint_uniform_color([0, 1, 1])


    # result_ransac, result_icp = allign_pointclouds(pcd_a, pcd_b, voxel_size=0.01)

    # transform_icp = torch.Tensor(result_ransac.transformation).cuda()
    # transform_icp = torch.Tensor(np.load('./data/transforms/east_west.npy')).cuda()
    # transform_icp = torch.Tensor(np.eye(4)).cuda()

    dim = 1000

    # R_dist = torch.nn.Parameter(torch.full((dim, 3), 1/dim), requires_grad=True)
    # T_dist = torch.nn.Parameter(torch.full((dim, 3), 1/dim), requires_grad=True)

    R_dist = torch.nn.Parameter(torch.zeros((dim, 3)), requires_grad=True)
    T_dist = torch.nn.Parameter(torch.zeros((dim, 3)), requires_grad=True)

    n_rays = 4096

    N = N_a
    H, W = 1500, 2000

    # optimizer = torch.optim.SGD([{'params': [R, T]}], lr=1e-6, momentum=0.9)
    optimizer = torch.optim.Adam([{'params': [R_dist, T_dist]}], lr=1e-4)

    losses = []

    for i in tqdm(range(1000)):
        optimizer.zero_grad()

        n = torch.randint(0, N, (n_rays,))
        h = torch.randint(0, H, (n_rays,))
        w = torch.randint(0, W, (n_rays,))

        K = model_a.intrinsics[n].to('cuda')
        E = model_a.extrinsics[n].to('cuda')

        n = n.to('cuda')
        h = h.to('cuda')
        w = w.to('cuda')

        rays_o, rays_d = helpers.get_rays(h, w, K, E)
        z_vals_log, z_vals = model_a.efficient_sampling(rays_o, rays_d, cfg.renderer.importance_steps)
        xyzs_a, _ = helpers.get_sample_points(rays_o, rays_d, z_vals)
        s_xyzs_a = helpers.mipnerf360_scale(xyzs_a, model_a.inner_bound, model_a.outer_bound)

        N_rays, N_samples = z_vals.shape[:2]



        sigmas_a = model_a.density(s_xyzs_a)

        delta = z_vals_log.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
        delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

        alpha = 1 - torch.exp(-sigmas_a * delta)  # [N_rays, N_samples]

        alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
        weights_a = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]


        R, T = sample_rt(torch.sigmoid(R_dist), torch.sigmoid(T_dist), n_rays)

        # (n_rays, 4, 4)
        transform = torch.clone(torch.eye(4, dtype=torch.float32, device='cuda')[None, :, :].expand(n_rays, 4, 4))
        transform[:, :3, :3] = Exp_vec(R)
        transform[:, :3, 3] = T

        b1 = torch.cat([xyzs_a, n.new_ones((*xyzs_a.shape[:-1], 1))], dim=-1)  # (n_rays, n_samples, 4)
        b2 = (torch.bmm(transform, b1.permute(0, 2, 1))).permute(0, 2, 1)  # (n_rays, n_samples, 4)
        xyzs_b = b2[..., :3].reshape(*xyzs_a.shape)

        # b = xyzs_a.reshape(-1, 3)
        # b1 = torch.cat([b, n.new_ones((b.shape[0], 1))], dim=1)
        # b2 = (transform_icp @ b1.T).T
        # b3 = (transform @ b2.T).T
        # xyzs_b = b3[:, :3].reshape(*xyzs_a.shape)

        s_xyzs_b = helpers.mipnerf360_scale(xyzs_b, model_b.inner_bound, model_b.outer_bound)

        sigmas_b = model_b.density(s_xyzs_b)

        sigmas_ab = sigmas_a + sigmas_b

        delta = z_vals_log.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
        delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

        alpha = 1 - torch.exp(-sigmas_ab * delta)  # [N_rays, N_samples]

        alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
        weights_b = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]


        wa = torch.clone(weights_a)
        wa[:, -1] = 1 - torch.sum(weights_a, dim=-1)

        wb = torch.clone(weights_b)
        wb[:, -1] = 1 - torch.sum(weights_b, dim=-1)

        w = torch.bmm(wa[:, :, None], wb[:, None, :])
        s = torch.abs(z_vals_log[:, :, None] - z_vals_log[:, None, :])
        loss = w * s
        loss = torch.mean(torch.sum(loss, dim=[1, 2]))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 100 == 0:
            if i == 0:
                print(f'{loss.item():.6f}')
                # print(R)
                # print(T)
            else:
                print(f'{(sum(losses[-100:]) / 100):.6f}')
                # print(R)
                # print(T)


    # transform = torch.eye(4, dtype=torch.float32, device='cuda')
    # transform[:3, :3] = Exp(R)
    # transform[:3, 3] = T

    # transform_comb = np.array(transform.detach().cpu()) @ np.array(transform_icp.detach().cpu())

    # pcd_a.transform(transform)
    # o3d.visualization.draw_geometries([pcd_a, pcd_b])

