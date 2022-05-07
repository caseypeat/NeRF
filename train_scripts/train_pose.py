import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d

from tqdm import tqdm

from loaders.camera_geometry_loader2 import CameraGeometryLoader
from renderer2 import NerfRenderer
from nets2 import NeRFNetwork
from utils.allign_pointclouds import allign_pointclouds

from config.overwriteable_config import OverwriteableConfig


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


class Transform(nn.Module):
    def __init__(self, init_transform):
        super().__init__()

        self.init_transform = torch.nn.Parameter(init_transform, requires_grad=False)

        self.R = torch.nn.Parameter(torch.zeros((3,)), requires_grad=True)
        self.T = torch.nn.Parameter(torch.zeros((3,)), requires_grad=True)

    def forward(self, xyzs_a):

        transform = torch.eye(4, dtype=torch.float32, device='cuda')
        transform[:3, :3] = Exp(self.R)
        transform[:3, 3] = self.T

        b = xyzs_a.reshape(-1, 3)
        b1 = torch.cat([b, b.new_ones((b.shape[0], 1))], dim=1)
        b2 = (self.init_transform @ b1.T).T
        b3 = (transform @ b2.T).T
        xyzs_b = b3[:, :3].reshape(*xyzs_a.shape)

        return xyzs_b



# class NerfRendererModels(NerfRenderer):
#     def __init__(self, **kwargs):
#         super().__init__()


#     def render_models(self, n, h, w, K, E, bg_color, model_a, model_b):

#         rays_o, rays_d = self.get_rays(h, w, K, E)
#         z_vals_log, z_vals = self.efficient_sampling(rays_o, rays_d, self.steps_importance, self.alpha_importance)
#         xyzs, dirs = self.get_sample_points(rays_o, rays_d, z_vals)
#         s_xyzs = self.mipnerf360_scale(xyzs, self.inner_bound, self.outer_bound)
#         n_expand = n[:, None].expand(-1, z_vals.shape[-1])

#         sigmas_a, rgbs_a, _ = model_a(s_xyzs, dirs, n_expand, self.outer_bound)
#         sigmas_b, rgbs_b, _ = model_b(s_xyzs, dirs, n_expand, self.outer_bound)
#         sigmas = sigmas_a + sigmas_b
#         rgbs = rgbs_a + rgbs_b

#         image, invdepth, weights = self.render_rays_log(sigmas, rgbs, z_vals, z_vals_log)
#         image = image + (1 - torch.sum(weights, dim=-1)[..., None]) * bg_color
#         z_vals_log_s = (z_vals_log - m.log10(self.z_bounds[0])) / (m.log10(self.z_bounds[-1]) - m.log10(self.z_bounds[0]))

#         return image, weights


class TrainerPose(object):
    def __init__(
        self,
        transform,
        model_a,
        model_b,
        dataloader_a,
        renderer,
        num_iters):

        self.num_iters = num_iters
        
        self.transform = transform
        self.model_a = model_a
        self.model_b = model_b

        self.dataloader_a = dataloader_a

        self.renderer = renderer

        self.n_rays = 4096

        self.optimizer = torch.optim.Adam([{'params': [self.transform.R, self.transform.T]}], lr=1e-3)

        lmbda = lambda x: 0.1**(x/(self.num_iters))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

        self.losses = []

    def train(self):
        for i in tqdm(range(self.num_iters)):
            self.optimizer.zero_grad()

            _, h, w, K, E, _, _ = self.dataloader_a.get_random_batch(self.n_rays)

            rays_o, rays_d = self.renderer.get_rays(h, w, K, E)
            z_vals_log, z_vals = self.renderer.efficient_sampling(rays_o, rays_d, self.renderer.steps_importance, self.renderer.alpha_importance)
            xyzs_a, _ = self.renderer.get_sample_points(rays_o, rays_d, z_vals)
            s_xyzs_a = self.renderer.mipnerf360_scale(xyzs_a, self.renderer.inner_bound, self.renderer.outer_bound)

            sigmas_a, _ = self.model_a.density(s_xyzs_a, self.renderer.outer_bound)

            delta = z_vals_log.new_zeros(z_vals_log.shape)  # [N_rays, N_samples]
            delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

            alpha_a = 1 - torch.exp(-sigmas_a * delta)  # [N_rays, N_samples]
            alpha_a_shift = torch.cat([alpha_a.new_zeros((alpha_a.shape[0], 1)), alpha_a], dim=-1)[:, :-1]  # [N_rays, N_samples]
            weights_a = alpha_a * torch.cumprod(1 - alpha_a_shift, dim=-1)  # [N_rays, N_samples]


            xyzs_b = self.transform(xyzs_a)
            s_xyzs_b = self.renderer.mipnerf360_scale(xyzs_b, self.renderer.inner_bound, self.renderer.outer_bound)
            sigmas_b, _ = self.model_b.density(s_xyzs_b, self.renderer.outer_bound)
            sigmas_ab = sigmas_a + sigmas_b

            alpha_ab = 1 - torch.exp(-sigmas_ab * delta)  # [N_rays, N_samples]
            alpha_ab_shift = torch.cat([alpha_ab.new_zeros((alpha_ab.shape[0], 1)), alpha_ab], dim=-1)[:, :-1]  # [N_rays, N_samples]
            weights_ab = alpha_ab * torch.cumprod(1 - alpha_ab_shift, dim=-1)  # [N_rays, N_samples]

            weights_a[:, -1] = 1 - torch.sum(weights_a[:, :-1], dim=-1)
            weights_ab[:, -1] = 1 - torch.sum(weights_ab[:, :-1], dim=-1)

            w = torch.bmm(weights_a[:, :, None], weights_ab[:, None, :])
            s = torch.abs(z_vals_log[:, :, None] - z_vals_log[:, None, :])
            loss = w * s
            loss = torch.mean(torch.sum(loss, dim=[1, 2]))

            # loss = F.mse_loss(weights_a, weights_ab)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())

            if i % 100 == 0:
                # print(self.transform.R)
                # print(self.transform.T)
                if i == 0:
                    print(f'{loss.item():.6f}')
                else:
                    print(f'{(sum(self.losses[-100:]) / 100):.6f}')


if __name__ == '__main__':

    pcd_a = o3d.io.read_point_cloud('./logs/hashtable_outputs/20220506_162403/pointclouds/pcd/10000.pcd')
    pcd_b = o3d.io.read_point_cloud('./logs/hashtable_outputs/20220506_143522/pointclouds/pcd/10000.pcd')

    # print(np.array(pcd_a.points).shape)
    
    # result_ransac, result_icp = allign_pointclouds(pcd_a.uniform_down_sample(100), pcd_b.uniform_down_sample(100), voxel_size=0.01)

    # o3d.visualization.draw_geometries([pcd_a.transform(result_ransac.transformation), pcd_b])

    init_transform = np.load('./data/transforms/east_west.npy')
    # o3d.visualization.draw_geometries([pcd_a.transform(init_transform), pcd_b])

    transform = Transform(torch.Tensor(init_transform)).cuda()

    cfg_a = OverwriteableConfig('./logs/hashtable_outputs/20220506_162403/config.yaml')
    cfg_b = OverwriteableConfig('./logs/hashtable_outputs/20220506_143522/config.yaml')

    model_a = NeRFNetwork(
        N = 1176,
        encoding_precision=cfg_a.nets.encoding.precision,
        encoding_n_levels=cfg_a.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg_a.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg_a.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg_a.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg_a.nets.sigma.hidden_dim,
        sigma_num_layers=cfg_a.nets.sigma.num_layers,
        encoding_dir_precision=cfg_a.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg_a.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg_a.nets.encoding_dir.degree,
        latent_embedding_dim=cfg_a.nets.latent_embedding.features,
        color_hidden_dim=cfg_a.nets.color.hidden_dim,
        color_num_layers=cfg_a.nets.color.num_layers,
    ).to('cuda')
    model_a.load_state_dict(torch.load('./logs/hashtable_outputs/20220506_162403/model/10000.pth'))

    model_b = NeRFNetwork(
        N = 972,
        encoding_precision=cfg_b.nets.encoding.precision,
        encoding_n_levels=cfg_b.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg_b.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg_b.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg_b.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg_b.nets.sigma.hidden_dim,
        sigma_num_layers=cfg_b.nets.sigma.num_layers,
        encoding_dir_precision=cfg_b.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg_b.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg_b.nets.encoding_dir.degree,
        latent_embedding_dim=cfg_b.nets.latent_embedding.features,
        color_hidden_dim=cfg_b.nets.color.hidden_dim,
        color_num_layers=cfg_b.nets.color.num_layers,
    ).to('cuda')
    model_b.load_state_dict(torch.load('./logs/hashtable_outputs/20220506_143522/model/10000.pth'))

    dataloader_a = CameraGeometryLoader(
        ['/home/casey/Documents/PhD/data/conan_scans/ROW_349_EAST_SLOW_0006/scene.json'],
        [None],
        0.5
        )

    renderer = NerfRenderer(
        model=model_a,
        inner_bound=cfg_a.scene.inner_bound,
        outer_bound=cfg_a.scene.outer_bound,
        z_bounds=[0.1, 1],
        steps_firstpass=[512],
        steps_importance=cfg_a.renderer.importance_steps,
        alpha_importance=cfg_a.renderer.alpha,
    )

    trainer = TrainerPose(
        transform=transform,
        model_a=model_a,
        model_b=model_b,
        dataloader_a=dataloader_a,
        renderer=renderer,
        num_iters=1000,
        )

    trainer.train()

    transform_r = torch.eye(4, dtype=torch.float32, device='cuda')
    transform_r[:3, :3] = Exp(transform.R)
    transform_r[:3, 3] = transform.T
    transform_comb = np.array(transform_r.detach().cpu()) @ np.array(transform.init_transform.detach().cpu())

    pcd_a.transform(np.array(transform.init_transform.detach().cpu()))
    o3d.visualization.draw_geometries([pcd_a, pcd_b])
    pcd_a.transform(np.array(transform_r.detach().cpu()))
    o3d.visualization.draw_geometries([pcd_a, pcd_b])