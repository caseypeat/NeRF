import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d

from omegaconf import OmegaConf

from tqdm import tqdm

from loaders.camera_geometry_loader import CameraGeometryLoader
from renderer import NerfRenderer
from nets import NeRFNetwork
from utils.allign_pointclouds import allign_pointclouds

# from config.overwriteable_config import OverwriteableConfig


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

# https://github.com/alisterburt/eulerangles/blob/master/eulerangles/math/rotation_matrix_to_eulers.py
def matrix2xyz_extrinsic(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Rz(k3) @ Ry(k2) @ Rx(k1) = [[c2c3, s1s2c3-c1s3, c1s2c3+s1s3],
                                [c2s3, s1s2s3+c1c3, c1s2s3-s1c3],
                                [-s2, s1c2, c1c2]]
    """
    rotation_matrices = rotation_matrices.reshape((-1, 3, 3))
    angles_radians = np.zeros((rotation_matrices.shape[0], 3))

    # Angle 2 can be taken directly from matrices
    angles_radians[:, 1] = -np.arcsin(rotation_matrices[:, 2, 0])

    # Gimbal lock case (c2 = 0)
    tolerance = 1e-4

    # Find indices where this is the case
    gimbal_idx = np.abs(rotation_matrices[:, 0, 0]) < tolerance

    # Calculate angle 1 and set angle 3 = 0 for those indices
    r23 = rotation_matrices[gimbal_idx, 1, 2]
    r22 = rotation_matrices[gimbal_idx, 1, 1]
    angles_radians[gimbal_idx, 0] = np.arctan2(-r23, r22)
    angles_radians[gimbal_idx, 2] = 0

    # Normal case (s2 > 0)
    idx = np.invert(gimbal_idx)
    r32 = rotation_matrices[idx, 2, 1]
    r33 = rotation_matrices[idx, 2, 2]
    r21 = rotation_matrices[idx, 1, 0]
    r11 = rotation_matrices[idx, 0, 0]
    angles_radians[idx, 0] = np.arctan2(r32, r33)
    angles_radians[idx, 2] = np.arctan2(r21, r11)

    # convert to degrees
    euler_angles = np.rad2deg(angles_radians)

    return angles_radians
    # return euler_angles


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


class ExtractDenseGeometry():
    def __init__(
        self,
        model,
        dataloader,
        mask_res,
        voxel_res,
        batch_size,
        thresh,
        outer_bound,
        ):

        self.model = model
        self.dataloader = dataloader
        self.mask_res = mask_res
        self.voxel_res = voxel_res
        self.batch_size = batch_size
        self.thresh = thresh
        self.outer_bound = outer_bound


    @torch.no_grad()
    def get_valid_positions(self, N, H, W, K, E, res):

        mask_full = torch.zeros((res, res, res), dtype=bool, device='cuda')

        for i in tqdm(range(res)):
            d = torch.linspace(-1, 1, res, device='cuda')
            D = torch.stack(torch.meshgrid(d[i], d, d, indexing='ij'), dim=-1)
            dist = torch.linalg.norm(D, dim=-1)[:, :, :, None].expand(-1, -1, -1, 3)
            mask = torch.zeros(dist.shape, dtype=bool, device='cuda')
            mask[dist < 1] = True

            # mask out parts outside camera coverage
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


    @torch.no_grad()
    def extract_dense_geometry(self, N, H, W, K, E):

        mask = self.get_valid_positions(N, H, W, K, E, res=self.mask_res)

        voxels = torch.linspace(-1+1/self.voxel_res, 1-1/self.voxel_res, self.voxel_res, device='cpu')

        num_samples = self.voxel_res**3

        points = torch.zeros((0, 3), device='cpu')
        colors = torch.zeros((0, 3), device='cpu')

        for a in tqdm(range(0, num_samples, self.batch_size)):
            b = min(num_samples, a+self.batch_size)

            n = torch.arange(a, b)

            x = voxels[torch.div(n, self.voxel_res**2, rounding_mode='floor')]
            y = voxels[torch.div(n, self.voxel_res, rounding_mode='floor') % self.voxel_res]
            z = voxels[n % self.voxel_res]

            xyz = torch.stack((x, y, z), dim=-1).cuda()

            x_i = ((x+1)/2*mask.shape[0]).to(int)
            y_i = ((y+1)/2*mask.shape[1]).to(int)
            z_i = ((z+1)/2*mask.shape[2]).to(int)

            xyz = xyz[mask[x_i, y_i, z_i]].view(-1, 3)
            
            dirs = torch.Tensor(np.array([0, 0, 1]))[None, ...].expand(xyz.shape[0], 3).cuda()
            n_i = torch.zeros((xyz.shape[0]), dtype=int).cuda()

            if xyz.shape[0] > 0:
                sigmas, rgbs, _ = self.model(xyz, dirs, n_i, self.outer_bound)
                new_points = xyz[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3).cpu()
                points = torch.cat((points, new_points))
                new_colors = rgbs[sigmas[..., None].expand(-1, 3) > self.thresh].view(-1, 3).cpu()
                colors = torch.cat((colors, new_colors))

        pointcloud = {}
        pointcloud['points'] = points.numpy()
        pointcloud['colors'] = colors.numpy()

        return points, colors


    @torch.no_grad()
    def generate_dense_pointcloud(self):
        N, H, W, K, E = self.dataloader.get_calibration()
        points, _ = self.extract_dense_geometry(N, H, W, K, E)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


class TrainerPose(object):
    def __init__(
        self,
        translation_init,
        transform,
        model_a,
        model_b,
        dataloader_a,
        renderer,
        num_iters,
        n_rays):

        self.num_iters = num_iters  

        self.translation_init = translation_init
        
        self.transform = transform
        self.model_a = model_a
        self.model_b = model_b

        self.dataloader_a = dataloader_a

        self.renderer = renderer

        self.n_rays = n_rays

        self.optimizer = torch.optim.Adam([{'params': [self.transform.R, self.transform.T]}], lr=1e-3)

        lmbda = lambda x: 0.01**(x/(self.num_iters))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

        self.losses = []

    def train(self):
        for i in tqdm(range(self.num_iters)):
            self.optimizer.zero_grad()

            _, h, w, K, E, _, _ = self.dataloader_a.get_random_batch(self.n_rays)

            rays_o, rays_d = self.renderer.get_rays(h, w, K, E)
            z_vals_log, z_vals = self.renderer.efficient_sampling(rays_o, rays_d, self.renderer.steps_importance, self.renderer.alpha_importance)
            xyzs_a, _ = self.renderer.get_sample_points(rays_o, rays_d, z_vals)
            xyzs_warped_a = self.renderer.mipnerf360_scale(xyzs_a, self.renderer.inner_bound, self.renderer.outer_bound)

            sigmas_a, _ = self.model_a.density(xyzs_warped_a, self.renderer.outer_bound)

            delta = z_vals_log.new_zeros(z_vals_log.shape)  # [N_rays, N_samples]
            delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

            alpha_a = 1 - torch.exp(-sigmas_a * delta)  # [N_rays, N_samples]
            alpha_a_shift = torch.cat([alpha_a.new_zeros((alpha_a.shape[0], 1)), alpha_a], dim=-1)[:, :-1]  # [N_rays, N_samples]
            weights_a = alpha_a * torch.cumprod(1 - alpha_a_shift, dim=-1)  # [N_rays, N_samples]


            xyzs_b = self.transform(xyzs_a)
            xyzs_warped_b = self.renderer.mipnerf360_scale(xyzs_b, self.renderer.inner_bound, self.renderer.outer_bound)
            sigmas_b, _ = self.model_b.density(xyzs_warped_b, self.renderer.outer_bound)
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

            if i % 100 == 0:
                if i == 0:
                    print(f'{loss.item():.6f}')
                else:
                    print(f'{(sum(self.losses[-100:]) / 100):.6f}')

                print(np.linalg.norm((self.transform.init_transform[:3, 3] + self.transform.T).detach().cpu().numpy() - self.translation_init.detach().cpu().numpy()))
                R_error = Exp(self.transform.R) @ self.transform.init_transform[:3, :3]
                r_error = matrix2xyz_extrinsic(R_error.detach().cpu().numpy())
                print(np.linalg.norm(r_error), np.rad2deg(np.linalg.norm(r_error)))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())


if __name__ == '__main__':

        # logdir_a = './logs/slow/20220612_173001'
    # logdir_a = './logs/slow/20220612_180801'
    # logdir_a = './logs/pose_optimise3/20220530_162359'
    # logdir_a = './logs/dense/20220615_131636'
    # logdir_a = './logs/zeronerf/20220616_151518'
    # logdir_b = './logs/slow/20220613_095006'
    # logdir_b = './logs/slow/20220613_090902'
    # logdir_b = './logs/pose_optimise3/20220530_170425'
    # logdir_b = './logs/dense/20220615_135611'
    # logdir_b = './logs/zeronerf/20220616_152915'

    logdir_a = './logs/zeronerf/vine_C6_0/20220616_151518'
    logdir_b = './logs/zeronerf/vine_C6_0/20220616_152915'

    cfg_a = OmegaConf.load(f'{logdir_a}/config.yaml')
    cfg_b = OmegaConf.load(f'{logdir_b}/config.yaml')

    dataloader_a = CameraGeometryLoader(
        cfg_a.scene.scene_paths,
        [None],
        [None],
        0.5,
        )

    dataloader_b = CameraGeometryLoader(
        cfg_b.scene.scene_paths,
        [None],
        [None],
        0.5,
        )

    model_a = NeRFNetwork(
        N = len(dataloader_a.images),
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
    model_a.load_state_dict(torch.load(f'{logdir_a}/model/3000.pth'))

    model_b = NeRFNetwork(
        N = len(dataloader_b.images),
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
    model_b.load_state_dict(torch.load(f'{logdir_b}/model/3000.pth'))

    n_rays = 4096
    num_iters = 2000

    # dense_inference_a = ExtractDenseGeometry(model_a, dataloader_a, 128, 512, cfg_a.renderer.importance_steps*n_rays, 100, cfg_a.scene.outer_bound)
    # pcd_dense_a = dense_inference_a.generate_dense_pointcloud()
    # dense_inference_b = ExtractDenseGeometry(model_b, dataloader_b, 128, 512, cfg_b.renderer.importance_steps*n_rays, 100, cfg_b.scene.outer_bound)
    # pcd_dense_b = dense_inference_b.generate_dense_pointcloud()

    # result_ransac, result_icp = allign_pointclouds(pcd_dense_a, pcd_dense_b, voxel_size=0.01)

    # init_transform = np.load('./data/transforms/east_west.npy')  

    T_init = dataloader_a.translation_center - dataloader_b.translation_center

    T_error = torch.Tensor(np.random.random(3))
    T_error = T_error / torch.linalg.norm(T_error) * 0.05 + T_init

    R_error = torch.Tensor(np.random.random(3))
    R_error = R_error / torch.linalg.norm(R_error) * 0.01

    transform_init = torch.eye(4, dtype=torch.float32, device='cuda')
    transform_init[:3, :3] = Exp(R_error)
    transform_init[:3, 3] = T_error

    # R_error = Exp(transform_init)
    # r_error = matrix2xyz_extrinsic(R_error.detach().cpu().numpy())
    # print(np.linalg.norm(r_error))

    # np.save('./data/both_sides_demo4/transform_init.npy', init_transform)

    # exit()

    transform = Transform(transform_init).cuda()

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
        translation_init=T_init,
        transform=transform,
        model_a=model_a,
        model_b=model_b,
        dataloader_a=dataloader_a,
        renderer=renderer,
        num_iters=num_iters,
        n_rays=n_rays
        )

    trainer.train()

    # transform_r = torch.eye(4, dtype=torch.float32, device='cuda')
    # transform_r[:3, :3] = Exp(transform.R)
    # transform_r[:3, 3] = transform.T
    # transform_comb = np.array(transform_r.detach().cpu()) @ np.array(transform.transform_init.detach().cpu())

    # t = transform_comb[:3, 3]
    # print(t)

    # pcd_a.transform(np.array(transform.init_transform.detach().cpu()))
    # o3d.visualization.draw_geometries([pcd_a, pcd_b])

    # pcd_both = o3d.geometry.PointCloud()
    # pcd_both += pcd_a
    # pcd_both += pcd_b
    # o3d.io.write_point_cloud('./data/both_sides_demo/ransac_allign.pcd', pcd_both)

    # pcd_a.transform(np.array(transform_r.detach().cpu()))
    # o3d.visualization.draw_geometries([pcd_a, pcd_b])

    # pcd_both = o3d.geometry.PointCloud()
    # pcd_both += pcd_a
    # pcd_both += pcd_b
    # o3d.io.write_point_cloud('./data/both_sides_demo/nerf_allign.pcd', pcd_both)

    # np.save('./data/both_sides_demo/transform_refine.npy', transform_comb)