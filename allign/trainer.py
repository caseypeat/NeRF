import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from allign.ransac import global_allign
from allign.rotation import vec2skew, Exp, matrix2xyz_extrinsic

class TrainerPose(object):
    def __init__(
        self,
        logger,
        transform,
        transform_ransac,
        transform_icp,
        model_a,
        model_b,
        pointcloud_a,
        pointcloud_b,
        dataloader_a,
        dataloader_b,
        renderer,
        measure,
        iters_per_epoch,
        num_epochs,
        n_rays):

        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.num_iters = self.num_epochs * self.iters_per_epoch

        self.logger = logger
        
        self.transform = transform
        self.transform_ransac = transform_ransac
        self.transform_icp = transform_icp

        self.model_a = model_a
        self.model_b = model_b

        self.pointcloud_a = pointcloud_a
        self.pointcloud_b = pointcloud_b

        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b

        self.renderer = renderer
        self.measure = measure

        self.n_rays = n_rays

        self.optimizer = torch.optim.Adam([
            {'name': 'translation', 'params': [self.transform.T], 'lr': 1e-3},
            {'name': 'rotation', 'params': [self.transform.R], 'lr': 1e-3},
            ], lr=1e-3, betas=(0.9, 0.99), eps=1e-15)
        # self.optimizer = torch.optim.SGD([{'params': [self.transform.R, self.transform.T]}], lr=1e-3)
        # self.optimizer = torch.optim.Rprop([{'params': [self.transform.R, self.transform.T]}], lr=1e-3)
        lmbda = lambda x: 0.01**(x/(self.num_iters))
        # lmbda = lambda x: 1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)
        # self.scaler = torch.cuda.amp.GradScaler()

        self.losses = []

        self.iter = 0


    def train(self):
        xyz_error = self.calculate_xyz_error()
        self.logger.log(f'init xyz_error: {np.linalg.norm(xyz_error):.6f}')
        self.logger.log(f'init rot_error: {self.calculate_rot_error():.6f}')
        self.logger.log('')
        # self.logger.log('Prerun')
        # with torch.no_grad():
        #     self.train_step()
        # self.logger.log('')
        
        # for i in range(10):
        #     with torch.no_grad():
        #         self.dry_run(100)
        #     # Output recorded scalars
        #     self.logger.log(f'Dry Iteration: {self.iter}')
        #     for key, val in self.logger.scalars.items():
        #         self.logger.log(f'Scalar: {key} - Value: {np.mean(np.array(val[-self.iters_per_epoch:])).item():.6f}')
        #     self.logger.log('')

        for epoch in range(self.num_epochs):
            self.train_epoch(self.iters_per_epoch)

            # Output recorded scalars
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.logger.scalars.items():
                self.logger.log(f'Scalar: {key} - Value: {np.mean(np.array(val[-self.iters_per_epoch:])).item():.6f}')
            self.logger.log('')

        # Output transforms
        init_transform = self.transform.init_transform.detach().cpu().numpy()
        # init_transform[:3, 3] = init_transform[:3, 3] + (self.dataloader_a.translation_center - self.dataloader_b.translation_center).detach().cpu().numpy()
        self.logger.save_transform(init_transform, 'init_transform')

        final_transform = np.eye(4)
        final_transform[:3, :3] = Exp(self.transform.R).detach().cpu().numpy()
        # final_transform[:3, 3] = self.transform.T.detach().cpu().numpy() + (self.dataloader_a.translation_center - self.dataloader_b.translation_center).detach().cpu().numpy()
        final_transform[:3, 3] = self.transform.T.detach().cpu().numpy()
        final_transform = final_transform @ init_transform
        self.logger.save_transform(final_transform, 'final_transform')

        # Output pointclouds
        self.logger.save_pointcloud(self.pointcloud_a, 'pointcloud_a')
        self.logger.save_pointcloud(self.pointcloud_b, 'pointcloud_b')
        self.logger.save_pointclouds_comb(self.pointcloud_a, self.pointcloud_b, init_transform, 'init_allign')
        self.logger.save_pointclouds_comb(self.pointcloud_a, self.pointcloud_b, final_transform, 'final_allign')
        
    def dry_run(self, iters_per_epoch):
        for i in tqdm(range(iters_per_epoch)):
            loss = self.train_step()
            self.iter += 1
    
    def train_epoch(self, iters_per_epoch):
        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            # with torch.cuda.amp.autocast():
            loss = self.train_step()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.iter += 1

    def train_step(self):
        self.optimizer.zero_grad()

        _, h, w, K, E, _, _, depth = self.dataloader_a.get_random_batch(self.n_rays)

        loss = self.calculate_scene_consistency(h, w, K, E)

        self.logger.scalar('loss', loss, self.iter)
        
        if self.dataloader_a.depths_bool and self.dataloader_b.depths_bool:
            xyz_error = self.calculate_xyz_error()
            self.logger.scalar('xyz_error', np.linalg.norm(xyz_error), self.iter)
            self.logger.scalar('x_error', xyz_error[0], self.iter)
            self.logger.scalar('y_error', xyz_error[1], self.iter)
            self.logger.scalar('z_error', xyz_error[2], self.iter)
            self.logger.scalar('rot_error', self.calculate_rot_error(), self.iter)

            point_error_znerf = self.measure.calculate_point_error(h, w, K, E, depth, self.transform)
            point_error_ransac = self.measure.calculate_point_error(h, w, K, E, depth, self.transform_ransac)
            point_error_icp = self.measure.calculate_point_error(h, w, K, E, depth, self.transform_icp)
            self.logger.scalar('point_error_znerf', point_error_znerf, self.iter)
            self.logger.scalar('point_error_ransac', point_error_ransac, self.iter)
            self.logger.scalar('point_error_icp', point_error_icp, self.iter)

        return loss

    def calculate_xyz_error(self):
        xyz_init = self.transform.init_transform[:3, 3].detach().cpu().numpy()
        xyz_pred = self.transform.T.detach().cpu().numpy() + xyz_init
        xyz_true = (self.dataloader_a.translation_center - self.dataloader_b.translation_center).detach().cpu().numpy()
        xyz_error = (xyz_pred - xyz_true)[0]
        return xyz_error

    def calculate_rot_error(self):
        rot_init = self.transform.init_transform[:3, :3].detach().cpu().numpy()
        rot_pred = Exp(self.transform.R).detach().cpu().numpy() @ rot_init
        rot_true = np.eye(3)  # no adjustment to rotation in synthetic data
        rot_error = np.linalg.norm(matrix2xyz_extrinsic(rot_pred @ np.linalg.inv(rot_true)))
        return rot_error

    def calculate_scene_consistency(self, h, w, K, E):

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

        return loss