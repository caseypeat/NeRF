import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from render import get_rays
from rotation import rot2euler
from metrics import pose_inv_error

from align.ransac import global_align


class TrainerAlign(object):
    def __init__(
        self,
        logger,

        transform,
        dataloader_a,
        renderer_a,
        renderer_ab,

        optimizer,
        scheduler,

        iters_per_epoch,
        num_epochs,
        n_rays,):
        # starting_error,):

        self.logger = logger

        self.transform = transform

        self.dataloader_a = dataloader_a
        self.renderer_a = renderer_a
        self.renderer_ab = renderer_ab

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.iters_per_epoch = iters_per_epoch
        self.num_epochs = num_epochs
        self.n_rays = n_rays

        # self.starting_error = starting_error

        self.iter = 0

    def train(self):
        for epoch in range(self.num_epochs):

            # print(self.transform.T - self.transform.init_transform)
            # Z = torch.eye(4, device='cuda')
            # Z[:3, :3] = Exp(self.transform.R)
            # Z[:3, 3] = self.transform.T
            
            # E = self.transform.init_transform @ Z
            # print('rot: ', np.linalg.norm(np.rad2deg(matrix2xyz_extrinsic(E[:3, :3].detach().cpu().numpy()))))
            # print('trans: ', np.linalg.norm(E[:3, 3].detach().cpu().numpy() * 1000))
            # print(self.transform.init_transform[:3, 3] + Z[:3, 3])

            pred = self.transform.get_matrix()
            target = torch.eye(4, device='cuda')
            # target = self.starting_error
            error_rot, error_trans = pose_inv_error(pred, target)
            self.logger.log(f"{self.iter} - Rotation Error (degrees): {torch.rad2deg(error_rot).item():.4f}")
            self.logger.log(f"{self.iter} - Translation Error (mm): {(error_trans * 1000).item():.4f}")
            self.logger.log("")
            # self.logger.scalar("Rotation Error (degrees)", torch.rad2deg(error_rot).item(), self.iter)
            # self.logger.scalar("Translation Error (mm)", (error_trans * 1000).item(), self.iter)

            self.train_epoch(self.iters_per_epoch)

        self.logger.save_transform(self.transform.get_matrix().detach().cpu().numpy(), 'final_transform')

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

        n, h, w, K, E, _, _, _ = self.dataloader_a.get_random_batch(self.n_rays, device='cuda')

        _, weights_a, z_vals_log_norm_a, aux_outputs_a = self.renderer_a.render(n, h, w, K, E)
        weights_a[:, -1] = 1 - torch.sum(weights_a[:, :-1], dim=-1)
        depths_norm_a = torch.sum(weights_a * z_vals_log_norm_a, dim=-1)
        # z_vals_a = aux_outputs_a['z_vals']
        # depths_a = torch.sum(weights_a * z_vals_a, dim=-1)

        _, weights_ab, z_vals_log_norm_ab, aux_outputs_ab = self.renderer_ab.render(n, h, w, K, E)
        weights_ab[:, -1] = 1 - torch.sum(weights_ab[:, :-1], dim=-1)
        depths_norm_ab = torch.sum(weights_ab * z_vals_log_norm_ab, dim=-1)
        # z_vals_ab = aux_outputs_ab['z_vals']
        # depths_ab = torch.sum(weights_ab * z_vals_ab, dim=-1)

        # depths_clipped = torch.clone(depths)
        # depths_clipped[depths_clipped > 1] = 1

        # print(depths_a - depths_clipped)

        # print(depths_a)

        # if self.iter % 100 == 0:
        #     print(calculate_point_error(h, w, K, E, depths, self.transform))

        loss = torch.mean(depths_norm_a - depths_norm_ab)

        # loss = torch.mean(weights_a * z_vals_log_s_a - weights_ab * z_vals_log_s_ab)

        return loss


# def calculate_point_error(h, w, K, E, depth, transform):

#     rays_o, rays_d = get_rays(h, w, K, E)
#     depth_broad = depth[:, None].repeat(1, 3)
#     points = rays_o + rays_d * depth_broad
#     points_target = points[depth_broad < 1].reshape(-1, 3) # remove points with depth > depth_thresh

#     points_pred = transform(points_target)

#     point_error = torch.norm(points_pred - points_target, dim=1)

#     return torch.mean(point_error)