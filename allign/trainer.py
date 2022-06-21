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
        # translation_init,
        transform,
        model_a,
        model_b,
        dataloader_a,
        renderer,
        num_iters,
        n_rays):

        self.num_iters = num_iters  

        # self.translation_init = translation_init
        
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

            if i % 100 == 0:
                if i == 0:
                    print(f'{loss.item():.6f}')
                else:
                    print(f'{(sum(self.losses[-100:]) / 100):.6f}')

                # print(np.linalg.norm((self.transform.init_transform[:3, 3] + self.transform.T).detach().cpu().numpy() - self.translation_init.detach().cpu().numpy()))
                # R_error = Exp(self.transform.R) @ self.transform.init_transform[:3, :3]
                # r_error = matrix2xyz_extrinsic(R_error.detach().cpu().numpy())
                # print(np.linalg.norm(r_error), np.rad2deg(np.linalg.norm(r_error)))

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.item())