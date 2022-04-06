import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

import helpers

from tqdm import tqdm


def color_depthmap_torch(grey, maxval=None, minval=None):

    if minval is None:
        minval = torch.min(grey)
    if maxval is None:
        maxval = torch.max(grey)

    grey -= minval
    grey[grey < 0] = 0
    grey /= maxval

    rgb = torch.Tensor(cm.get_cmap(plt.get_cmap('jet'))(grey)[:, :, :3])

    return rgb


def criterion_dist(weights, z_vals):
    w = torch.bmm(weights[:, :, None], weights[:, None, :])
    s = torch.abs(z_vals[:, :, None] - z_vals[:, None, :])
    loss = w * s
    loss = torch.mean(torch.sum(loss, dim=[1, 2]))
    return loss


def criterion_rgb(rgb_, rgb):
    loss = F.huber_loss(rgb_, rgb, delta=0.1)
    return loss


class Trainer(object):
    def __init__(
        self,
        model,
        images,
        depths,
        intrinsics,
        extrinsics,
        optimizer,
        n_rays,
        bound,
        device,
        mask):

        self.n_rays = n_rays
        self.bound = bound
        self.device = device

        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.optimizer = optimizer

        self.model = model

        self.scaler = torch.cuda.amp.GradScaler()

        self.num_epochs = 101
        self.iters_per_epoch = 100
        self.num_iters = self.num_epochs * self.iters_per_epoch

        self.evaluate_freq = 20

        self.iter = 0

        self.mask = mask

        self.scalars = {}
        self.scalars['loss'] = np.zeros((self.num_iters))
        self.scalars['loss_rgb'] = np.zeros((self.num_iters))
        self.scalars['loss_dist'] = np.zeros((self.num_iters))
        self.scalars['dist_scalar'] = np.zeros((self.num_iters))


    def train(self):
        t0 = time.time()
        N, H, W, C = self.images.shape

        for epoch in range(self.num_epochs):

            l_dist_scalar = 10**(self.iter/self.num_iters * 2 - 4)

            self.train_epoch(self.iters_per_epoch)
            
            print(f'Time: {time.time() - t0} s')
            print(f'Iteration: {self.iter}')
            for key, val in self.scalars.items():
                print(f'Scalar: {key} - Value: {np.mean(val[epoch*self.iters_per_epoch: (epoch+1)*self.iters_per_epoch]).item()}')
            print()

            # print(f'Epoch: {epoch} - Time (s): {time.time() - t0:.2f} - Loss: {self.loss_avg:.7f} - Loss RGB: {self.loss_rgb_avg:.7f} - Loss Dist: {self.loss_dist_avg:.7f}')
            # print(f'L Dist Scalar: {l_dist_scalar}')
            # print()
            # self.loss_avg = 0
            # self.loss_rgb_avg = 0
            # self.loss_dist_avg = 0

            if epoch % self.evaluate_freq == 0 and epoch != 0:
            # if epoch % self.evaluate_freq == 0:
                self.evaluate_image()
                self.extract_geometry(self.bound, self.mask)

    def train_epoch(self, iters_per_epoch):

        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            dist_scalar = 10**(self.iter/self.num_iters * 2 - 4)
            # dist_scalar = 10**(-4)

            loss_rgb, loss_dist = self.train_step()
            loss = loss_rgb + dist_scalar * loss_dist

            self.scalars['loss'][self.iter] = loss
            self.scalars['loss_rgb'][self.iter] = loss_rgb
            self.scalars['loss_dist'][self.iter] = loss_dist
            self.scalars['dist_scalar'][self.iter] = dist_scalar

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.iter += 1

    def train_step(self):
        self.model.train()
        N, H, W, C = self.images.shape

        n = torch.randint(0, N, (self.n_rays,))
        h = torch.randint(0, H, (self.n_rays,))
        w = torch.randint(0, W, (self.n_rays,))

        K = self.intrinsics[n].to(self.device)
        E = self.extrinsics[n].to(self.device)

        color_bg = torch.rand(3, device=self.device) # [3], frame-wise random.
        if C == 4:
            rgba_gt = self.images[n, h, w, :].to(self.device)
            rgb_gt = rgba_gt[..., :3] * rgba_gt[..., 3:] + color_bg * (1 - rgba_gt[..., 3:])
        else:
            rgb_gt = self.images[n, h, w, :].to(self.device)

        n = n.to(self.device)
        h = h.to(self.device)
        w = w.to(self.device)
        
        rays_o, rays_d = helpers.get_rays(h, w, K, E)

        # rgb_gt = rgb_gt[None, ...]
        # rays_o = rays_o[None, ...]
        # rays_d = rays_d[None, ...]

        rgb_pred, _, weights, z_vals_log = self.model.render(rays_o, rays_d, self.bound, color_bg)

        loss_rgb = criterion_rgb(rgb_pred, rgb_gt)
        loss_dist = criterion_dist(weights, z_vals_log)

        return loss_rgb, loss_dist

    def extract_geometry(self, bound, mask):
        with torch.no_grad():
            scale = 4
            res = 512 * scale
            thresh = 10

            points = torch.zeros((0, 4), device='cuda')

            for i in tqdm(range(res)):
                d = torch.linspace(-1, 1, res, device='cuda')
                D = torch.stack(torch.meshgrid(d[i], d, d), dim=-1)

                mask_ = mask[i//scale, :, :, None].expand(-1, -1, 3)[None, ...].to('cuda')
                # mask_ = F.interpolate(mask_.permute(0, 3, 1, 2).to(float), (res, res), mode='nearest-exact').to(bool).permute(0, 2, 3, 1)
                mask_ = F.interpolate(mask_.permute(0, 3, 1, 2).to(float), (res, res), mode='nearest').to(bool).permute(0, 2, 3, 1)

                xyzs = D[mask_].view(-1, 3)

                if xyzs.shape[0] > 0:
                    sigmas = self.model.density(xyzs, bound).to(torch.float32)
                    new_points = torch.cat((xyzs[sigmas[..., None].expand(-1, 3) > thresh].view(-1, 3), sigmas[sigmas > thresh][..., None]), dim=-1)
                    points = torch.cat((points, new_points))

            print(points.shape)
            np.save('./data/points.npy', points.cpu().numpy())

    def evaluate_image(self):
        self.model.eval()
        with torch.no_grad():

            N, H, W, C = self.images.shape

            image_idx = N//2

            image_gt = self.images[image_idx]
            image_gt = image_gt[..., :3]
            image_gt[image_gt == 0] = 1
        
            n = torch.full((1,), image_idx)
            h = torch.arange(0, H)
            w = torch.arange(0, W)

            n, h, w = torch.meshgrid(n, h, w)

            n_f = torch.reshape(n, (-1,))
            h_f = torch.reshape(h, (-1,))
            w_f = torch.reshape(w, (-1,))

            image = torch.zeros((*n_f.shape, 3))
            inv_depth = torch.zeros(n_f.shape)

            for i in tqdm(range(0, len(n_f), self.n_rays)):
                end = min(len(n_f), i+self.n_rays)

                n_fb = n_f[i:end]
                h_fb = h_f[i:end]
                w_fb = w_f[i:end]

                K = self.intrinsics[n_fb].to(self.device)
                E = self.extrinsics[n_fb].to(self.device)

                n_fb = n_fb.to(self.device)
                h_fb = h_fb.to(self.device)
                w_fb = w_fb.to(self.device)

                color_bg = torch.ones(3, device=self.device) # [3], fixed white background

                rays_o, rays_d = helpers.get_rays(h_fb, w_fb, K, E)

                image_fb, inv_depth_fb, _, _ = self.model.render(rays_o, rays_d, self.bound, bg_color=color_bg)

                image[i:end] = image_fb[...]
                inv_depth[i:end] = inv_depth_fb[...]

            image_uf = torch.reshape(image, (*n.shape, 3))[0]
            inv_depth_uf = torch.reshape(inv_depth, n.shape)[0]

            inv_depth_color = color_depthmap_torch(inv_depth_uf.detach().cpu())[..., torch.Tensor([2, 1, 0]).to(int)]

            image_comb = torch.cat([inv_depth_color, image_uf.detach().cpu(), image_gt], dim=0).permute((1, 0, 2))

            cv2.imwrite('./data/test_vine_comb.png', np.uint8(cv2.flip(image_comb.numpy()*255, 1)))