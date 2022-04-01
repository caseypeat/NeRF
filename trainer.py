import numpy as np
import torch
import cv2
import time
import matplotlib.pyplot as plt

from matplotlib import cm

import helpers

from nets import NerfHash

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


class Trainer(object):
    def __init__(
        self,
        model,
        images,
        depths,
        intrinsics,
        extrinsics,
        optimizer,
        criterion,
        n_rays,
        bound,
        device):

        self.n_rays = n_rays
        self.bound = bound
        self.device = device

        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.optimizer = optimizer
        self.criterion = criterion

        self.model = model

        # self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_len = 1001
        self.epoch_len = 100

        self.loss_avg = 0
        self.loss_rgb_avg = 0
        self.loss_dist_avg = 0


    def train(self):
        t0 = time.time()
        N, H, W, C = self.images.shape
        # ids = torch.Tensor(np.array([0, 5, 24, 25, 48, 49, 72, 73, 96, 97])).to(int).to('cuda')
        # ids = torch.Tensor(np.array([N//2])).to(int).to('cuda')
        for epoch in range(self.train_len):
            # if epoch == 0:
            #     self.train_epoch(1000)
            # else:
            #     self.train_epoch(100)
            self.train_epoch(self.epoch_len, t0)

            print(f'Epoch: {epoch} - Time (s): {time.time() - t0:.2f} - Loss: {self.loss_avg:.7f} - Loss RGB: {self.loss_rgb_avg:.7f} - Loss Dist: {self.loss_dist_avg:.7f}\n')
            self.loss_avg = 0
            self.loss_rgb_avg = 0
            self.loss_dist_avg = 0

            if epoch % 10 == 0 and epoch != 0:
                self.evaluate_image()

                # with torch.cuda.amp.autocast():
            if epoch % 10 == 0 and epoch != 0:
            # if epoch % 50 == 0:
                self.model.extract_geometry(self.bound, H, W, self.intrinsics[:, ...].to('cuda'), self.extrinsics[:, ...].to('cuda'))

    def train_epoch(self, epoch_len, t0):
        # print(f'start update extra state: {time.time() - t0:.2f}')
        # self.model.update_extra_state(self.bound)
        # print(f'finish update extra state: {time.time() - t0:.2f}')

        for i in range(epoch_len):
            self.optimizer.zero_grad()

            loss_rgb, l_dist = self.train_step()
            loss = loss_rgb + 0.001 * l_dist

            self.loss_avg += float(loss) / epoch_len
            self.loss_rgb_avg += float(loss_rgb) / epoch_len
            self.loss_dist_avg += float(l_dist) / epoch_len

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # print('mean_density_inner: ', self.model.mean_density_inner)
        # print('mean_density_outer: ', self.model.mean_density_outer)
        # print()

        # print(self.loss_avg)
        # self.loss_avg = 0

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

        rgb_gt = rgb_gt[None, ...]
        rays_o = rays_o[None, ...]
        rays_d = rays_d[None, ...]

        # with torch.cuda.amp.autocast():
        rgb_pred, depth_pred, l_dist = self.model.render(rays_o, rays_d, self.bound, color_bg, perturb=True, force_all_rays=False)

        loss = self.criterion(rgb_pred, rgb_gt) #+ 0.01 * l_dist

        return loss, l_dist

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
            depth = torch.zeros(n_f.shape)

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
                rays_o = rays_o[None, ...]
                rays_d = rays_d[None, ...]

                # with torch.cuda.amp.autocast():
                image_fb, depth_fb, _ = self.model.render(rays_o, rays_d, self.bound, bg_color=color_bg, perturb=False, force_all_rays=True)

                image[i:end] = image_fb[0, ...]
                depth[i:end] = depth_fb[0, ...]

            image_uf = torch.reshape(image, (*n.shape, 3))[0]
            depth_uf = torch.reshape(depth, n.shape)[0]

            image_uf_grey = torch.sum(image_uf, dim=-1) / 3
            image_gt_grey = torch.sum(image_gt, dim=-1) / 3

            image_diff = torch.ones(image_uf.shape)
            image_diff[..., 0] = 1 - (image_gt_grey - image_uf_grey)
            image_diff[..., 1] = 1 - (image_uf_grey - image_gt_grey)
            image_diff[image_diff < 0] = 0
            image_diff[image_diff > 1] = 1

            # depth_uf[depth_uf == 0] = 1000
            depth_color = color_depthmap_torch(depth_uf.detach().cpu())[..., torch.Tensor([2, 1, 0]).to(int)]

            # plt.imshow(1 / depth_uf)
            # plt.show()

            image_comb = torch.cat([depth_color, image_uf.detach().cpu(), image_gt], dim=0).permute((1, 0, 2))

            cv2.imwrite('./data/test_vine_comb.png', np.uint8(cv2.flip(image_comb.numpy()*255, 1)))