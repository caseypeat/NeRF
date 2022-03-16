import numpy as np
import torch
import cv2

import helpers

from nets2 import NerfHash

from tqdm import tqdm

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

        self.scaler = torch.cuda.amp.GradScaler()

        self.train_len = 1000
        self.epoch_len = 100

        self.loss_avg = 0


    def train(self):
        for epoch in range(self.train_len):
            self.train_epoch()

            if epoch % 10 == 0 and epoch != 0:
                self.evaluate_image()

    def train_epoch(self):
        with torch.cuda.amp.autocast():
            self.model.update_extra_state(self.bound)


        for i in range(self.epoch_len):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = self.train_step()

            self.loss_avg += float(loss) / self.epoch_len

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        print(self.loss_avg)
        self.loss_avg = 0

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
        # print(rays_o.device, rays_d.device)

        rgb_gt = rgb_gt[None, ...]
        rays_o = rays_o[None, ...]
        rays_d = rays_d[None, ...]

        rgb_pred, depth_pred = self.model.render(rays_o, rays_d, self.bound, color_bg, perturb=True)

        loss = self.criterion(rgb_pred, rgb_gt)
        # print(torch.amax(rgb_pred))

        return loss

    def evaluate_image(self):
        self.model.eval()
        with torch.no_grad():

            N, H, W, C = self.images.shape
        
            n = torch.full((1,), 1)
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

                n_fb  = n_fb.to(self.device)
                h_fb  = h_fb.to(self.device)
                w_fb  = w_fb.to(self.device)

                color_bg = torch.ones(3, device=self.device) # [3], fixed white background
                # color_bg = torch.rand(3, device=self.device) # [3], frame-wise random.
                # if C == 4:
                #     rgba_gt = self.images[n_fb, h_fb, w_fb, :].to(self.device)
                #     rgb_gt = rgba_gt[..., :3] * rgba_gt[..., 3:] + color_bg * (1 - rgba_gt[..., 3:])
                # else:
                #     rgb_gt = self.images[n_fb, h_fb, w_fb, :].to(self.device)

                rays_o, rays_d = helpers.get_rays(h_fb, w_fb, K, E)
                rays_o = rays_o[None, ...]
                rays_d = rays_d[None, ...]

                # print(rays_o.shape, rays_d.shape)
                # print(rays_o.dtype, rays_d.dtype)

                with torch.cuda.amp.autocast():
                    image_fb, depth_fb = self.model.render(rays_o, rays_d, self.bound, bg_color=None, perturb=True)

                # print(image_fb.shape, depth_fb.shape)
                # exit()

                image[i:end] = image_fb[0, ...]
                depth[i:end] = depth_fb[0, ...]

            image_uf = torch.reshape(image, (*n.shape, 3))
            depth_uf = torch.reshape(depth, n.shape)

            cv2.imwrite('./data/test.png', np.uint8(image_uf[0].detach().cpu().numpy()*255))