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


class Trainer(object):
    def __init__(
        self,
        model,
        logger,
        inference,
        images,
        depths,
        intrinsics,
        extrinsics,
        optimizer,
        n_rays,
        num_epochs,
        iters_per_epoch,
        eval_freq,
        eval_image_num,):

        self.model = model
        self.logger = logger
        self.inference = inference

        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

        self.n_rays = n_rays
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.num_iters = self.num_epochs * self.iters_per_epoch
        self.eval_freq = eval_freq
        self.eval_image_num = eval_image_num

        self.iter = 0

        self.scalars = {}
        self.scalars['loss'] = np.zeros((self.num_iters))
        self.scalars['loss_rgb'] = np.zeros((self.num_iters))
        self.scalars['loss_dist'] = np.zeros((self.num_iters))
        self.scalars['dist_scalar'] = np.zeros((self.num_iters))


    def train(self):
        t0 = time.time()
        N, H, W, C = self.images.shape

        for epoch in range(self.num_epochs):

            self.train_epoch(self.iters_per_epoch)
            
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.scalars.items():
                self.logger.log(f'Scalar: {key} - Value: {np.mean(val[epoch*self.iters_per_epoch: (epoch+1)*self.iters_per_epoch]).item()}')
            self.logger.log('')

            if epoch % self.eval_freq == 0:
                self.logger.log('Rending Image...')
                image, invdepth = self.inference.render_image(H, W, self.intrinsics[self.eval_image_num], self.extrinsics[self.eval_image_num])
                self.logger.image(image.cpu().numpy(), self.iter)
                self.logger.invdepth(invdepth.cpu().numpy(), self.iter)
                
                self.logger.log('Generating Pointcloud')
                pointcloud = self.inference.extract_geometry()
                self.logger.pointcloud(pointcloud.cpu().numpy(), self.iter)

    def train_epoch(self, iters_per_epoch):
        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            # dist_scalar = 10**(self.iter/self.num_iters * 2 - 4)
            dist_scalar = 10**(-4)

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

        K = self.intrinsics[n].to('cuda')
        E = self.extrinsics[n].to('cuda')

        color_bg = torch.rand(3, device='cuda') # [3], frame-wise random.
        if C == 4:
            rgba_gt = self.images[n, h, w, :].to('cuda')
            rgb_gt = rgba_gt[..., :3] * rgba_gt[..., 3:] + color_bg * (1 - rgba_gt[..., 3:])
        else:
            rgb_gt = self.images[n, h, w, :].to('cuda')

        n = n.to('cuda')
        h = h.to('cuda')
        w = w.to('cuda')
        
        rays_o, rays_d = helpers.get_rays(h, w, K, E)

        rgb, _, weights, z_vals_log = self.model.render(rays_o, rays_d, color_bg)

        loss_rgb = helpers.criterion_rgb(rgb, rgb_gt)
        loss_dist = helpers.criterion_dist(weights, z_vals_log)

        return loss_rgb, loss_dist