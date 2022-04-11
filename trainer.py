import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import math as m
import matplotlib.pyplot as plt
import cProfile

from torch.profiler import profile, record_function, ProfilerActivity

from matplotlib import cm

import helpers

from tqdm import tqdm

from config import cfg


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

        # self.scalars = {}
        # self.scalars['loss'] = np.zeros((self.num_iters))
        # self.scalars['loss_rgb'] = np.zeros((self.num_iters))
        # self.scalars['loss_dist'] = np.zeros((self.num_iters))
        # self.scalars['dist_scalar'] = np.zeros((self.num_iters))


    def train(self):
        N, H, W, C = self.images.shape

        # with torch.profiler.profile(
        #     # activities=[],
        #     schedule=torch.profiler.schedule(skip_first=10, wait=5, warmup=5, active=5, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logger.tensorboard_dir),
        #     record_shapes=True,
        #     profile_memory=True,
        #     # with_stack=True,
        #     ) as prof:

        for epoch in range(self.num_epochs):

            if epoch % self.eval_freq == 0 and epoch != 0:
                self.logger.log('Rending Image...')
                image, invdepth = self.inference.render_image(H, W, self.intrinsics[self.eval_image_num], self.extrinsics[self.eval_image_num])
                self.logger.image(image.cpu().numpy(), self.iter)
                self.logger.invdepth(invdepth.cpu().numpy(), self.iter)
                
            if epoch % cfg.inference.pointcloud_eval_freq == 0 and epoch != 0:
                self.logger.log('Generating Pointcloud')
                pointcloud = self.inference.extract_geometry()
                self.logger.pointcloud(pointcloud.cpu().numpy(), self.iter)

            self.train_epoch(self.iters_per_epoch)
            # self.train_epoch(self.iters_per_epoch)
            
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.logger.scalars.items():
                self.logger.log(f'Scalar: {key} - Value: {np.mean(np.array(val[-self.iters_per_epoch:])).item():.6f}')
            self.logger.log('')

    def train_epoch(self, iters_per_epoch):
        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            # dist_scalar = 10**(self.iter/self.num_iters * 2 - 4)

            with torch.cuda.amp.autocast():
                loss = self.train_step()
                # loss_rgb, loss_dist = self.train_step()
                # loss = loss_rgb + dist_scalar * loss_dist

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.iter += 1

            # prof.step()

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

        rgb, _, weights, z_vals_log_s = self.model.render(rays_o, rays_d, color_bg)

        loss_rgb = helpers.criterion_rgb(rgb, rgb_gt)
        loss_dist = helpers.criterion_dist(weights, z_vals_log_s)

        dist_scalar = 10**(self.iter/self.num_iters * (m.log10(cfg.trainer.dist_loss_lambda2) - m.log10(cfg.trainer.dist_loss_lambda1)) + m.log10(cfg.trainer.dist_loss_lambda1))
        loss = loss_rgb + dist_scalar * loss_dist

        self.logger.scalar('loss', loss, self.iter)
        self.logger.scalar('loss_rgb', loss_rgb, self.iter)
        self.logger.scalar('loss_dist', loss_dist, self.iter)
        self.logger.scalar('dist_scalar', dist_scalar, self.iter)
        self.logger.scalar('psnr_rgb', helpers.psnr(rgb, rgb_gt), self.iter)

        # return loss_rgb, loss_dist
        return loss