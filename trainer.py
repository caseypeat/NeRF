import numpy as np
import torch
import time
import math as m
import matplotlib.pyplot as plt

import helpers

from tqdm import tqdm

from config import cfg


class Trainer(object):
    def __init__(
        self,
        model,
        logger,
        inferencer,
        images,
        depths,
        intrinsics,
        extrinsics,
        optimizer,
        scheduler):

        self.model = model
        self.logger = logger
        self.inferencer = inferencer

        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.n_rays = cfg.trainer.n_rays
        self.num_epochs = cfg.trainer.num_epochs
        self.iters_per_epoch = cfg.trainer.iters_per_epoch
        self.num_iters = self.num_epochs * self.iters_per_epoch

        self.iter = 0


    def train(self):
        N, H, W, C = self.images.shape

        self.t0_train = time.time()

        for epoch in range(self.num_epochs):

            self.train_epoch(self.iters_per_epoch)
            
            # Log useful data
            if cfg.log.eval_image_freq is not None:
                if (epoch+1) % cfg.log.eval_image_freq == 0:
                    self.logger.log('Rending Image...')
                    image, invdepth = self.inferencer.render_image(H, W, self.intrinsics[cfg.inference.image_num], self.extrinsics[cfg.inference.image_num])
                    self.logger.image_color('image', image.cpu().numpy(), self.iter)
                    self.logger.image_grey('invdepth', invdepth.cpu().numpy(), self.iter)

            if cfg.log.eval_image_freq is not None:
                if (epoch+1) % cfg.log.eval_image_freq == 0:
                    self.logger.log('Rending Invdepth Thresh...')
                    invdepth_thresh = self.inferencer.render_invdepth_thresh(H, W, self.intrinsics[cfg.inference.image_num], self.extrinsics[cfg.inference.image_num])
                    self.logger.image_grey('invdepth_thresh', invdepth_thresh.cpu().numpy(), self.iter)

            if cfg.log.eval_pointcloud_freq is not None:
                if (epoch+1) % cfg.log.eval_pointcloud_freq == 0:
                    self.logger.log('Generating Pointcloud...')
                    pointcloud = self.inferencer.extract_geometry(N, H, W, self.intrinsics, self.extrinsics)
                    self.logger.pointcloud(pointcloud.cpu().numpy(), self.iter)

            if cfg.log.save_weights_freq is not None:
                if (epoch+1) % cfg.log.save_weights_freq == 0:
                    self.logger.log('Saving Model...')
                    self.logger.model(self.model, self.iter)
            
            # Output recorded scalars
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.logger.scalars.items():
                self.logger.log(f'Scalar: {key} - Value: {np.mean(np.array(val[-self.iters_per_epoch:])).item():.6f}')
            self.logger.log('')


    def train_epoch(self, iters_per_epoch):
        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = self.train_step()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()

            self.iter += 1


    def train_step(self):
        self.model.train()
        N, H, W, C = self.images.shape

        n = torch.randint(0, N, (self.n_rays,))
        h = torch.randint(0, H, (self.n_rays,))
        w = torch.randint(0, W, (self.n_rays,))

        K = self.intrinsics[n].to('cuda')
        E = self.extrinsics[n].to('cuda')

        # If image data is stored as uint8, convert to float32 and scale to (0, 1)
        # is alpha channel is present, add random background color to image data
        color_bg = torch.rand(3, device='cuda') # [3], frame-wise random.
        if C == 4:
            if self.images.dtype == torch.uint8:
                rgba_gt = (self.images[n, h, w, :].to(torch.float32) / 255).to('cuda')
            else:
                rgba_gt = self.images[n, h, w, :].to('cuda')
            rgb_gt = rgba_gt[..., :3] * rgba_gt[..., 3:] + color_bg * (1 - rgba_gt[..., 3:])
        else:
            if self.images.dtype == torch.uint8:
                rgb_gt = (self.images[n, h, w, :].to(torch.float32) / 255).to('cuda')
            else:
                rgb_gt = self.images[n, h, w, :].to('cuda')

        n = n.to('cuda')
        h = h.to('cuda')
        w = w.to('cuda')

        rgb, weights, z_vals_log_s, aux_outputs = self.model.render(n, h, w, K, E, color_bg)

        # Calculate losses
        loss_rgb = helpers.criterion_rgb(rgb, rgb_gt)
        if cfg.trainer.dist_loss_lambda1 == 0 or cfg.trainer.dist_loss_lambda2 == 0:
            loss_dist = 0  # could still calculate loss_dist and multiply with a zero scalar, but has non-trivial computational cost
            dist_scalar = 0
            loss = loss_rgb
        else:
            loss_dist = helpers.criterion_dist(weights, z_vals_log_s)
            dist_scalar = 10**(self.iter/self.num_iters * (m.log10(cfg.trainer.dist_loss_lambda2) - m.log10(cfg.trainer.dist_loss_lambda1)) + m.log10(cfg.trainer.dist_loss_lambda1))
            loss = loss_rgb + dist_scalar * loss_dist

        # Log scalars
        self.logger.scalar('loss', loss, self.iter)
        self.logger.scalar('loss_rgb', loss_rgb, self.iter)
        self.logger.scalar('loss_dist', loss_dist, self.iter)
        self.logger.scalar('dist_scalar', dist_scalar, self.iter)
        self.logger.scalar('psnr_rgb', helpers.psnr(rgb, rgb_gt), self.iter)

        self.logger.scalar('loss (seconds)', loss, int(time.time() - self.t0_train))
        self.logger.scalar('loss_rgb (seconds)', loss_rgb, int(time.time() - self.t0_train))
        self.logger.scalar('psnr_rgb (seconds)', helpers.psnr(rgb, rgb_gt), int(time.time() - self.t0_train))

        self.logger.scalar('R', torch.linalg.norm(aux_outputs['R']), self.iter)
        self.logger.scalar('T', torch.linalg.norm(aux_outputs['T']), self.iter)

        return loss