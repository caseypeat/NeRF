import numpy as np
import torch
import torchmetrics
import time
import math as m
import matplotlib.pyplot as plt

import helpers

from tqdm import tqdm

# from config import cfg


class Trainer(object):
    def __init__(
        self,
        model,
        dataloader,
        logger,
        inferencer,
        renderer,
        optimizer,
        scheduler,
        
        n_rays,
        num_epochs,
        iters_per_epoch,
        
        eval_image_freq,
        eval_pointcloud_freq,
        save_weights_freq,
        
        dist_loss_lambda1,
        dist_loss_lambda2,

        depth_loss_lambda1,
        depth_loss_lambda2,
        ):

        self.model = model
        self.dataloader = dataloader
        self.logger = logger
        self.inferencer = inferencer
        self.renderer = renderer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.n_rays = n_rays
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.num_iters = self.num_epochs * self.iters_per_epoch

        self.eval_image_freq = eval_image_freq if eval_image_freq != 'end' else num_epochs
        self.eval_pointcloud_freq = eval_pointcloud_freq if eval_pointcloud_freq != 'end' else num_epochs
        self.save_weights_freq =  save_weights_freq if save_weights_freq != 'end' else num_epochs

        self.dist_loss_lambda1 = dist_loss_lambda1
        self.dist_loss_lambda2 = dist_loss_lambda2

        self.depth_loss_lambda1 = depth_loss_lambda1
        self.depth_loss_lambda2 = depth_loss_lambda2

        self.iter = 0

        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def train(self):

        self.t0_train = time.time()

        for epoch in range(self.num_epochs):

            self.train_epoch(self.iters_per_epoch)

            # Output recorded scalars
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.logger.scalars.items():
                self.logger.log(f'Scalar: {key} - Value: {np.mean(np.array(val[-self.iters_per_epoch:])).item():.6f}')
            self.logger.log('')
            
            # Log useful data
            if self.eval_image_freq is not None:
                if (epoch+1) % self.eval_image_freq == 0:
                    self.logger.log('Rending Image...')
                    n, h, w, K, E, rgb_gt, _, _ = self.dataloader.get_image_batch(self.inferencer.image_num, device='cuda')
                    image, invdepth = self.inferencer.render_image(n, h, w, K, E)
                    self.logger.image_color('image', image, self.iter)
                    self.logger.image_grey('invdepth', invdepth, self.iter)

                    # Calculate image metrics
                    pred_bchw = torch.clamp(torch.Tensor(image.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
                    pred_lpips = torch.clamp(pred_bchw * 2 - 1, min=-1, max=1)
                    target_bchw = torch.clamp(rgb_gt.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
                    target_lpips = torch.clamp(target_bchw * 2 - 1, min=-1, max=1)
                    self.logger.eval_scalar('eval_lpips', self.lpips(pred_lpips, target_lpips), self.iter)
                    self.logger.eval_scalar('eval_ssim', self.ssim(pred_bchw, target_bchw), self.iter)
                    self.logger.eval_scalar('eval_psnr', helpers.psnr(pred_bchw, target_bchw), self.iter)

                    for key, val in self.logger.eval_scalars.items():
                        self.logger.log(f'Eval Scalar: {key} - Value: {val[-1]:.6f}')
                    self.logger.log('')

            if self.eval_image_freq is not None:
                if (epoch+1) % self.eval_image_freq == 0:
                    self.logger.log('Rending Invdepth Thresh...')
                    n, h, w, K, E, _, _, _ = self.dataloader.get_image_batch(self.inferencer.image_num, device='cuda')
                    invdepth_thresh = self.inferencer.render_invdepth_thresh(n, h, w, K, E)
                    self.logger.image_grey('invdepth_thresh', invdepth_thresh, self.iter)

            if self.eval_pointcloud_freq is not None:
                if (epoch+1) % self.eval_pointcloud_freq == 0:
                    self.logger.log('Generating Pointcloud...')
                    n, h, w, K, E, _, _, _ = self.dataloader.get_pointcloud_batch(cams=self.inferencer.cams, freq=self.inferencer.freq)
                    pointcloud = self.inferencer.extract_surface_geometry(n, h, w, K, E)
                    self.logger.pointcloud(pointcloud, self.iter, self.inferencer.max_variance_pcd)

            if self.save_weights_freq is not None:
                if (epoch+1) % self.save_weights_freq == 0:
                    self.logger.log('Saving Model...')
                    self.logger.model(self.model, self.iter)


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
        
        n, h, w, K, E, rgb_gt, color_bg, _ = self.dataloader.get_random_batch(self.n_rays)

        rgb, weights, z_vals_log_s, aux_outputs = self.renderer.render(n, h, w, K, E, color_bg)

        # depth_scalar = 0.0

        # Calculate losses
        loss_rgb = helpers.criterion_rgb(rgb, rgb_gt)
        if self.dist_loss_lambda1 == 0 or self.dist_loss_lambda2 == 0:
            loss_dist = 0  # could still calculate loss_dist and multiply with a zero scalar, but has non-trivial computational cost
            dist_scalar = 0
            loss = loss_rgb
        else:
            loss_dist = helpers.criterion_dist(weights, z_vals_log_s)
            dist_scalar = 10**(self.iter/self.num_iters * (m.log10(self.dist_loss_lambda2) - m.log10(self.dist_loss_lambda1)) + m.log10(self.dist_loss_lambda1))
            loss = loss_rgb + dist_scalar * loss_dist

        if self.depth_loss_lambda1 == 0 or self.depth_loss_lambda2 == 0:
            depth_scalar = 0
            loss_depth = 0
        else:
            depth_scalar = 10**(self.iter/self.num_iters * (m.log10(self.depth_loss_lambda2) - m.log10(self.depth_loss_lambda2)) + m.log10(self.depth_loss_lambda2))
            loss_depth = torch.mean(weights * (1 - z_vals_log_s))
            loss = loss + depth_scalar * loss_depth

        # Log scalars
        self.logger.scalar('loss', loss, self.iter)
        self.logger.scalar('loss_rgb', loss_rgb, self.iter)
        self.logger.scalar('psnr_rgb', helpers.psnr(rgb, rgb_gt), self.iter)
        self.logger.scalar('dist_scalar', dist_scalar, self.iter)
        self.logger.scalar('loss_dist', loss_dist, self.iter)
        self.logger.scalar('depth_scalar', depth_scalar, self.iter)
        self.logger.scalar('loss_depth', loss_depth, self.iter)

        self.logger.scalar('loss (seconds)', loss, int(time.time() - self.t0_train))
        self.logger.scalar('loss_rgb (seconds)', loss_rgb, int(time.time() - self.t0_train))
        self.logger.scalar('psnr_rgb (seconds)', helpers.psnr(rgb, rgb_gt), int(time.time() - self.t0_train))

        return loss