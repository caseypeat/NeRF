
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import argparse
import torchmetrics

from omegaconf import DictConfig, OmegaConf

import helpers

from loaders.camera_geometry_loader import CameraGeometryLoader

from nets import NeRFNetwork
from render import NerfRenderer
from trainer import Trainer
from logger import Logger
from inference import Inferencer


if __name__ == '__main__':

    run = '20220723_190910'
    pair = '0006_0007'

    cfg = OmegaConf.load(f'./logs/bothsides/{pair}/{run}/config.yaml')

    logger = Logger(
        root_dir=f'./logs/measure/{pair}/final',
        cfg=cfg,
        )

    logger.log('Initiating Dataloader...')
    dataloader = CameraGeometryLoader(
        scene_paths=cfg.scene.scene_paths,
        frame_ranges=cfg.scene.frame_ranges,
        transforms=cfg.scene.transforms,
        image_scale=cfg.scene.image_scale,
        )

    logger.log('Initilising Model...')
    model = NeRFNetwork(
        N = dataloader.images.shape[0],
        encoding_precision=cfg.nets.encoding.precision,
        encoding_n_levels=cfg.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg.nets.sigma.hidden_dim,
        sigma_num_layers=cfg.nets.sigma.num_layers,
        encoding_dir_precision=cfg.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg.nets.encoding_dir.degree,
        latent_embedding_dim=cfg.nets.latent_embedding.features,
        color_hidden_dim=cfg.nets.color.hidden_dim,
        color_num_layers=cfg.nets.color.num_layers,
    ).to('cuda')
    model.load_state_dict(torch.load(os.path.join(f'{cfg.log.root_dir}', run, 'model', '3000.pth')))

    logger.log('Initilising Renderer...')
    renderer = NerfRenderer(
        model=model,
        inner_bound=cfg.scene.inner_bound,
        outer_bound=cfg.scene.outer_bound,
        z_bounds=cfg.renderer.z_bounds,
        steps_firstpass=cfg.renderer.steps,
        steps_importance=cfg.renderer.importance_steps,
        alpha_importance=cfg.renderer.alpha,
        translation_center=dataloader.translation_center,
    )

    logger.log('Initiating Inference...')
    inferencer = Inferencer(
        renderer=renderer,
        n_rays=cfg.trainer.n_rays,
        image_num=0,
        rotate=cfg.inference.image.rotate,
        max_variance_npy=cfg.inference.pointcloud.max_variance_npy,
        max_variance_pcd=cfg.inference.pointcloud.max_variance_pcd,
        distribution_area=cfg.inference.pointcloud.distribution_area,
        cams=cfg.inference.pointcloud.cams,
        freq=cfg.inference.pointcloud.freq,
        )

    n_indexs = []
    for i in range(dataloader.images.shape[0]):
        if i % 6 in [1, 2, 3, 4]:
            if i//6 % 20 == 0:
                n_indexs.append(i)

    lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')
    ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    for i in n_indexs:
        n, h, w, K, E, rgb_gt, _, _ = dataloader.get_image_batch(i, device='cuda')
        image, invdepth = inferencer.render_image(n, h, w, K, E)

        pred_bchw = torch.clamp(torch.Tensor(image.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        pred_lpips = torch.clamp(pred_bchw * 2 - 1, min=-1, max=1)
        target_bchw = torch.clamp(rgb_gt.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        target_lpips = torch.clamp(target_bchw * 2 - 1, min=-1, max=1)

        lpips_result = lpips(pred_lpips, target_lpips)
        ssim_result = ssim(pred_bchw, target_bchw)
        psnr_result = helpers.psnr(pred_bchw, target_bchw)

        logger.eval_scalar('eval_lpips', lpips_result, 0)
        logger.eval_scalar('eval_ssim', ssim_result, 0)
        logger.eval_scalar('eval_psnr', psnr_result, 0)

    for key, val in logger.eval_scalars.items():
        logger.log(f'Eval Scalar: {key} - Value: {np.mean(np.array(val))}')
        logger.log('')

    

