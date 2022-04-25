from logging import root
import numpy as np
import torch
import cv2
import commentjson as json
import yaml
import os
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from box import Box

import cProfile, pstats, io
from pstats import SortKey

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader, camera_geometry_loader_real, meta_camera_geometry, meta_camera_geometry_real
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from trainer import Trainer
from logger import Logger
from inference import Inference

from misc import extract_foreground, remove_background, remove_background2

from config import cfg


if __name__ == '__main__':

    logger = Logger(
        root_dir=cfg.log.root_dir,
        )

    logger.log('Loading Data...')
    if cfg.scene.real:
        images, depths, intrinsics, extrinsics = meta_camera_geometry_real(cfg.scene.scene_path_front)
        images_back, depths_back, intrinsics_back, extrinsics_back = meta_camera_geometry_real(cfg.scene.scene_path_back)

        bothsides_transform = torch.Tensor(np.load('./data/bothsides_transform.npy'))
        extrinsics = bothsides_transform @ extrinsics

        print(torch.amax(extrinsics[:, :3, 3], axis=0), torch.amin(extrinsics[:, :3, 3], axis=0))
        print(torch.amax(extrinsics_back[:, :3, 3], axis=0), torch.amin(extrinsics_back[:, :3, 3], axis=0))
        
        images = torch.cat([images, images_back], dim=0)
        intrinsics = torch.cat([intrinsics, intrinsics_back], dim=0)
        extrinsics = torch.cat([extrinsics, extrinsics_back], dim=0)

        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - torch.mean(extrinsics[..., :3, 3], dim=0, keepdims=True)

        print(torch.amax(extrinsics[:, :3, 3], axis=0), torch.amin(extrinsics[:, :3, 3], axis=0))
        

    else:
        images, depths, intrinsics, extrinsics = meta_camera_geometry(cfg.scene.scene_path, cfg.scene.remove_background_bool)



    logger.log('Initilising Model...')
    model = NeRFNetwork(
        # Render args
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        # Net args
        n_levels=cfg.nets.encoding.n_levels,
        n_features_per_level=cfg.nets.encoding.n_features,
        log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
        encoding_precision=cfg.nets.encoding.precision,

        encoding_dir=cfg.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg.nets.encoding_dir.degree,
        encoding_dir_precision=cfg.nets.encoding_dir.precision,

        num_layers=cfg.nets.sigma.num_layers,
        hidden_dim=cfg.nets.sigma.hidden_dim,
        geo_feat_dim=cfg.nets.sigma.geo_feat_dim,

        num_layers_color=cfg.nets.color.num_layers,
        hidden_dim_color=cfg.nets.color.hidden_dim,

        N = images.shape[0]
    ).to('cuda')

    logger.log('Generating Mask...')
    N, H, W = images.shape[:3]
    # mask = helpers.get_valid_positions(N, H, W, intrinsics.to('cuda'), extrinsics.to('cuda'), res=256)
    mask = torch.zeros([256]*3)

    logger.log('Initiating Inference...')
    inference = Inference(
        model=model,
        mask=mask,
        n_rays=cfg.inference.n_rays,
        voxel_res=cfg.inference.voxel_res,
        thresh=cfg.inference.thresh,
        batch_size=cfg.inference.batch_size,
        )

    logger.log('Initiating Optimiser...')

    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': cfg.optimizer.encoding.lr},
            {'name': 'latent_emb', 'params': [model.latent_emb], 'lr': cfg.optimizer.latent_emb.lr},
            {'name': 'transform', 'params': [model.R, model.T], 'lr': cfg.optimizer.latent_emb.lr},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': cfg.optimizer.net.weight_decay, 'lr': cfg.optimizer.net.lr},
        ], betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

    if cfg.scheduler == 'step':
        lmbda = lambda x: 1
    elif cfg.scheduler == 'exp_decay':
        lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
    else:
        raise ValueError
    scheduler = LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    logger.log('Initiating Trainer...')
    trainer = Trainer(
        model=model,
        logger=logger,
        inference=inference,
        images=images,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        optimizer=optimizer,
        scheduler=scheduler,

        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,
        eval_freq=cfg.trainer.eval_freq,
        eval_image_num=cfg.inference.image_num,
        )

    logger.log('Beginning Training...\n')
    trainer.train()
