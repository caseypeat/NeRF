import numpy as np
import torch
import matplotlib.pyplot as plt

from loaders.camera_geometry_loader import camera_geometry_loader, camera_geometry_loader_real, meta_camera_geometry, meta_camera_geometry_real
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from trainer import Trainer
from logger import Logger
from inference import Inferencer

from config import cfg


if __name__ == '__main__':

    logger = Logger(
        root_dir=cfg.log.root_dir,
        )

    logger.log('Loading Data...')
    images_a, depths_a, intrinsics_a, extrinsics_a = meta_camera_geometry_real(cfg.scene.scene_path_a, cfg.scene.frame_range)
    images_b, depths_b, intrinsics_b, extrinsics_b = meta_camera_geometry_real(cfg.scene.scene_path_b, cfg.scene.frame_range)

    transform = np.load('./data/transforms/east_west.npy')
    extrinsics_a = torch.Tensor(transform) @ extrinsics_a

    images = torch.cat([images_a, images_b], dim=0)
    depths = None
    intrinsics = torch.cat([intrinsics_a, intrinsics_b], dim=0)
    extrinsics = torch.cat([extrinsics_a, extrinsics_b], dim=0)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - torch.mean(extrinsics[..., :3, 3], dim=0, keepdims=True)

    logger.log('Initilising Model...')
    model = NeRFNetwork(
        # renderer
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        # net
        N = images.shape[0]
    ).to('cuda')

    logger.log('Initiating Inference...')
    inferencer = Inferencer(model)

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
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    logger.log('Initiating Trainer...')
    trainer = Trainer(
        model=model,
        logger=logger,
        inferencer=inferencer,
        images=images,
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        optimizer=optimizer,
        scheduler=scheduler)

    logger.log('Beginning Training...\n')
    trainer.train()
