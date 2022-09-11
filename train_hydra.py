import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse

from omegaconf import DictConfig, OmegaConf

from loaders.camera_geometry_loader_re2 import CameraGeometryLoader

from nets import NeRFNetwork
# from renderer import NerfRenderer
from trainer import Trainer
from logger import Logger
from inference import Inferencer


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    logger = Logger(
        root_dir=cfg.log.root_dir,
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
    if cfg.inference.image.image_num == 'middle':
        inference_image_num = int(dataloader.images.shape[0]/2) + 3
    else:
        inference_image_num = cfg.inference.image.image_num
    inferencer = Inferencer(
        renderer=renderer,
        n_rays=cfg.trainer.n_rays,
        image_num=inference_image_num,
        rotate=cfg.inference.image.rotate,
        max_variance_npy=cfg.inference.pointcloud.max_variance_npy,
        max_variance_pcd=cfg.inference.pointcloud.max_variance_pcd,
        distribution_area=cfg.inference.pointcloud.distribution_area,
        cams=cfg.inference.pointcloud.cams,
        freq=cfg.inference.pointcloud.freq,
        )

    logger.log('Initiating Optimiser...')
    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': cfg.optimizer.encoding.lr},
            {'name': 'latent_emb', 'params': [model.latent_emb], 'lr': cfg.optimizer.latent_emb.lr},
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
        dataloader=dataloader,
        logger=logger,
        inferencer=inferencer,
        renderer=renderer,

        optimizer=optimizer,
        scheduler=scheduler, 
        
        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,

        eval_image_freq=cfg.log.eval_image_freq,
        eval_pointcloud_freq=cfg.log.eval_pointcloud_freq,
        save_weights_freq=cfg.log.save_weights_freq,

        dist_loss_lambda1=cfg.trainer.dist_loss_lambda1,
        dist_loss_lambda2=cfg.trainer.dist_loss_lambda2,

        depth_loss_lambda1=cfg.trainer.depth_loss_lambda1,
        depth_loss_lambda2=cfg.trainer.depth_loss_lambda2,
        )

    logger.log('Beginning Training...\n')
    trainer.train()


if __name__ == '__main__':

    train()