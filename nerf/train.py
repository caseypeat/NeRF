import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import argparse

from omegaconf import DictConfig, OmegaConf

from loaders.camera_geometry_loader_re import CameraGeometryLoader, IndexMapping

from nerf.nets import NeRFNetwork
from nerf.renderer import NerfRenderer, PoseOptimise
from nerf.trainer import Trainer
from nerf.logger import Logger
from nerf.inference import Inferencer


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    logger = Logger(
        root_dir=cfg.log.root_dir,
        cfg=cfg,
        )

    logger.log('Initiating Dataloader...')
    dataloader = CameraGeometryLoader(
        scan_paths=cfg.scene.scan_paths,
        frame_ranges=cfg.scene.frame_ranges,
        frame_strides=cfg.scene.frame_strides,
        transforms=cfg.scene.transforms,
        image_scale=cfg.scene.image_scale,
        )
    index_mapping = IndexMapping(dataloader.index2src_mapping, dataloader.src2index_mapping)

    logger.log('Initilising Model...')
    model = NeRFNetwork(
        N = dataloader.N,
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
    # pose_optimise = PoseOptimise(index_mapping).to('cuda')
    pose_optimise = None
    renderer = NerfRenderer(
        model=model,
        pose_optimise=pose_optimise,
        inner_bound=cfg.scene.inner_bound,
        outer_bound=cfg.scene.outer_bound,
        z_bounds=cfg.renderer.z_bounds,
        steps_firstpass=cfg.renderer.steps,
        steps_importance=cfg.renderer.importance_steps,
        alpha_importance=cfg.renderer.alpha,
        translation_center=dataloader.translation_center,
    )

    logger.log('Initilising Inference...')
    scan = dataloader.get_num_scans() // 2 if cfg.inference.image.scan == 'middle' else cfg.inference.image.scan
    logger.log(f'Scan: {scan}')
    rig = dataloader.get_num_rigs(scan) // 2 if cfg.inference.image.rig == 'middle' else cfg.inference.image.rig
    logger.log(f'Rig: {rig}')
    cam = dataloader.get_num_cams(scan, rig) // 2 if cfg.inference.image.cam == 'middle' else cfg.inference.image.cam
    logger.log(f'Cam: {cam}')
    inference_image_num = dataloader.src_to_index(scan, rig, cam)
    logger.log(f'Image: {inference_image_num}')
    inferencer = Inferencer(
        dataloader=dataloader,
        renderer=renderer,
        n_rays=cfg.trainer.n_rays,
        image_num=inference_image_num,
        # rotate=cfg.inference.image.rotate,
        max_variance_npy=cfg.inference.pointcloud.max_variance_npy,
        max_variance_pcd=cfg.inference.pointcloud.max_variance_pcd,
        distribution_area=cfg.inference.pointcloud.distribution_area,
        cams=cfg.inference.pointcloud.cams,
        freq=cfg.inference.pointcloud.freq,
        side_buffer=cfg.inference.pointcloud.side_buffer
        )

    logger.log('Initilising Optimiser...')
    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': cfg.optimizer.encoding.lr},
            {'name': 'latent_emb', 'params': [model.latent_emb], 'lr': cfg.optimizer.latent_emb.lr},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': cfg.optimizer.net.weight_decay, 'lr': cfg.optimizer.net.lr},
            # {'name': 'rig_pose', 'params': list(renderer.pose_optimise.parameters()), 'lr': 0},
        ], betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

    if cfg.scheduler == 'step':
        lmbda = lambda x: 1
    elif cfg.scheduler == 'exp_decay':
        lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
    else:
        raise ValueError
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    logger.log('Initilising Trainer...')
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

        loss_dist_lambda=cfg.trainer.loss_dist_lambda,
        loss_depth_lambda=cfg.trainer.loss_depth_lambda,
        )

    logger.log('Initiating Training...\n')
    trainer.train()


if __name__ == '__main__':

    train()