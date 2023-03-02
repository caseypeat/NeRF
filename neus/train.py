import hydra
import numpy as np
from pandas import DataFrame
import torch
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf

from loaders.camera_geometry_loader_recon import CameraGeometryLoader
from loaders.synthetic2 import SyntheticLoader

from nets import NeRFCoordinateWrapper, NeRFNetwork
from sdf.nets import NeRFNetworkSecondDerivative
# from neus.nets import NeRFNeusWrapper
from neus.trainer import NeusNeRFTrainer
from nerf.logger import Logger
from metrics import PSNRWrapper, SSIMWrapper, LPIPSWrapper
# from neus.render import NeusRender
from neus.render2 import NeuSRenderer
# from inference import ImageInference, InvdepthThreshInference, PointcloudInference
from neus.inference import ImageInference

# from misc import configurator


def remove_backgrounds(dataloader, max_depth):
    mask = torch.full(dataloader.depths.shape, fill_value=255, dtype=torch.uint8)[..., None]
    mask[dataloader.depths > 1] = 0
    dataloader.images = torch.cat([dataloader.images, mask], dim=-1)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg : DictConfig) -> None:

    # cfg = configurator('./configs/config_base.yaml')

    logger = Logger(
        root_dir=cfg.log.root_dir,
        cfg=cfg,
        )

    logger.log('Initiating Dataloader...')
    dataloader = CameraGeometryLoader(
        scan_paths=cfg.scan.scan_paths,
        scan_pose_paths=cfg.scan.scan_pose_paths,
        frame_ranges=cfg.scan.frame_ranges,
        frame_strides=cfg.scan.frame_strides,
        image_scale=cfg.scan.image_scale,
        load_depths_bool=False,
        )
    dataloader.extrinsics[..., :3, 3] = dataloader.extrinsics[..., :3, 3] - torch.mean(dataloader.extrinsics[..., :3, 3], dim=0, keepdims=True)
    # print(torch.mean(dataloader.extrinsics[..., :3, 3], dim=0, keepdims=True))

    # remove_backgrounds(dataloader, 1)
    # plt.imshow(dataloader.images[0, ... , 3])
    # plt.show()
    # print(dataloader.images.shape)

    # dataloader = SyntheticLoader(rootdir="/home/casey/PhD/data/nerf_datasets/nerf_synthetic/lego")
    # dataloader = SyntheticLoader(rootdir="/home/cpe44/nerf_synthetic/ficus")

    logger.log('Initilising Model...')
    # model = NeRFNetworkSecondDerivative(
    #     N = dataloader.images.shape[0],
    #     encoding_precision=cfg.nets.encoding.precision,
    #     encoding_n_levels=cfg.nets.encoding.n_levels,
    #     encoding_n_features_per_level=cfg.nets.encoding.n_features_per_level,
    #     encoding_log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
    #     geo_feat_dim=cfg.nets.sigma.geo_feat_dim,
    #     sigma_hidden_dim=cfg.nets.sigma.hidden_dim,
    #     sigma_num_layers=cfg.nets.sigma.num_layers,
    #     encoding_dir_precision=cfg.nets.encoding_dir.precision,
    #     encoding_dir_encoding=cfg.nets.encoding_dir.encoding,
    #     encoding_dir_degree=cfg.nets.encoding_dir.degree,
    #     latent_embedding_dim=cfg.nets.latent_embedding.features,
    #     color_hidden_dim=cfg.nets.color.hidden_dim,
    #     color_num_layers=cfg.nets.color.num_layers,
    # ).to('cuda')

    # model_coord = NeRFCoordinateWrapper(
    #     model=model,
    #     transform=None,
    #     inner_bound=cfg.scan.inner_bound,
    #     outer_bound=cfg.scan.outer_bound,
    #     translation_center=dataloader.translation_center
    # ).to('cuda')

    # model_coord_neus = NeRFNeusWrapper(
    #     model = model_coord,
    # ).to('cuda')

    # renderer = NeusRender(
    #     model=model_coord_neus,
    #     steps_firstpass=cfg.renderer.steps,
    #     z_bounds=cfg.renderer.z_bounds,
    #     steps_importance=cfg.renderer.importance_steps,
    #     alpha_importance=cfg.renderer.alpha,
    # ).to('cuda')

    renderer = NeuSRenderer()

    logger.log('Initiating Optimiser...')
    optimizer = torch.optim.Adam([
            {'name': 'sdf_encoding', 'params': list(renderer.sdf_network.encoder.parameters()), 'lr': cfg.optimizer.encoding.lr},
            {'name': 'sdf_net', 'params': list(renderer.sdf_network.network.parameters()), 'lr': cfg.optimizer.net.lr},
            {'name': 'color', 'params': list(renderer.color_network.parameters()), 'lr': cfg.optimizer.net.lr},
            {'name': 'deviation_network', 'params': list(renderer.deviation_network.parameters()), 'lr': 1e-3},
        ], betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

    if cfg.scheduler == 'step':
        lmbda = lambda x: 1
    elif cfg.scheduler == 'exp_decay':
        lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
    elif cfg.scheduler == 'warmup':
        # alpha = 0.05
        # learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        # lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
        def lmbda(iter_step):
            warm_up_end = 500
            end_iter = 300000
            if iter_step < warm_up_end:
                learning_factor = iter_step / warm_up_end
            else:
                alpha = 0.05
                progress = (iter_step - warm_up_end) / (end_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor
    else:
        raise ValueError
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    inferencers = {
        "image": ImageInference(
            renderer, dataloader, cfg.trainer.n_rays, cfg.inference.image.image_num),
    }
    
    logger.log('Initiating Trainer...')
    trainer = NeusNeRFTrainer(
        # model=model,
        dataloader=dataloader,
        logger=logger,
        renderer=renderer,
        inferencers=inferencers,

        optimizer=optimizer,
        scheduler=scheduler, 
        
        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,

        dist_loss_range=cfg.trainer.dist_loss_range,
        depth_loss_range=cfg.trainer.depth_loss_range,

        eval_image_freq=cfg.log.eval_image_freq,
        eval_pointcloud_freq=cfg.log.eval_pointcloud_freq,
        save_weights_freq=cfg.log.save_weights_freq,

        metrics={},
        )

    logger.log('Beginning Training...\n')
    trainer.train()


if __name__ == '__main__':
    train()