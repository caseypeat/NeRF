
import numpy as np
import torch
import cv2
import commentjson as json
import tinycudann as tcnn
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR

from tqdm import tqdm

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader
from loaders.synthetic import load_image_set


class NeRF(torch.nn.Module):
    def __init__(self, encoding, network_rgb, network_sigma):
        super(NeRF, self).__init__()

        self.encoding = encoding
        self.network_rgb = network_rgb
        self.network_sigma = network_sigma

    def forward(self, x):

        x_flat = x.reshape(x.shape[0]*x.shape[1], 3)
        
        x_enc = self.encoding(x_flat)
        x_rgb = self.network_rgb(x_enc)
        x_sigma = self.network_sigma(x_enc)

        x_samples = torch.cat((x_rgb, x_sigma), dim=-1)

        x_samples_unflat = x_samples.reshape(x.shape[0], x.shape[1], 4)

        return x_samples_unflat


def meta_loader(loader):
    if loader == 'synthetic':
        scene_path = '/local/v100/home/casey/Datasets/NeRF_Data/nerf_synthetic/lego'
        images, depths, intrinsics, extrinsics, bds = load_image_set(scene_path, near=2, far=6, scale=0.04)
    elif loader == 'camera_geometry':
        scene_dir = '/local/v100/mnt/maara/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'
        images, depths, intrinsics, extrinsics, bds = camera_geometry_loader(scene_dir, image_scale=1, frame_range=(0, 2))

    return images, depths, intrinsics, extrinsics, bds


if __name__ == '__main__':

    ## Params
    n_samples = 128
    n_rays = 1024
    device = 'cuda'
    # scene_dir = '/local/v100/mnt/maara/conan_scans/blenheim-21-6-28/30-06_13-05/ROW_349_EAST_SLOW_0006/scene.json'
    # scene_dir = '/local/v100/mnt/maara/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'

    ## Load Data
    # images, intrinsics, extrinsics, depths = camera_geometry_loader(scene_dir, image_scale=1, frame_range=(0, 2))

    images, depths, intrinsics, extrinsics, bds = meta_loader('synthetic')
    N, H, W = images.shape[:3]

    with open('./configs/config.json', 'r') as f:
        config = json.load(f)

    encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config['encoding'])
    network_rgb = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=3, network_config=config['network_rgb'])
    network_sigma = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=1, network_config=config['network_sigma'])

    model = NeRF(encoding, network_rgb, network_sigma)

    ## Optimiser
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))


    for i in tqdm(range(100000)):

        n = np.random.randint(0, N, (n_rays,))
        h = np.random.randint(0, H, (n_rays,))
        w = np.random.randint(0, W, (n_rays,))

        rgb_gt = images[n, h, w, :]
        invdepth_gt = 1 / depths[n, h, w]
        K = intrinsics[n]
        E = extrinsics[n]
        B = bds[n]

        n = torch.Tensor(n).to(device)
        h = torch.Tensor(h).to(device)
        w = torch.Tensor(w).to(device)

        rgb_gt = torch.Tensor(rgb_gt).to(device)
        invdepth_gt = torch.Tensor(invdepth_gt).to(device)
        K = torch.Tensor(K).to(device)
        E = torch.Tensor(E).to(device)
        B = torch.Tensor(B).to(device)

        ## Auxilary Ray Calculations
        rays_o, rays_d = helpers.get_rays(h, w, K, E)
        z_vals = helpers.get_z_vals_log(B[..., 0], B[..., 1], len(rays_o), n_samples, device)
        xyz, dir = helpers.get_sample_points(rays_o, rays_d, z_vals)
        
        ## Network Inference
        samples = model(xyz)
        samples[..., 3] += torch.normal(mean=samples.new_zeros(samples.shape[:-1]), std=samples.new_ones(samples.shape[:-1]) * 0.1)

        ## Render Pixels
        rgb, invdepth, weights, sigma, color = helpers.render_rays_log(samples, z_vals)

        ## Calculate Error, backpropergate, and update weights
        loss = F.mse_loss(rgb, rgb_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%1000 == 0:
            print(loss)

        if i%20000 == 0 and i != 0:
            with torch.no_grad():
                
                n = np.full((1,), 1)
                h = np.arange(0, H)
                w = np.arange(0, W)

                n_m, h_m, w_m = np.meshgrid(n, h, w)

                n_mf = np.reshape(n_m, (-1,))
                h_mf = np.reshape(h_m, (-1,))
                w_mf = np.reshape(w_m, (-1,))

                rgb_flat = torch.Tensor(np.zeros((H*W, 3)))

                for i in tqdm(range(0, len(n_mf), n_rays)):

                    if i+n_rays < len(n_mf):
                        end = i+n_rays
                    else:
                        end = len(n_mf) - 1

                    n_mfb, h_mfb, w_mfb = n_mf[i: end], h_mf[i: end], w_mf[i: end]

                    K = intrinsics[n_mfb]
                    E = extrinsics[n_mfb]
                    B = bds[n_mfb]

                    n_mfb = torch.Tensor(n_mfb).to(device)
                    h_mfb = torch.Tensor(h_mfb).to(device)
                    w_mfb = torch.Tensor(w_mfb).to(device)

                    K = torch.Tensor(K).to(device)
                    E = torch.Tensor(E).to(device)
                    B = torch.Tensor(B).to(device)

                    ## Auxilary Ray Calculations
                    rays_o, rays_d = helpers.get_rays(h_mfb, w_mfb, K, E)
                    z_vals = helpers.get_z_vals_log(B[..., 0], B[..., 1], len(rays_o), n_samples, device)
                    xyz, dir = helpers.get_sample_points(rays_o, rays_d, z_vals)
                    
                    ## Network Inference
                    samples = model(xyz)

                    rgb, invdepth, weights, sigma, color = helpers.render_rays_log(samples, z_vals)

                    rgb_flat[i: end] = rgb

                rgb = torch.reshape(rgb_flat, (H, W, 3)).cpu().numpy()

                plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                plt.show()