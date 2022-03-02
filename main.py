from pickletools import optimize
import numpy as np
import torch
import commentjson as json
import tinycudann as tcnn

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR

from tqdm import tqdm

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader


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


if __name__ == '__main__':

    ## Params
    n_samples = 256
    n_rays = 1024
    device = 'cuda'
    # scene_dir = '/local/v100/mnt/maara/conan_scans/blenheim-21-6-28/30-06_13-05/ROW_349_EAST_SLOW_0006/scene.json'
    scene_dir = '/local/v100/mnt/maara/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'

    ## Load Data
    images, intrinsics, extrinsics, depths = camera_geometry_loader(scene_dir, image_scale=1, frame_range=(0, 2))
    N, H, W = images.shape[:3]

    with open('./configs/config.json', 'r') as f:
        config = json.load(f)

    encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config['encoding'])
    network_rgb = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=3, network_config=config['network_rgb'])
    network_sigma = tcnn.Network(n_input_dims=encoding.n_output_dims, n_output_dims=1, network_config=config['network_sigma'])

    model = NeRF(encoding, network_rgb, network_sigma)

    ## Optimiser
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))


    for i in tqdm(range(10000)):

        # n = torch.randint(0, N, (n_rays,), device=device)
        # h = torch.randint(0, H, (n_rays,), device=device)
        # w = torch.randint(0, W, (n_rays,), device=device)

        n = np.random.randint(0, N, (n_rays,))
        h = np.random.randint(0, H, (n_rays,))
        w = np.random.randint(0, W, (n_rays,))

        rgb_gt = images[n, h, w, :]
        invdepth_gt = 1 / depths[n, h, w]
        K = intrinsics[n]
        E = extrinsics[n]
        B = np.zeros((n_rays, 2))
        B[..., 1] = 1

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

        ## Render Pixels
        rgb, invdepth, weights, sigma, color = helpers.render_rays_log(samples, z_vals)

        ## Calculate Error, backpropergate, and update weights
        loss = F.mse_loss(rgb, rgb_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()