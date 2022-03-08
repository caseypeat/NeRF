
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

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader
from loaders.synthetic import load_image_set

from nets import NerfHash, NerfRenderOccupancy, NerfRender


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
        scene_path = '/home/casey/Datasets/NeRF_Data/nerf_synthetic/lego'
        images, depths, intrinsics, extrinsics, bds = load_image_set(scene_path, near=2, far=6, scale=0.04)
    elif loader == 'camera_geometry':
        scene_dir = '/mnt/maara/synthetic_tree_assets/trees3/renders/vine_C2_1/back_close/cameras.json'
        images, depths, intrinsics, extrinsics, bds = camera_geometry_loader(scene_dir, image_scale=1, frame_range=(0, 2))

    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)
    bds = torch.Tensor(bds)

    return images, depths, intrinsics, extrinsics, bds


if __name__ == '__main__':

    ## Params
    n_samples = 256
    n_rays = 1024
    device = 'cuda'

    images, depths, intrinsics, extrinsics, bds = meta_loader('synthetic')
    N, H, W = images.shape[:3]

    # model = NerfRenderOccupancy(images, depths, intrinsics, extrinsics, bds).to('cuda')
    model = NerfRender(images, depths, intrinsics, extrinsics, bds).to('cuda')

    ## Optimiser
    optimizer = Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))


    for i in tqdm(range(10000+1)):

        # n = np.random.randint(0, N, (n_rays,))
        # h = np.random.randint(0, H, (n_rays,))
        # w = np.random.randint(0, W, (n_rays,))

        # n = torch.Tensor(n).to(torch.long)
        # h = torch.Tensor(h).to(torch.long)
        # w = torch.Tensor(w).to(torch.long)

        n = torch.randint(0, N, (n_rays,))
        h = torch.randint(0, H, (n_rays,))
        w = torch.randint(0, W, (n_rays,))

        scaler = GradScaler()

        ## Network Inference
        with autocast():

            optimizer.zero_grad()
            
            image_gt = images[n, h, w].to('cuda')

            image, depth = model(n, h, w)

            ## Calculate Error
            loss = F.mse_loss(image, image_gt)

            # loss.backward()
            # optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if i%100 == 0:
            print(loss)

        # if i%10000 == 0 and i != 0:
        #     with torch.no_grad():
                
        #         n = np.full((1,), 1)
        #         h = np.arange(0, H)
        #         w = np.arange(0, W)

        #         n_m, h_m, w_m = np.meshgrid(n, h, w)

        #         n_mf = np.reshape(n_m, (-1,))
        #         h_mf = np.reshape(h_m, (-1,))
        #         w_mf = np.reshape(w_m, (-1,))

        #         rgb_flat = torch.Tensor(np.zeros((H*W, 3)))

        #         for i in tqdm(range(0, len(n_mf), n_rays)):

        #             if i+n_rays < len(n_mf):
        #                 end = i+n_rays
        #             else:
        #                 end = len(n_mf) - 1

        #             n_mfb, h_mfb, w_mfb = n_mf[i: end], h_mf[i: end], w_mf[i: end]

        #             K = intrinsics[n_mfb]
        #             E = extrinsics[n_mfb]
        #             B = bds[n_mfb]

        #             n_mfb = torch.Tensor(n_mfb).to(device)
        #             h_mfb = torch.Tensor(h_mfb).to(device)
        #             w_mfb = torch.Tensor(w_mfb).to(device)

        #             K = torch.Tensor(K).to(device)
        #             E = torch.Tensor(E).to(device)
        #             B = torch.Tensor(B).to(device)

        #             ## Auxilary Ray Calculations
        #             rays_o, rays_d = helpers.get_rays(h_mfb, w_mfb, K, E)
        #             z_vals = helpers.get_z_vals_log(B[..., 0], B[..., 1], len(rays_o), n_samples, device)
        #             xyz, dir = helpers.get_sample_points(rays_o, rays_d, z_vals)
                    
        #             ## Network Inference
        #             samples = model(xyz)

        #             rgb, invdepth, weights, sigma, color = helpers.render_rays_log(samples, z_vals)

        #             rgb_flat[i: end] = rgb

        #         rgb = torch.reshape(rgb_flat, (H, W, 3)).cpu().numpy()

        #         plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        #         plt.show()