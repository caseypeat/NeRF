
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

import helpers
import raymarching

class NerfHash(torch.nn.Module):
    def __init__(self):
        super(NerfHash, self).__init__()

        self.geo_feat_dim = 16

        # sigma (density) network
        self.encoder_hash = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            })

        self.network_sigma = tcnn.Network(
            n_input_dims=self.encoder_hash.n_output_dims,
            n_output_dims=1+self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            })

        # color network
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.network_rgb = tcnn.Network(
            n_input_dims=self.encoder_dir.n_output_dims + self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            })

    def forward(self, xyz, dir):

        xyz_flat = xyz.reshape(-1, 3)
        dir_flat = dir.reshape(-1, 3)
        
        enc_hash = self.encoder_hash(xyz_flat)
        geometry = self.network_sigma(enc_hash)
        sigma, geometry = geometry[..., :1], geometry[..., 1:]

        dir_enc = self.encoder_dir(dir_flat)
        rgb_input = torch.cat([dir_enc, geometry], dim=-1)
        rgb = self.network_rgb(rgb_input)

        rgb_unflat = rgb.reshape(*xyz.shape)
        sigma_unflat = sigma.reshape(*xyz.shape[:-1], 1)

        # samples = torch.cat((rgb, sigma), dim=-1)
        # samples_unflat = samples.reshape(*xyz.shape[:-1], 4)
        return rgb_unflat, sigma_unflat


class NerfRender(nn.Module):
    def __init__(self, images, depths, intrinsics, extrinsics, bds):
        super(NerfRender, self).__init__()

        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.bds = bds

        self.nerf = NerfHash()

    def forward(self, n, h, w):

        K = self.intrinsics[n]
        E = self.extrinsics[n]
        B = self.bds[n]

        h = h.to('cuda')
        w = w.to('cuda')
        K = K.to('cuda')
        E = E.to('cuda')
        B = B.to('cuda')

        rays_o, rays_d = helpers.get_rays(h, w, K, E)

        z_vals = helpers.get_z_vals_log(B[..., 0], B[..., 1], len(rays_o), 256, 'cuda')
        xyz, dir = helpers.get_sample_points(rays_o, rays_d, z_vals)

        # xyz, dir = xyz.to('cuda'), dir.to('cuda')
        # z_vals = z_vals.to('cuda')
        
        ## Network Inference
        image, depth = self.nerf(xyz, dir)
        samples = torch.cat([image, depth], dim=-1)
        samples[..., 3] += torch.normal(mean=samples.new_zeros(samples.shape[:-1]), std=samples.new_ones(samples.shape[:-1]) * 0.1)
        # print(samples.shape)
        # exit()

        ## Render Pixels
        image, invdepth, weights, sigma, color = helpers.render_rays_log(samples, z_vals)

        return image, invdepth


class NerfRenderOccupancy(nn.Module):
    def __init__(self, images, depths, intrinsics, extrinsics, bds):
        super(NerfRenderOccupancy, self).__init__()

        self.images = images
        self.depths = depths
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.bds = bds

        self.nerf = NerfHash()

        density_grid = torch.zeros([128] * 3)
        self.register_buffer('density_grid', density_grid)
        self.mean_density = 0
        self.iter_density = 0

        step_counter = torch.zeros(64, 2, dtype=torch.int32)
        self.register_buffer('step_counter', step_counter)
        self.mean_count = 0
        self.local_step = 0

    def forward(self, n, h, w):

        K = self.intrinsics[n]
        E = self.extrinsics[n]
        B = self.bds[n]

        bg_color = torch.zeros((3,)).to('cuda')
        bound = 1
        perturb = True

        rays_o, rays_d = helpers.get_rays(h, w, K, E)

        rays_o = rays_o.to('cuda')
        rays_d = rays_d.to('cuda')

        # setup counter
        counter = self.step_counter[self.local_step % 64]
        counter.zero_() # set to 0
        self.local_step += 1

        xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, perturb, 128, False)

        rgbs, sigmas = self.nerf(xyzs, dirs)
        depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound, bg_color)

        return image, depth