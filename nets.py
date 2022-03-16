
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

import helpers
import raymarching

from tqdm import tqdm


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

        # print(torch.max(xyz), torch.min(xyz))
        # print(torch.max(dir), torch.min(dir))
        # print(xyz)
        # exit()

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
        return torch.sigmoid(rgb_unflat), F.relu(sigma_unflat)

    def density(self, xyz):
        xyz_flat = xyz.reshape(-1, 3)

        enc_hash = self.encoder_hash(xyz_flat)
        geometry = self.network_sigma(enc_hash)
        sigma, geometry = geometry[..., :1], geometry[..., 1:]

        sigma_unflat = sigma.reshape(*xyz.shape[:-1], 1)

        return F.relu(sigma_unflat)


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
        print(xyz.shape, dir.shape)
        exit()
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

        # print(rays_o.shape, rays_d.shape)

        # setup counter
        counter = self.step_counter[self.local_step % 64]
        counter.zero_() # set to 0
        self.local_step += 1

        xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, bound, self.density_grid, self.mean_density, self.iter_density, counter, self.mean_count, perturb, 128, False)

        # print(xyzs.shape, dirs.shape)

        rgbs, sigmas = self.nerf(xyzs, dirs)
        # print(rgbs.shape, sigmas.shape)
        depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays, bound, bg_color)
        # print(depth.shape, image.shape)
        # exit()

        return image, depth

    def inference(self, n, h, w):
        with torch.no_grad():
            max_batch_size = 1024

            n_f = torch.reshape(n, (-1,))
            h_f = torch.reshape(h, (-1,))
            w_f = torch.reshape(w, (-1,))

            image = torch.zeros((*n_f.shape, 3))
            depth = torch.zeros(n_f.shape)

            for i in tqdm(range(0, len(n_f), max_batch_size)):
                end = min(len(n_f), i+max_batch_size)

                n_fb = n_f[i:end]
                h_fb = h_f[i:end]
                w_fb = w_f[i:end]

                image_fb, depth_fb = self(n_fb, h_fb, w_fb)

                image[i:end] = image_fb
                depth[i:end] = depth_fb

            image_uf = torch.reshape(image, (*n.shape, 3))
            depth_uf = torch.reshape(depth, n.shape)

            return image_uf, depth_uf

    def update_extra_state(self, bound, decay=0.95):
        # call before each epoch to update extra states.
        
        ### update density grid
        resolution = self.density_grid.shape[0]

        half_grid_size = bound / resolution
        
        X = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)
        Y = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)
        Z = torch.linspace(-bound + half_grid_size, bound - half_grid_size, resolution).split(128)

        tmp_grid = torch.zeros_like(self.density_grid)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        lx, ly, lz = len(xs), len(ys), len(zs)
                        # construct points
                        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3]
                        # add noise in [-hgs, hgs]
                        pts += (torch.rand_like(pts) * 2 - 1) * half_grid_size
                        # manual padding for ffmlp
                        n = pts.shape[0]
                        pad_n = 128 - (n % 128)
                        if pad_n != 0:
                            pts = torch.cat([pts, torch.zeros(pad_n, 3)], dim=0)
                        # query density
                        density = self.nerf.density(pts.to(tmp_grid.device))[:n].reshape(lx, ly, lz).detach()
                        tmp_grid[xi * 128: xi * 128 + lx, yi * 128: yi * 128 + ly, zi * 128: zi * 128 + lz] = density
        
        # ema update
        self.density_grid = torch.maximum(self.density_grid * decay, tmp_grid)
        self.mean_density = torch.mean(self.density_grid).item()
        self.iter_density += 1

        ### update step counter
        total_step = min(64, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0