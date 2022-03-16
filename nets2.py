import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

import helpers
import raymarching

from renderer import NerfRenderer


class NerfHash(NerfRenderer):
    def __init__(self):
        super().__init__()

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

        return torch.sigmoid(rgb_unflat), F.relu(sigma_unflat)

    def density(self, xyz):
        xyz_flat = xyz.reshape(-1, 3)

        enc_hash = self.encoder_hash(xyz_flat)
        geometry = self.network_sigma(enc_hash)
        sigma, geometry = geometry[..., :1], geometry[..., 1:]

        sigma_unflat = sigma.reshape(*xyz.shape[:-1], 1)

        return F.relu(sigma_unflat)



class NeRFNetwork(NerfRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 cuda_ray=True,
                 ):
        super().__init__()

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    
    def forward(self, x, d, bound):
        # x: [B, N, 3], in [-bound, bound]
        # d: [B, N, 3], nomalized in [-1, 1]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)
        d = d.view(-1, 3)

        # sigma
        x = (x + bound) / (2 * bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
    
        sigma = sigma.view(*prefix)
        color = color.view(*prefix, -1)

        return sigma, color

    def density(self, x, bound):
        # x: [B, N, 3], in [-bound, bound]

        prefix = x.shape[:-1]
        x = x.view(-1, 3)

        x = (x + bound) / (2 * bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        #sigma = torch.exp(torch.clamp(h[..., 0], -15, 15))
        sigma = F.relu(h[..., 0])

        sigma = sigma.view(*prefix)

        return sigma