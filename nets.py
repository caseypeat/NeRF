
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn


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

    def forward(self, x):

        x_flat = x.reshape(x.shape[0]*x.shape[1], 3)
        
        x_enc_hash = self.encoder_hash(x_flat)
        x_geometry = self.network_sigma(x_enc_hash)
        x_sigma, x_geometry = x_geometry[..., :1], x_geometry[..., 1:]

        x_enc_dir = self.encoder_dir(x_flat)
        x_rgb_input = torch.cat([x_enc_dir, x_geometry], dim=-1)
        x_rgb = self.network_rgb(x_rgb_input)

        x_samples = torch.cat((x_rgb, x_sigma), dim=-1)
        x_samples_unflat = x_samples.reshape(x.shape[0], x.shape[1], 4)
        return x_samples_unflat