import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from renderer import NerfRenderer


class NeRFNetwork(NerfRenderer):
    def __init__(self,
                # encoding (hashgrid)
                n_levels=18,
                n_features_per_level=2,
                log2_hashmap_size=24,
                encoding_precision='float32',

                # directional encoding
                encoding_dir="SphericalHarmonics",
                encoding_dir_degree=4,
                encoding_dir_precision='float32',

                # sigma network
                num_layers=2,
                hidden_dim=64,
                geo_feat_dim=15,
                 
                # color network
                num_layers_color=3,
                hidden_dim_color=64,

                **kwargs,
                ):
        super().__init__(**kwargs)

        if encoding_precision == 'float16':
            self.encoding_precision = torch.float16
        elif encoding_precision == 'float32':
            self.encoding_precision = torch.float32
        else:
            return ValueError


        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=self.encoding_precision
        )

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        self.sigma_net = tcnn.Network(
            n_input_dims=n_levels*n_features_per_level,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        if encoding_dir_precision == 'float16':
            self.encoding_dir_precision = torch.float16
        elif encoding_dir_precision == 'float32':
            self.encoding_dir_precision = torch.float32
        else:
            return ValueError

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": encoding_dir,
                "degree": encoding_dir_degree,
            },
            dtype=self.encoding_dir_precision,
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

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

    
    def forward(self, x, d):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
    
        sigma = sigma.reshape(*prefix)
        color = color.reshape(*prefix, -1)

        return sigma, color

    def density(self, x):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        sigma = F.relu(h[..., 0])

        sigma = sigma.reshape(*prefix)

        return sigma