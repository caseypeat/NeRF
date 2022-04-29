import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from renderer import NerfRenderer

from config import cfg


class NeRFNetwork(NerfRenderer):
    def __init__(self, N, **kwargs):
        super().__init__(**kwargs)

        if cfg.nets.encoding.precision == 'float16':
            self.encoding_precision = torch.float16
        elif cfg.nets.encoding.precision == 'float32':
            self.encoding_precision = torch.float32
        else:
            return ValueError

        # hastable encoding
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": cfg.nets.encoding.n_levels,
                "n_features_per_level": cfg.nets.encoding.n_features_per_level,
                "log2_hashmap_size": cfg.nets.encoding.log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=self.encoding_precision
        )

        # sigma network
        self.sigma_net = tcnn.Network(
            n_input_dims=cfg.nets.encoding.n_levels*cfg.nets.encoding.n_features_per_level,
            n_output_dims=1 + cfg.nets.sigma.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": cfg.nets.sigma.hidden_dim,
                "n_hidden_layers": cfg.nets.sigma.num_layers - 1,
            },
        )

        # directional encoding
        if cfg.nets.encoding_dir.degree != 0:
            if cfg.nets.encoding_dir.precision == 'float16':
                self.encoding_dir_precision = torch.float16
            elif cfg.nets.encoding_dir.precision == 'float32':
                self.encoding_dir_precision = torch.float32
            else:
                return ValueError

            self.encoder_dir = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": cfg.nets.encoding_dir.encoding,
                    "degree": cfg.nets.encoding_dir.degree,
                },
                dtype=self.encoding_dir_precision,
            )
            self.directional_dim = self.encoder_dir.n_output_dims
        else:
            self.directional_dim = 0

        # Camera latent embedding
        self.N = N
        if cfg.nets.latent_embedding.features != 0:
            self.latent_emb_dim = cfg.nets.latent_embedding.features
            self.latent_emb = nn.Parameter(torch.zeros(self.N, self.latent_emb_dim, device='cuda'), requires_grad=True)
        else:
            self.latend_emb_dim = 0

        # color network
        self.in_dim_color = cfg.nets.sigma.geo_feat_dim + self.directional_dim + self.latent_emb_dim
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": cfg.nets.color.hidden_dim,
                "n_hidden_layers": cfg.nets.color.num_layers - 1,
            },
        )

    
    def forward(self, x, d, n, **kwargs):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)
        n = n.reshape(-1)

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_hashtable = self.encoder(x)

        sigma_network_output = self.sigma_net(x_hashtable)

        sigma = F.relu(sigma_network_output[..., 0])
        geo_feat = sigma_network_output[..., 1:]

        # color
        color_network_input = geo_feat
        if self.directional_dim != 0:
            d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.encoder_dir(d)
            color_network_input = torch.cat([color_network_input, d], dim=-1)
        if self.latent_emb_dim != 0:
            l = self.latent_emb[n]
            color_network_input = torch.cat([color_network_input, l], dim=1)

        color_network_output = self.color_net(color_network_input)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(color_network_output)
    
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