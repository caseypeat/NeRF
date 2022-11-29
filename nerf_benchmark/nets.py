import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

class NeRFNetworkPartial(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding_precision = torch.float16

        self.encoding_n_levels = 20
        self.encoding_n_features_per_level = 2
        self.encoding_log2_hashmap_size = 24

        self.network_n_neurons = 64
        self.network_n_hidden_layers = 1

        # hastable encoding
        self.encoder_network = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=4,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.encoding_n_levels,
                "n_features_per_level": self.encoding_n_features_per_level,
                "log2_hashmap_size": self.encoding_log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.network_n_neurons,
                "n_hidden_layers": self.network_n_hidden_layers,
            },
            # dtype=self.encoding_precision,
        )

    def forward(self, x):
        y_ = self.encoder_network(x)
        return y_



class NeRFNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding_precision = torch.float16

        self.encoding_n_levels = 20
        self.encoding_n_features_per_level = 2
        self.encoding_log2_hashmap_size = 24

        self.sigma_n_neurons = 64
        self.sigma_n_hidden_layers = 1

        self.color_n_neurons = 64
        self.color_n_hidden_layers = 2

        # hastable encoding
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.encoding_n_levels,
                "n_features_per_level": self.encoding_n_features_per_level,
                "log2_hashmap_size": self.encoding_log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=self.encoding_precision
        )

        # sigma network
        self.sigma_net = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=1 + 15,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.sigma_n_neurons,
                "n_hidden_layers": self.sigma_n_hidden_layers,
            },
        )

        self.color_net = tcnn.Network(
            n_input_dims=15,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.color_n_neurons,
                "n_hidden_layers": self.color_n_hidden_layers,
            },
        )

    def forward(self, x):
        e = self.encoder(x)
        s_o = self.sigma_net(e)
        s, g = s_o[:, :1], s_o[:, 1:]
        c = self.color_net(g)
        y_ = torch.cat([s, c], dim=-1)
        return y_