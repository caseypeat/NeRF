
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from typing import Optional, Union

from loaders.camera_geometry_loader_re2 import IndexMapping
from rotation import Exp_vector, Exp

from nets import NeRFNetwork, NeRFCoordinateWrapper
from sdf.model.density import LaplaceDensity


class NeRFNetworkSecondDerivative(nn.Module):
    def __init__(
        self,
        N,
        encoding_precision,
        encoding_n_levels,
        encoding_n_features_per_level,
        encoding_log2_hashmap_size,
        geo_feat_dim,
        sigma_hidden_dim,
        sigma_num_layers,
        encoding_dir_precision,
        encoding_dir_encoding,
        encoding_dir_degree,
        latent_embedding_dim,
        color_hidden_dim,
        color_num_layers):
        super().__init__()

        if encoding_precision == 'float16':
            self.encoding_precision = torch.float16
        elif encoding_precision == 'float32':
            self.encoding_precision = torch.float32
        else:
            return ValueError

        # hastable encoding
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": encoding_n_levels,
                "n_features_per_level": encoding_n_features_per_level,
                "log2_hashmap_size": encoding_log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=self.encoding_precision
        )

        # sigma network
        # self.sigma_net = tcnn.Network(
        #     n_input_dims=encoding_n_levels*encoding_n_features_per_level,
        #     n_output_dims=1 + geo_feat_dim,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": sigma_hidden_dim,
        #         "n_hidden_layers": sigma_num_layers - 1,
        #     },
        # )
        self.sigma_net_list = nn.ModuleList()
        self.sigma_net_list.append(nn.Linear(encoding_n_levels*encoding_n_features_per_level, sigma_hidden_dim))
        self.sigma_net_list.append(nn.ReLU())
        for i in range(sigma_num_layers - 1):
            self.sigma_net_list.append(nn.Linear(sigma_hidden_dim, sigma_hidden_dim))
            self.sigma_net_list.append(nn.ReLU())
        self.sigma_net_list.append(nn.Linear(sigma_hidden_dim, 1 + geo_feat_dim))
        self.sigma_net = nn.Sequential(*self.sigma_net_list)

        # directional encoding
        if encoding_dir_degree != 0:
            if encoding_dir_precision == 'float16':
                self.encoding_dir_precision = torch.float16
            elif encoding_dir_precision == 'float32':
                self.encoding_dir_precision = torch.float32
            else:
                return ValueError

            self.encoder_dir = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": encoding_dir_encoding,
                    "degree": encoding_dir_degree,
                },
                dtype=self.encoding_dir_precision,
            )
            self.directional_dim = self.encoder_dir.n_output_dims
        else:
            self.directional_dim = 0

        # camera latent embedding
        self.N = N
        if latent_embedding_dim != 0:
            self.latent_emb_dim = latent_embedding_dim
            self.latent_emb = nn.Parameter(torch.zeros(self.N, self.latent_emb_dim, device='cuda'), requires_grad=True)
        else:
            self.latend_emb_dim = 0

        # color network
        self.in_dim_color = geo_feat_dim + self.directional_dim + self.latent_emb_dim
        # self.color_net = tcnn.Network(
        #     n_input_dims=self.in_dim_color,
        #     n_output_dims=3,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": color_hidden_dim,
        #         "n_hidden_layers": color_num_layers - 1,
        #     },
        # )
        # self.color_net = nn.Sequential(
        #     nn.Linear(self.in_dim_color, color_hidden_dim),
        #     nn.ReLU(),

        #     nn.Linear(color_hidden_dim, 3),
        # )
        self.color_net_list = nn.ModuleList()
        self.color_net_list.append(nn.Linear(self.in_dim_color, color_hidden_dim))
        self.color_net_list.append(nn.ReLU())
        for i in range(color_num_layers - 1):
            self.color_net_list.append(nn.Linear(color_hidden_dim, color_hidden_dim))
            self.color_net_list.append(nn.ReLU())
        self.color_net_list.append(nn.Linear(color_hidden_dim, 3))
        self.color_net = nn.Sequential(*self.color_net_list)


    
    def forward(self, x, d, n):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)
        n = n.reshape(-1)

        # sigma
        x_hashtable = self.encoder(x)

        sigma_network_output = self.sigma_net(x_hashtable)

        # sigma = F.relu(sigma_network_output[..., 0])
        sigma = sigma_network_output[..., 0]
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

        # aux_outputs_net = {}
        # aux_outputs_net['x_hashtable'] = x_hashtable.reshape(*prefix, -1).detach()

        return sigma, color

    def density(self, x):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)

        x_hashtable = self.encoder(x)
        sigma_network_output = self.sigma_net(x_hashtable)

        # sigma = F.relu(sigma_network_output[..., 0])
        sigma = sigma_network_output[..., 0]

        sigma = sigma.reshape(*prefix)

        # aux_outputs_net = {}
        # aux_outputs_net['x_hashtable'] = x_hashtable.reshape(*prefix, -1).detach()

        return sigma


class NeRFSDFWrapper(nn.Module):
    def __init__(self,
        model):
        super().__init__()

        self.model = model
        self.sdf_bounding_sphere = 2.5

        self.laplace_density = LaplaceDensity(
            params_init = {"beta": 0.1},
            beta_min = 0.0001
        ).to('cuda')

    def forward(self, xyzs, dirs, n):
        sdf, color = self.model(xyzs, dirs, n)
        return self.laplace_density(sdf), color

    def density(self, xyzs):
        return self.laplace_density(self.model.density(xyzs))

    def get_sdf_vals(self, xyzs):
        sdf = self.model.density(xyzs)
        # ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        # if self.sdf_bounding_sphere > 0.0:
        #     sphere_sdf = self.sdf_bounding_sphere - xyzs.norm(2,1, keepdim=True)
        #     sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

    def gradient(self, xyzs):
        xyzs.requires_grad_(True)
        y = self.get_sdf_vals(xyzs)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=xyzs,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients