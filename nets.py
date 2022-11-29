
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from typing import Optional, Union

from loaders.camera_geometry_loader_re2 import IndexMapping
from rotation import Exp_vector, Exp

# from renderer import NerfRenderer

# from config import cfg


class NeRFNetwork(nn.Module):
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
        self.sigma_net = tcnn.Network(
            n_input_dims=encoding_n_levels*encoding_n_features_per_level,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": sigma_hidden_dim,
                "n_hidden_layers": sigma_num_layers - 1,
            },
        )

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
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": color_hidden_dim,
                "n_hidden_layers": color_num_layers - 1,
            },
        )

    # @torch.no_grad()
    def forward(self, x, d, n):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)
        n = n.reshape(-1)

        # sigma
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

        # aux_outputs_net = {}
        # aux_outputs_net['x_hashtable'] = x_hashtable.reshape(*prefix, -1).detach()

        return sigma, color

    def density(self, x):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)

        x_hashtable = self.encoder(x)
        sigma_network_output = self.sigma_net(x_hashtable)

        sigma = F.relu(sigma_network_output[..., 0])

        sigma = sigma.reshape(*prefix)

        # aux_outputs_net = {}
        # aux_outputs_net['x_hashtable'] = x_hashtable.reshape(*prefix, -1).detach()

        return sigma
    

class Transform(nn.Module):
    def __init__(self, init_transform):
        super().__init__()

        self.init_transform = torch.nn.Parameter(init_transform, requires_grad=False)

        self.R = torch.nn.Parameter(torch.zeros((3,)), requires_grad=True)
        self.T = torch.nn.Parameter(torch.zeros((3,)), requires_grad=True)

    def forward(self, xyzs_a):

        transform = torch.eye(4, dtype=torch.float32, device='cuda')
        transform[:3, :3] = Exp(self.R)
        transform[:3, 3] = self.T

        b = xyzs_a.reshape(-1, 3)
        b1 = torch.cat([b, b.new_ones((b.shape[0], 1))], dim=1)
        b2 = (self.get_matrix() @ b1.T).T
        # b2 = (self.init_transform @ b1.T).T
        # b3 = (transform @ b2.T).T
        xyzs_b = b2[:, :3].reshape(*xyzs_a.shape)

        return xyzs_b

    def get_matrix(self):
        transform_local = torch.eye(4, dtype=torch.float32, device='cuda')
        transform_local[:3, :3] = Exp(self.R)
        transform_local[:3, 3] = self.T
        
        transform = self.init_transform @ transform_local
        return transform


class OptimizePose(nn.Module):
    def __init__(self, index_mapping:IndexMapping, mode:str):
        super().__init__()

        self.index_mapping = index_mapping
        self.mode = mode

        if self.mode == 'scan':
            self.num_poses = self.index_mapping.get_num_scans()
        if self.mode == 'rig':
            self.num_poses = self.index_mapping.get_num_rigs()
        if self.mode == 'camera':
            self.num_poses = self.index_mapping.get_num_cams()

        self.R = torch.nn.Parameter(torch.zeros((self.num_poses, 3)), requires_grad=True)
        self.T = torch.nn.Parameter(torch.zeros((self.num_poses, 3)), requires_grad=True)

    def forward(self, n:torch.LongTensor, E:torch.Tensor):
        if self.mode == 'scan':
            pose = self.index_mapping.idx_to_src(n)[0]
        if self.mode == 'rig':
            pose = self.index_mapping.idx_to_rig(n)
        if self.mode == 'camera':
            pose = torch.clone(n)

        pose_transforms = torch.zeros((len(n), 4, 4))
        pose_transforms[:, 3, 3] = 1
        pose_transforms[:, :3, :3] = Exp_vector(self.R[pose])
        pose_transforms[:, :3, 3] = self.T[pose]

        E_transformed = pose_transforms @ E

        return E_transformed


def mipnerf360_scale(xyzs, bound_inner, bound_outer):
    if len(xyzs.shape) == 2:
        d = torch.linalg.norm(xyzs, dim=-1)[..., None].expand(-1, 3)
    else:
        d = torch.linalg.norm(xyzs, dim=-1)[..., None].expand(-1, -1, 3)
    xyzs_warped = torch.clone(xyzs)
    xyzs_warped[d > bound_inner] = xyzs_warped[d > bound_inner] * ((bound_outer - (bound_outer - bound_inner) / d[d > bound_inner]) / d[d > bound_inner])
    return xyzs_warped


class NeRFCoordinateWrapper(nn.Module):
    def __init__(self, 
        model:NeRFNetwork,
        transform:Optional[Transform],
        inner_bound:int,
        outer_bound:int,
        translation_center:torch.Tensor):
        super().__init__()

        self.model = model
        self.transform = transform

        self.inner_bound = inner_bound
        self.outer_bound = outer_bound
        self.translation_center = translation_center

    def forward(self, xyzs, dirs, n):
        if self.transform is None:
            xyzs_transform = xyzs
        else:
            xyzs_transform = self.transform(xyzs)

        xyzs_centered = xyzs_transform - self.translation_center.to(xyzs.device)
        xyzs_warped = mipnerf360_scale(xyzs_centered, self.inner_bound, self.outer_bound)
        xyzs_norm = (xyzs_warped + self.outer_bound) / (2 * self.outer_bound) # to [0, 1]

        sigmas, rgbs = self.model(xyzs_norm, dirs, n)

        return sigmas, rgbs

    def density(self, xyzs):
        if self.transform is None:
            xyzs_transform = xyzs
        else:
            xyzs_transform = self.transform(xyzs)

        xyzs_centered = xyzs_transform - self.translation_center.to(xyzs.device)
        xyzs_warped = mipnerf360_scale(xyzs_centered, self.inner_bound, self.outer_bound)
        xyzs_norm = (xyzs_warped + self.outer_bound) / (2 * self.outer_bound) # to [0, 1]

        sigmas = self.model.density(xyzs_norm)

        return sigmas
