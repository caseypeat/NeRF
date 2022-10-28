import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from typing import Optional, Union

from loaders.camera_geometry_loader_re2 import IndexMapping
from rotation import Exp_vector, Exp

from nets import NeRFNetwork, NeRFCoordinateWrapper


class NeRFNeusWrapper(nn.Module):
    def __init__(self,
        model):
        super().__init__()

        self.model = model

    def forward(self, xyzs, dirs, n):
        sdf, color = self.model(xyzs, dirs, n)
        return sdf, color

    def sdf(self, xyzs):
        return self.model.density(xyzs)

    def gradient(self, xyzs):
        xyzs.requires_grad_(True)
        y = self.sdf(xyzs)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=xyzs,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients