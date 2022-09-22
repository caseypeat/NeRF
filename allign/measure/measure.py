import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import os

from omegaconf import OmegaConf

from tqdm import tqdm

from loaders.camera_geometry_loader import CameraGeometryLoader
from render import NerfRenderer
from nets import NeRFNetwork

from misc import configurator

from allign.ransac import global_allign
from allign.rotation import vec2skew, Exp, matrix2xyz_extrinsic
from allign.trainer import TrainerPose
from allign.extract_dense_geometry import ExtractDenseGeometry
from allign.logger import AllignLogger
from allign.metrics import Measure