import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import os

from omegaconf import OmegaConf, DictConfig

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


if __name__ == '__main__':

    # root_dir = './allign/logs/allign/0004_0009/20220723_180133'
    root_dir = './allign/logs/allign/0005_0008/20220723_144640'
    # root_dir = './allign/logs/allign/0006_0007/20220723_180558'

    pcd_a = o3d.io.read_point_cloud(f'{root_dir}/pcd_ransac_a.pcd')
    pcd_b = o3d.io.read_point_cloud(f'{root_dir}/pcd_ransac_b.pcd')

    init_transform = np.load(f'{root_dir}/init_transform.npy')

    reg_p2p = o3d.pipelines.registration.registration_icp(pcd_a, pcd_b, 0.01, init_transform, o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(init_transform)
    print(reg_p2p.transformation)
    
    np.save(f'{root_dir}/icp_transform.npy', reg_p2p.transformation)