import numpy as np
import torch
import open3d as o3d
import os
import cv2
import time

from datetime import datetime
from omegaconf import OmegaConf
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter


class AllignLogger(object):
    def __init__(self, root_dir, cfg):
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
        self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')

        self.cfg = cfg

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        os.mkdir(self.log_dir)
        os.mkdir(self.tensorboard_dir)

        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        self.log_file = os.path.join(self.log_dir, 'events.log')
        self.t0 = time.time()

        self.scalars = {}

        OmegaConf.save(config=self.cfg, f=os.path.join(self.log_dir, 'config.yaml'))

    def log(self, string):
        with open(self.log_file, 'a') as f:
            dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
            output_str = f'[{dt}] [{time.time()-self.t0:.2f}s] {string}\n'
            output_str_ndt = f'[{time.time()-self.t0:.2f}s] {string}\n'
            print(output_str_ndt, end='')
            f.write(output_str)

    def scalar(self, name, value, step):
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            value = value.item()
        self.writer.add_scalar(name, value, step)
        if name not in self.scalars.keys():
            self.scalars[name] = []
        self.scalars[name].append(value)

    def save_transform(self, transform, filename):
        np.save(f'{self.log_dir}/{filename}.npy', transform)

    def save_pointcloud(self, pointcloud, filename):
        o3d.io.write_point_cloud(f'{self.log_dir}/{filename}.pcd', pointcloud)

    def save_pointclouds_comb(self, pointcloud_a, pointcloud_b, transform, filename):
        pcd_a = deepcopy(pointcloud_a)
        pcd_b = deepcopy(pointcloud_b)

        pcd_a.transform(transform)

        pcd_both = o3d.geometry.PointCloud()
        pcd_both += pcd_a
        pcd_both += pcd_b
        o3d.io.write_point_cloud(f'{self.log_dir}/{filename}.pcd', pcd_both)