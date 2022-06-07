import numpy as np
import torch
import open3d as o3d
import os
import cv2
import time

from datetime import datetime
from omegaconf import OmegaConf

from torch.utils.tensorboard import SummaryWriter

from misc import color_depthmap

# from config import cfg


def tensorboard_test(root_dir):

    log_dir = os.path.join(root_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
    
    writer = SummaryWriter(log_dir=log_dir)

    for i in range(10):
        writer.add_scalar('Test', i**2, i)


class Logger(object):
    def __init__(self, root_dir, cfg):
        self.root_dir = root_dir
        self.log_dir = os.path.join(root_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
        self.images_dir = os.path.join(self.log_dir, 'images')
        self.pointcloud_dir = os.path.join(self.log_dir, 'pointclouds')
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')

        self.cfg = cfg

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        os.mkdir(self.log_dir)
        os.mkdir(self.images_dir)
        os.mkdir(self.pointcloud_dir)
        os.mkdir(self.model_dir)
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
            print(output_str, end='')
            f.write(output_str)

    def image_color(self, string, image, step):
        image = image.transpose(1, 0, 2)[:, ::-1]
        # image = image.transpose(1, 0, 2)[::-1]
        directory_path = os.path.join(self.images_dir, string)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        file_path = os.path.join(directory_path, f'{step}.jpg')
        self.writer.add_image(string, image[..., np.array([2, 1, 0], dtype=int)], step, dataformats='HWC')
        cv2.imwrite(file_path, np.uint8(image*255))

    def image_grey(self, string, grey, step):
        grey_c = color_depthmap(grey)
        grey_c = grey_c.transpose(1, 0, 2)[:, ::-1]
        # grey_c = grey_c.transpose(1, 0, 2)[::-1]
        directory_path = os.path.join(self.images_dir, string)
        if not os.path.exists(directory_path):
            os.mkdir(directory_path)
        file_path = os.path.join(directory_path, f'{step}.jpg')
        self.writer.add_image(string, grey_c, step, dataformats='HWC')
        cv2.imwrite(file_path, np.uint8(grey_c[..., np.array([2, 1, 0], dtype=int)]*255))

    def pointcloud(self, pointcloud, step):
        directory_path_np = os.path.join(self.pointcloud_dir, 'numpy')
        if not os.path.exists(directory_path_np):
            os.mkdir(directory_path_np)
        file_path_np = os.path.join(directory_path_np, f'{step}.npy')
        np.save(file_path_np, pointcloud)
        
        directory_path_pcd = os.path.join(self.pointcloud_dir, 'pcd')
        if not os.path.exists(directory_path_pcd):
            os.mkdir(directory_path_pcd)
        file_path_pcd = os.path.join(directory_path_pcd, f'{step}.pcd')

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud['points'][np.broadcast_to(pointcloud['depth_variance'], pointcloud['points'].shape) < 0.02].reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(pointcloud['colors'][np.broadcast_to(pointcloud['depth_variance'], pointcloud['colors'].shape) < 0.02].reshape(-1, 3))
        o3d.io.write_point_cloud(file_path_pcd, pcd)

    def model(self, model, step):
        file_path = os.path.join(self.model_dir, f'{step}.pth')
        torch.save(model.state_dict(), file_path)

    def scalar(self, name, value, step):
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            value = value.item()
        self.writer.add_scalar(name, value, step)
        if name not in self.scalars.keys():
            self.scalars[name] = []
        self.scalars[name].append(value)


if __name__ == '__main__':
    tensorboard_test('./logs/test')
