import numpy as np
import torch
import os
import cv2
import time
import yaml

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from misc import color_depthmap

from config import cfg


def tensorboard_test(root_dir):

    log_dir = os.path.join(root_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
    
    writer = SummaryWriter(log_dir=log_dir)

    for i in range(10):
        writer.add_scalar('Test', i**2, i)


class Logger(object):
    def __init__(self, root_dir):
        self.log_dir = os.path.join(root_dir, datetime.today().strftime('%Y%m%d_%H%M%S'))
        self.image_dir = os.path.join(self.log_dir, 'image')
        self.invdepth_dir = os.path.join(self.log_dir, 'invdepth')
        self.pointcloud_dir = os.path.join(self.log_dir, 'pointcloud')
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.tensorboard_dir = os.path.join(self.log_dir, 'tensorboard')

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        os.mkdir(self.log_dir)
        os.mkdir(self.image_dir)
        os.mkdir(self.invdepth_dir)
        os.mkdir(self.pointcloud_dir)
        os.mkdir(self.model_dir)
        os.mkdir(self.tensorboard_dir)

        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        self.log_file = os.path.join(self.log_dir, 'events.log')
        self.t0 = time.time()

        self.scalars = {}

        cfg.save_yaml(os.path.join(self.log_dir, 'config.yaml'))

    def log(self, string):
        with open(self.log_file, 'a') as f:
            dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
            output_str = f'[{dt}] [{time.time()-self.t0:.2f}s] {string}\n'
            print(output_str, end='')
            f.write(output_str)

    def image(self, image, step):
        file_path = os.path.join(self.image_dir, f'{step}.jpg')
        self.writer.add_image('image', image[..., np.array([2, 1, 0], dtype=int)], step, dataformats='HWC')
        cv2.imwrite(file_path, np.uint8(image*255))

    def invdepth(self, invdepth, step):
        invdepth_c = color_depthmap(invdepth)
        file_path = os.path.join(self.invdepth_dir, f'{step}.jpg')
        self.writer.add_image('invdepth', invdepth_c, step, dataformats='HWC')
        cv2.imwrite(file_path, np.uint8(invdepth_c[..., np.array([2, 1, 0], dtype=int)]*255))

    def pointcloud(self, pointcloud, step):
        file_path = os.path.join(self.pointcloud_dir, f'{step}.npy')
        np.save(file_path, pointcloud)

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
