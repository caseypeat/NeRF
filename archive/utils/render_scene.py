from logging import root
import numpy as np
import torch
import cv2
import commentjson as json
import yaml
import os
import matplotlib.pyplot as plt
import math as m

from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD, LBFGS
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, MultiStepLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from box import Box

import cProfile, pstats, io
from pstats import SortKey

import helpers

from loaders.camera_geometry_loader import camera_geometry_loader, camera_geometry_loader_real
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from trainer import Trainer
from logger import Logger
from inference import Inference

from misc import extract_foreground, remove_background, remove_background2, color_depthmap

from config import cfg


if __name__ == '__main__':

    model = torch.load('./logs/priority3_real/20220423_164938/model/3000.pth')

    mask = torch.zeros([128]*3)

    inference = Inference(
        model=model,
        mask=mask,
        n_rays=cfg.inference.n_rays,
        voxel_res=cfg.inference.voxel_res,
        thresh=cfg.inference.thresh,
        batch_size=cfg.inference.batch_size,
        )

    H, W = 1500, 2000

    steps = 60

    for i in range(steps):

        E = torch.clone(model.extrinsics[303])
        E[0, 3] += 0.1 * m.sin(i / steps * m.pi * 2)
        E[1, 3] += 0.1 * m.cos(i / steps * m.pi * 2)

        image, depth, weights, z_vals_s = inference.render_image(H, W, model.intrinsics[303], E)

        # depth = z_vals_s[torch.argmax(weights, dim=-1)]

        # depth = 1 / depth
        # depth[depth < 0.1] = 0.1
        # depth[depth > 1000] = 1000
        # depth = 3 - torch.log10(depth)
        # print(torch.amax(depth), torch.amin(depth))

        disp = np.concatenate([image.cpu().numpy()[..., np.array([2, 1, 0], dtype=int)], color_depthmap(depth.cpu().numpy(), 4, 0)], axis=0).transpose(1, 0, 2)[::-1, ...]

        cv2.imwrite(f'./data/demo_clip2/{i:03d}.png', np.uint8(disp[..., np.array([2, 1, 0], dtype=int)] * 255))

    plt.imshow(disp)
    plt.show()