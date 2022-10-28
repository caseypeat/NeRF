import numpy as np
import torch
import cv2
import os
import json
import math as m
import matplotlib.pyplot as plt 

from tqdm import tqdm


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


def build_intrinsics(fx, fy, cx, cy):
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    return K


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def load_image_set(rootdir, near=2, far=6, scale=1):
    with open(os.path.join(rootdir, 'transforms_train.json')) as fp:
        transforms = json.load(fp)

    images = []
    intrinsics = []
    extrinsics = []
    bds = []

    for i in range(len(transforms['frames'])):
        image_path = os.path.join(rootdir, transforms['frames'][i]['file_path']) + '.png'

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        image = image.astype(np.float32) / 255

        images.append(image)

        # Get calibration
        H, W = image.shape[:2]
        alpha = float(transforms['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * alpha)
        intrinsics.append(build_intrinsics(focal, focal, W/2, H/2))

        # extrinsics.append(np.array(transforms['frames'][i]['transform_matrix']))
        extrinsics.append(nerf_matrix_to_ngp(np.array(transforms['frames'][i]['transform_matrix']), 0.8))

        bds.append(np.array([near, far], dtype=np.float32))

    images = np.stack(images, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    bds = np.stack(bds, axis=0)

    N, H, W, _ = np.shape(images)

    depths = np.ones((N, H, W))

    ## scaling to fit within (-1, 1)
    # scale = 100
    # extrinsics[:,:3,3] = extrinsics[:,:3,3] * scale
    # extrinsics[:,:3,3] = extrinsics[:,:3,3] + 0.5
    # bds = bds * scale

    # extrinsics = nerf_matrix_to_ngp(extrinsics)

    #
    # extrinsics[:,:3,1] *= -1
    # extrinsics[:,:3,2] *= -1

    # print(np.amax(extrinsics[:,:3,3]), np.amin(extrinsics[:,:3,3]))

    return images, depths, intrinsics, extrinsics, bds


class SyntheticLoader(object):

    def __init__(self, rootdir, load_images_bool=True, load_depths_bool=False):

        self.rootdir = rootdir
        self.load_images_bool = load_images_bool
        self.load_depths_bool = load_depths_bool

        with open(os.path.join(rootdir, 'transforms_train.json')) as fp:
            transforms = json.load(fp)

        images = []
        intrinsics = []
        extrinsics = []

        for i in tqdm(range(len(transforms['frames']))):
            image_path = os.path.join(rootdir, transforms['frames'][i]['file_path']) + '.png'

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # image = image.astype(np.float32) / 255

            images.append(torch.ByteTensor(image))

            # Get calibration
            H, W = image.shape[:2]
            alpha = float(transforms['camera_angle_x'])
            focal = 0.5 * W / np.tan(0.5 * alpha)
            intrinsics.append(torch.Tensor(build_intrinsics(focal, focal, W/2, H/2)))

            extrinsics.append(torch.Tensor(nerf_matrix_to_ngp(np.array(transforms['frames'][i]['transform_matrix']), 1)))

        self.images = torch.stack(images, dim=0)
        self.intrinsics = torch.stack(intrinsics, dim=0)
        self.extrinsics = torch.stack(extrinsics, dim=0)

        self.translation_center = torch.mean(self.extrinsics[..., :3, 3], dim=0, keepdims=True)

        self.N, self.H, self.W, self.C = self.images.shape

    def format_groundtruth(self, gt, background=None):
        # If image data is stored as uint8, convert to float32 and scale to (0, 1)
        # is alpha channel is present, add random background color to image data
        if background is None:
            color_bg = torch.rand(3, device=gt.device) # [3], frame-wise random.
        else:
            color_bg = torch.Tensor(background).to(gt.device)

        if gt.shape[-1] == 4:
            if self.images.dtype == torch.uint8:
                rgba_gt = (gt.to(torch.float32) / 255).to(gt.device)
            else:
                rgba_gt = gt.to(gt.device)
            rgb_gt = rgba_gt[..., :3] * rgba_gt[..., 3:] + color_bg * (1 - rgba_gt[..., 3:])
        else:
            if self.images.dtype == torch.uint8:
                rgb_gt = (gt.to(torch.float32) / 255).to(gt.device)
            else:
                rgb_gt = gt.to(gt.device)

        return rgb_gt, color_bg

    
    def get_custom_batch(self, n, h, w, background, device):
        K = self.intrinsics[n].to(device)
        E = self.extrinsics[n].to(device)

        if self.load_images_bool:
            rgb_gt, color_bg = self.format_groundtruth(self.images[n, h, w, :].to(device), background)
        else:
            rgb_gt, color_bg = None, None
        if self.load_depths_bool:
            depth = self.depths[n, h, w].to(device)
        else:
            depth = None

        n = n.to(device)
        h = h.to(device)
        w = w.to(device)

        return n, h, w, K, E, rgb_gt, color_bg, depth


    def get_random_batch(self, batch_size, device):
        n = torch.randint(0, self.N, (batch_size,))
        h = torch.randint(0, self.H, (batch_size,))
        w = torch.randint(0, self.W, (batch_size,))

        return self.get_custom_batch(n, h, w, background=None, device=device)


    def get_image_batch(self, image_num, device):
        h = torch.arange(0, self.H)
        w = torch.arange(0, self.W)
        h, w = torch.meshgrid(h, w, indexing='ij')
        n = torch.full(h.shape, fill_value=image_num)

        return self.get_custom_batch(n, h, w, background=(1, 1, 1), device=device)