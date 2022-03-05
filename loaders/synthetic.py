import numpy as np
import torch
import cv2
import os
import json
import math as m
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


def build_intrinsics(fx, fy, cx, cy):
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    return K


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

        # depth_path = os.path.join(rootdir, transforms['frames'][i]['file_path']) + '_depth_0001.png'
        # depth = cv2.imread(depth_path, -1)

        image = image.astype(np.float32) / 255
        image = image[..., :3]

        images.append(image)

        # Get calibration
        H, W = image.shape[:2]
        alpha = float(transforms['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * alpha)
        intrinsics.append(build_intrinsics(focal, focal, W/2, H/2))

        extrinsics.append(np.array(transforms['frames'][i]['transform_matrix']))

        bds.append(np.array([near, far], dtype=np.float32))

    images = np.stack(images, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    bds = np.stack(bds, axis=0)

    N, H, W, _ = np.shape(images)

    depths = np.ones((N, H, W))

    ## scaling to fit within (-1, 1)
    # scale = 100
    extrinsics[:,:3,3] = extrinsics[:,:3,3] * scale
    extrinsics[:,:3,3] = extrinsics[:,:3,3] + 0.5
    bds = bds * scale

    #
    extrinsics[:,:3,1] *= -1
    extrinsics[:,:3,2] *= -1

    return images, depths, intrinsics, extrinsics, bds