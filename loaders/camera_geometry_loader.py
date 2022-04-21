import numpy as np
import torch
import cv2
import os
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm

import camera_geometry
from camera_geometry.scan import load_scan
from camera_geometry.scan.views import load_frames

import helpers
from misc import remove_background

np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)
    

def camera_geometry_loader(scene_path, image_scale=1, frame_range=None):
    # scene_path = os.path.join(scene_dir, 'scene.json')
    scene = load_scan(scene_path, image_scale=image_scale, frame_range=frame_range)

    images_list = []
    depths_list = []
    intrinsics_list = []
    extrinsics_list = []
    ids_list = []

    frames = load_frames(scene)

    for frame_list in tqdm(frames):
        for frame in frame_list:
            image_temp = frame.rgb.astype(np.float32) / 255
            images_list.append(image_temp)

            intrinsic_temp = frame.camera.intrinsic
            intrinsics_list.append(intrinsic_temp.astype(np.float32))

            extrinsic_temp = frame.camera.parent_to_camera
            extrinsics_list.append(extrinsic_temp.astype(np.float32))

            if 'depth' in frame.keys():
                depth_temp = frame.depth.astype(np.float32)
            else:
                depth_temp = np.ones(image_temp.shape[:-1], dtype=np.float32)
            depths_list.append(depth_temp)

            if 'ids' in frame.keys():
                id_temp = frame.ids / (16.0/65536.0)
            else:
                ids_temp = np.ones(image_temp.shape[:-1], dtype=np.float32)
            ids_list.append(id_temp)
    
    images = np.stack(images_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)
    extrinsics = np.stack(extrinsics_list, axis=0)
    depths = np.stack(depths_list, axis=0)
    ids = np.stack(ids_list, axis=0)

    return images, depths, intrinsics, extrinsics, ids


def meta_camera_geometry(scene_path, remove_background_bool):

    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader(scene_path, image_scale=0.5)
    if remove_background_bool:
        images, depths, ids = remove_background(images, depths, ids, threshold=1)

    images_ds = images[:, ::4, ::4, :]
    depths_ds = depths[:, ::4, ::4]
    ids_ds = ids[:, ::4, ::4]
    images_ds_nb, depths_ds_nb, ids_ds_nb = remove_background(images_ds, depths_ds, ids_ds, threshold=1)

    xyz_min, xyz_max = helpers.calculate_bounds(images_ds_nb, depths_ds_nb, intrinsics, extrinsics)
    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - (xyz_max + xyz_min) / 2
    xyz_min_norm, xyz_max_norm = helpers.calculate_bounds_sphere(images_ds_nb, depths_ds_nb, intrinsics, extrinsics)
    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / xyz_max_norm

    depths = depths / xyz_max_norm

    images = torch.Tensor(images)
    depths = torch.Tensor(depths)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)

    return images, depths, intrinsics, extrinsics


def camera_geometry_loader_real(scene_path, image_scale=1, frame_range=None):
    # scene_path = os.path.join(scene_dir, 'scene.json')
    scene = load_scan(scene_path, image_scale=image_scale, frame_range=frame_range)

    images_list = []
    intrinsics_list = []
    extrinsics_list = []

    frames = load_frames(scene)

    for frame_list in tqdm(frames):
        for frame in frame_list:
            image_temp = frame.rgb
            images_list.append(image_temp)

            intrinsic_temp = frame.camera.intrinsic
            intrinsics_list.append(intrinsic_temp.astype(np.float32))

            extrinsic_temp = frame.camera.parent_to_camera
            extrinsics_list.append(extrinsic_temp.astype(np.float32))

    
    images = np.stack(images_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)
    extrinsics = np.stack(extrinsics_list, axis=0)

    return images, None, intrinsics, extrinsics, None


def meta_camera_geometry_real(scene_path):

    images, depths, intrinsics, extrinsics, ids = camera_geometry_loader_real(scene_path, image_scale=0.5)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] - np.mean(extrinsics[..., :3, 3], axis=0, keepdims=True)

    extrinsics[..., :3, 3] = extrinsics[..., :3, 3] / 2

    images = torch.ByteTensor(images)
    intrinsics = torch.Tensor(intrinsics)
    extrinsics = torch.Tensor(extrinsics)

    return images, None, intrinsics, extrinsics