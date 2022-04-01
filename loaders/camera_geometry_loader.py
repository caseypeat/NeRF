import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm

# sys.path.insert(0, '..')

import camera_geometry
from camera_geometry.scan import load_scan
from camera_geometry.scan.views import load_frames


np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)


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

            id_temp = frame.ids / (16.0/65536.0)
            ids_list.append(id_temp)
    
    images = np.stack(images_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)
    extrinsics = np.stack(extrinsics_list, axis=0)
    depths = np.stack(depths_list, axis=0)
    ids = np.stack(ids_list, axis=0)

    return images, depths, intrinsics, extrinsics, ids