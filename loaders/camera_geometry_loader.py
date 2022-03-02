import numpy as np
import cv2
import os
import sys

from tqdm import tqdm

# sys.path.insert(0, '..')

import camera_geometry
from camera_geometry.scan import load_scan
from camera_geometry.scan.views import load_frames


np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)
    

def camera_geometry_loader(scene_path, image_scale=1, frame_range=None):
    # scene_path = os.path.join(scene_dir, 'scene.json')
    scene = load_scan(scene_path, image_scale=image_scale, frame_range=frame_range)

    images_list = []
    intrinsics_list = []
    extrinsics_list = []

    depths_list = []

    frames = load_frames(scene)

    for frame_list in tqdm(frames):
        for frame in frame_list:
            image_temp = frame.rgb.astype(np.float32) / 255
            images_list.append(image_temp)

            intrinsic_temp = frame.camera.intrinsic
            intrinsics_list.append(intrinsic_temp.astype(np.float32))

            extrinsic_temp = frame.camera.parent_to_camera
            extrinsics_list.append(extrinsic_temp.astype(np.float32))

            if 'depth' in frame.values():
                depth_temp = frame.depth.astype(np.float32)
            else:
                depth_temp = np.ones(image_temp.shape[:-1], dtype=np.float32)
            depths_list.append(depth_temp)
    
    images = np.stack(images_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)
    extrinsics = np.stack(extrinsics_list, axis=0)
    depths = np.stack(depths_list, axis=0)

    return images, intrinsics, extrinsics, depths