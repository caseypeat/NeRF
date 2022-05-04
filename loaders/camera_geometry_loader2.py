import numpy as np
import torch

from tqdm import tqdm

from camera_geometry.scan import load_scan
from camera_geometry.scan.views import load_frames

class CameraGeometryLoader(object):
    def __init__(self, scene_paths, frame_ranges, image_scale):
        scene_list = []
        frames_list = []

        self.N = 0

        for scene_path, frame_range in zip(scene_paths, frame_ranges):
            scene = load_scan(scene_path, frame_range=frame_range, image_scale=image_scale)
            frames = load_frames(scene)

            scene_list.append(scene)
            frames_list.append(frames)

            self.N += len(frames) * len(frames[0])
            self.H, self.W = frames[0][0].rgb.shape[:2]

        self.images = torch.ones([self.N, self.H, self.W, 4], dtype=torch.uint8)
        self.intrinsics = torch.zeros([self.N, 3, 3], dtype=torch.float32)
        self.extrinsics = torch.zeros([self.N, 4, 4], dtype=torch.float32)

        for i, (scene, frames) in enumerate(zip(scene_list, frames_list)):
            for rig in tqdm(frames):
                for frame in rig:
                    self.images[i, :, :, :3] = torch.ByteTensor(frame.rgb)
                    self.intrinsics[i] = torch.Tensor(frame.camera.intrinsic)
                    self.extrinsics[i] = torch.Tensor(frame.camera.extrinsic)


if __name__ == '__main__':
    scene_path = '/home/casey/Documents/PhD/data/conan_scans/ROW_349_EAST_SLOW_0006/scene.json'

    data_loader = CameraGeometryLoader([scene_path], [(0, 20)], image_scale=0.5)