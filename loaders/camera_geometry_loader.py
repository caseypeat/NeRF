import numpy as np
import torch

from tqdm import tqdm

from camera_geometry.scan import Scan
from camera_geometry.scan.views import load_frames


class CameraGeometryLoader(object):
    def __init__(self, scene_paths, frame_ranges, transforms, image_scale):

        scene_list = []
        frames_list = []
        transform_list = []

        self.N = 0

        for scene_path, frame_range, transform_path in zip(scene_paths, frame_ranges, transforms):
            # scene = Scan.load(scene_path, frame_range=frame_range, image_scale=image_scale)
            scene = Scan.load(scene_path)
            frames = load_frames(scene)
            if transform_path is not None:
                transform = np.load(transform_path)
            else:
                transform = np.eye(4)

            if frames[0][0].depth is None:
                self.depths_bool = False
            else:
                self.depths_bool = True


            scene_list.append(scene)
            frames_list.append(frames)
            transform_list.append(transform)

            self.N += len(frames) * len(frames[0])
            self.H, self.W = frames[0][0].rgb.shape[:2]

        self.images = torch.full([self.N, self.H, self.W, 4], fill_value=255, dtype=torch.uint8)
        if self.depths_bool:
            self.depths = torch.full([self.N, self.H, self.W], fill_value=1, dtype=torch.float32)
        self.intrinsics = torch.zeros([self.N, 3, 3], dtype=torch.float32)
        self.extrinsics = torch.zeros([self.N, 4, 4], dtype=torch.float32)

        i = 0
        for scene, frames, transform in zip(scene_list, frames_list, transform_list):
            for rig in tqdm(frames):
                for frame in rig:
                    self.images[i, :, :, :3] = torch.ByteTensor(frame.rgb)
                    if self.depths_bool:
                        self.depths[i, :, :] = torch.Tensor(frame.depth) * 16/65536
                    self.intrinsics[i] = torch.Tensor(frame.camera.intrinsic)
                    self.extrinsics[i] =  torch.Tensor(transform) @ torch.Tensor(frame.camera.extrinsic)
                    i += 1

        self.translation_center = torch.mean(self.extrinsics[..., :3, 3], dim=0, keepdims=True)

        del scene_list
        del frames_list


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

    
    def get_custom_batch(self, n, h, w, background=None, device='cuda'):
        K = self.intrinsics[n].to(device)
        E = self.extrinsics[n].to(device)

        rgb_gt, color_bg = self.format_groundtruth(self.images[n, h, w, :].to(device), background)
        if self.depths_bool:
            depth = self.depths[n, h, w].to(device)
        else:
            depth = None

        n = n.to(device)
        h = h.to(device)
        w = w.to(device)

        return n, h, w, K, E, rgb_gt, color_bg, depth


    def get_random_batch(self, batch_size, device='cuda'):
        n = torch.randint(0, self.N, (batch_size,))
        h = torch.randint(0, self.H, (batch_size,))
        w = torch.randint(0, self.W, (batch_size,))

        return self.get_custom_batch(n, h, w)


    def get_image_batch(self, image_num, device='cpu'):
        h = torch.arange(0, self.H)
        w = torch.arange(0, self.W)
        h, w = torch.meshgrid(h, w, indexing='ij')
        n = torch.full(h.shape, fill_value=image_num)

        return self.get_custom_batch(n, h, w, background=(1, 1, 1), device=device)


    def get_pointcloud_batch(self, cams, freq, device='cpu'):

        n = []
        for i in range(self.N):
            # if (i > int(self.images.shape[0]*0.25) and i < int(self.images.shape[0]*0.75)):
            if i % 6 in cams:
                if i//6 % freq == 0:
                    n.append(i)
                        
        n = torch.Tensor(np.array(n)).to(int)
        h = torch.arange(0, self.H)
        w = torch.arange(0, self.W)
        n, h, w = torch.meshgrid(n, h, w, indexing='ij')

        return self.get_custom_batch(n, h, w, background=(1, 1, 1), device=device)


    def get_calibration(self, device='cuda'):
        K = self.intrinsics.to(device)
        E = self.extrinsics.to(device)

        return self.N, self.H, self.W, K, E

if __name__ == '__main__':
    scene_path = '/home/casey/Documents/PhD/data/conan_scans/ROW_349_EAST_SLOW_0006/scene.json'

    data_loader = CameraGeometryLoader([scene_path], [(0, 20)], image_scale=0.5)

    # n, h, w, K, E, rgb_gt, color_bg = data_loader.get_random_batch(32)
    n, h, w, K, E, rgb_gt, color_bg = data_loader.get_image_batch(32)

    print(n.shape, h.shape, K.shape, E.shape, rgb_gt.shape, color_bg)