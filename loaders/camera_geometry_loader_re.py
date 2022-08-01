import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path
from functools import cached_property

from camera_geometry.scan.scan import Scan
from camera_geometry.json import load_json
from camera_geometry.scan import load_scan
from camera_geometry.scan.views import load_frames


class CameraGeometryLoader(object):

    def __init__(self, scan_paths, frame_ranges, frame_strides, transforms, image_scale):
        scan_list = []
        frames_list = []
        transform_list = []

        self.N = 0
        for scan_path, frame_range, frame_stride, transform_path in zip(scan_paths, frame_ranges, frame_strides, transforms):
            scan = load_scan(scan_path, frame_range=frame_range, frame_stride=frame_stride, image_scale=image_scale)
            frames = load_frames(scan)
            if transform_path is not None:
                transform = np.load(transform_path)
            else:
                transform = np.eye(4)

            if frames[0][0].depth is None:
                self.depths_bool = False
            else:
                self.depths_bool = True


            scan_list.append(scan)
            frames_list.append(frames)
            transform_list.append(transform)

            self.N += len(frames) * len(frames[0])
            self.H, self.W = frames[0][0].rgb.shape[:2]

        self.images = torch.full([self.N, self.H, self.W, 4], fill_value=255, dtype=torch.uint8)
        if self.depths_bool:
            self.depths = torch.full([self.N, self.H, self.W], fill_value=1, dtype=torch.float32)
        self.intrinsics = torch.zeros([self.N, 3, 3], dtype=torch.float32)
        self.extrinsics = torch.zeros([self.N, 4, 4], dtype=torch.float32)

        self.index_mapping = []
        i = 0
        s = 0
        for scan, frames, transform in zip(scan_list, frames_list, transform_list):
            r = 0
            for rig in tqdm(frames):
                c = 0
                for frame in rig:
                    self.images[i, :, :, :3] = torch.ByteTensor(frame.rgb)
                    if self.depths_bool:
                        self.depths[i, :, :] = torch.Tensor(frame.depth)
                    self.intrinsics[i] = torch.Tensor(frame.camera.intrinsic)
                    self.extrinsics[i] =  torch.Tensor(transform) @ torch.Tensor(frame.camera.extrinsic)

                    self.mapping_index.append({'scan': s, 'rig': r, 'camera': c})

                    i += 1
                    c += 1
                r += 1
            s += 1



        del scan_list
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

    
    def get_custom_batch(self, n, h, w, background, device):
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


    def get_pointcloud_batch(self, cams, freq, side_buffer, device):

        n = []
        for i in range(self.N):
            if self.get_rig(i) > side_buffer and self.get_rig(i) < self.get_num_rigs(self.get_scan(i)) - side_buffer:
                if self.get_camera(i) in cams:
                    if self.get_rig(i) % freq == 0:
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

    @cached_property
    def translation_center(self):
        return torch.mean(self.extrinsics[..., :3, 3], dim=0, keepdims=True)

    def get_scan(self, index):
        return self.index_mapping[index]['scan']

    def get_num_scans(self):
        return

    def get_rig(self, index):
        return self.index_mapping[index]['rig']

    def get_num_rigs(self, scan):
        max_rig = 0
        for i in range(self.N):
            if self.get_scan(i) == scan:
                if self.get_rig(i) > max_rig:
                    max_rig = self.get_rig(i)
        return max_rig

    def get_camera(self, index):
        return self.index_mapping[index]['camera']

    # def get_num_cameras(self, scan, rig):
    #     count = 0
    #     for i in range(self.N):
    #         if self.get_scan(i) == scan and self.get_rig(i) == rig:
    #             count += 1
    #     return count


if __name__ == '__main__':
    scan_path = '/home/casey/PhD/data/conan_scans/ROW_349_EAST_SLOW_0006/scan.json'

    data_loader = CameraGeometryLoader([scan_path], [(0, 20)], image_scale=0.5)

    # n, h, w, K, E, rgb_gt, color_bg = data_loader.get_random_batch(32)
    n, h, w, K, E, rgb_gt, color_bg = data_loader.get_image_batch(32)

    print(n.shape, h.shape, K.shape, E.shape, rgb_gt.shape, color_bg)