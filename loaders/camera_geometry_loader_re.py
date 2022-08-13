import numpy as np
import torch

from tqdm import tqdm
from pathlib import Path
from functools import cached_property

from camera_geometry.scan.scan import Scan
from camera_geometry.json import load_json
from camera_geometry.scan import load_scan
from camera_geometry.scan.views import load_frames


class IndexMapping(object):
    def __init__(self, index2src_mapping, src2index_mapping):
        self.index2src_mapping = index2src_mapping
        self.src2index_mapping = src2index_mapping

        scan_max = 0
        rig_max = 0
        cam_max = 0

        for i in range(len(index2src_mapping)):
            s, r, c, = index2src_mapping[i]
            scan_max = scan_max if scan_max > s else s
            rig_max = rig_max if rig_max > r else r
            cam_max = cam_max if cam_max > c else c

        self.index2src_mapping_t = torch.zeros((len(index2src_mapping), 3), dtype=torch.long)
        for i in range(len(index2src_mapping)):
            s, r, c, = index2src_mapping[i]
            self.index2src_mapping_t[i, 0] = s
            self.index2src_mapping_t[i, 1] = r
            self.index2src_mapping_t[i, 2] = c


    def src_to_index(self, scan, rig, camera):
        if (scan, rig, camera) in self.src2index_mapping.keys():
            return self.src2index_mapping[(scan, rig, camera)]
        else:
            return None

    def index_to_src(self, index):
        src = self.index2src_mapping_t[index]
        return src[:, 0], src[:, 1], src[:, 2]

    # def index_to_src(self, index):
    #     scans = torch.zeros(index.shape, dtype=torch.long, device=index.device)
    #     rigs = torch.zeros(index.shape, dtype=torch.long, device=index.device)
    #     cams = torch.zeros(index.shape, dtype=torch.long, device=index.device)

    #     for i, ind in enumerate(index):
    #         scans[i], rigs[i], cams[i] = self.index2src_mapping[ind.item()]
    #         # rigs[i] = self.index2src_mapping[ind]
    #         # cams[i] = self.index2src_mapping[ind]

    #     return scans, rigs, cams

    def get_num_scans(self):
        s = 0
        while self.src_to_index(s, 0, 0) is not None:
            s += 1
        return s

    def get_num_rigs(self, scan):
        s = scan
        r = 0
        while self.src_to_index(s, r, 0) is not None:
            r += 1
        return r

    def get_num_cams(self, scan, rig):
        s = scan
        r = rig
        c = 0
        while self.src_to_index(s, r, c) is not None:
            c += 1
        return c


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

        self.index2src_mapping = {}
        self.src2index_mapping = {}
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

                    self.index2src_mapping[i] = (s, r, c)
                    self.src2index_mapping[(s, r, c)] = i

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
            s, r, c = self.index_to_src(i)
            if r > side_buffer and r < self.get_num_rigs(s) - side_buffer:
                if c in cams:
                    if r % freq == 0:
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

    def src_to_index(self, scan, rig, camera):
        if (scan, rig, camera) in self.src2index_mapping.keys():
            return self.src2index_mapping[(scan, rig, camera)]
        else:
            return None

    def index_to_src(self, index):
        return self.index2src_mapping[index]


    def get_num_scans(self):
        s = 0
        while self.src_to_index(s, 0, 0) is not None:
            s += 1
        return s

    def get_num_rigs(self, scan):
        s = scan
        r = 0
        while self.src_to_index(s, r, 0) is not None:
            r += 1
        return r

    def get_num_cams(self, scan, rig):
        s = scan
        r = rig
        c = 0
        while self.src_to_index(s, r, c) is not None:
            c += 1
        return c

    # def get_num_scans(self):
    #     scans_unique = 0
    #     for i in range(self.N):
    #         s, r, c = self.index_to_src(i)
    #         if s not in scans_unique:
    #             scans_unique += 1
    #     return scans_unique

    # def get_num_rigs(self, scan=None):
    #     rigs_unique = 0
    #     for i in range(self.N):
    #         s, r, c = self.index_to_src(i)
    #         if scan == None or scan == s:
    #             if r not in rigs_unique:
    #                 rigs_unique += 1
    #     return rigs_unique

    # def get_num_cam(self, scan=None, rig=None):
    #     cam_unique = 0
    #     for i in range(self.N):
    #         s, r, c = self.index_to_src(i)
    #         if (scan == None or scan == s) and (rig == None or rig == r):
    #             if c not in cam_unique:
    #                 cam_unique += 1
    #     return cam_unique

    # def get_rig(self, index):
    #     return self.index_mapping[index]['rig']

    # def get_num_rigs(self, scan):
    #     max_rig = 0
    #     for i in range(self.N):
    #         if self.get_scan(i) == scan:
    #             if self.get_rig(i) > max_rig:
    #                 max_rig = self.get_rig(i)
    #     return max_rig

    # def get_camera(self, index):
    #     return self.index_mapping[index]['camera']

    # def get_num_cameras(self, scan, rig):
    #     pass

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