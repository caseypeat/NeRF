from cv2 import transform
import numpy as np
import torch
import matplotlib.pyplot as plt

from allign.rotation import vec2skew, Exp, matrix2xyz_extrinsic

from loaders.camera_geometry_loader import CameraGeometryLoader


class Measure(object):
    # def __init__(self, transform, renderer, depth_thresh, translation_center_a, translation_center_b):
    def __init__(self, renderer, depth_thresh, translation_center_a, translation_center_b):

        # self.transform = transform
        self.renderer = renderer

        self.depth_thresh = depth_thresh
        self.translation_center_a = translation_center_a
        self.translation_center_b = translation_center_b
    
    @torch.no_grad()
    def calculate_point_error(self, h, w, K, E, depth, transform):

        rays_o, rays_d = self.renderer.get_rays(h, w, K, E)
        depth_broad = depth[:, None].repeat(1, 3)
        points = rays_o + rays_d * depth_broad
        points = points[depth_broad < self.depth_thresh].reshape(-1, 3) # remove points with depth > depth_thresh

        points_pred = transform(points)

        points_target = points + (self.translation_center_a - self.translation_center_b).to('cuda')

        point_error = torch.norm(points_pred - points_target, dim=1)

        return torch.mean(point_error)








if __name__ == '__main__':
    scene_path = '/home/casey/Documents/PhD/data/my_renders/vine_C6_0/back_close/cameras.json'
    loader = CameraGeometryLoader([scene_path], [None], [None], image_scale=0.5)

    n, h, w, K, E, rgb_gt, color_bg, depth = loader.get_image_batch(34)

    depth[depth > 0.7] = 0

    plt.imshow(depth)
    plt.show()

