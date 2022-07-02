from cv2 import transform
import numpy as np
import torch
import matplotlib.pyplot as plt

from allign.rotation import vec2skew, Exp, matrix2xyz_extrinsic

from loaders.camera_geometry_loader import CameraGeometryLoader


class Measure(object):
    def __init__(self, transform, renderer, depth_thresh, translation_center_a, translation_center_b):

        self.transform = transform
        self.renderer = renderer

        self.depth_thresh = depth_thresh
        self.translation_center_a = translation_center_a
        self.translation_center_b = translation_center_b

    # @torch.no_grad()
    # def calculate_point_error(self, h, w, K, E, depth):
    #     xyz_init = self.transform.init_transform[:3, 3].detach().cpu().numpy()
    #     xyz_pred = self.transform.T.detach().cpu().numpy() + xyz_init
    #     rot_init = self.transform.init_transform[:3, :3].detach().cpu().numpy()
    #     rot_pred = Exp(self.transform.R).detach().cpu().numpy() @ rot_init
    #     transform_pred = np.eye(4)
    #     transform_pred[:3, :3] = rot_pred
    #     transform_pred[:3, 3] = xyz_pred

    #     E_pred = transform_pred @ E.detach().cpu().numpy()
    #     rays_o_pred, rays_d_pred = self.renderer.get_rays(h, w, K, E_pred)
    #     points_pred = rays_o_pred + rays_d_pred * depth
    #     points_pred = points_pred[depth < self.depth_thresh]

    #     E_target = np.array(E_target.detach().cpu().numpy())
    #     E_target[:3, 3] = E_target[:3, 3] + (self.dataloader_a.translation_center - self.dataloader_b.translation_center)
    #     rays_o_target, rays_d_target = self.renderer.get_rays(h, w, K, E_target)
    #     points_target = rays_o_target + rays_d_target * depth
    #     points_target = points_target[depth < self.depth_thresh]

    #     point_error = np.linalg.norm(points_pred - points_target, dim=1)

    #     return np.mean(point_error)
    
    @torch.no_grad()
    def calculate_point_error(self, h, w, K, E, depth):

        rays_o, rays_d = self.renderer.get_rays(h, w, K, E)
        depth_broad = depth[:, None].repeat(1, 3)
        points = rays_o + rays_d * depth_broad
        points = points[depth_broad < self.depth_thresh].reshape(-1, 3) # remove points with depth > depth_thresh

        points_pred = self.transform(points)

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

