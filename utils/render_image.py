import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from loaders.camera_geometry_loader import camera_geometry_loader, camera_geometry_loader_real, meta_camera_geometry, meta_camera_geometry_real
from loaders.synthetic import load_image_set

from nets import NeRFNetwork
from inference import Inferencer

from misc import color_depthmap

from config import cfg


if __name__ == '__main__':

    if cfg.scene.real:
        images, depths, intrinsics, extrinsics = meta_camera_geometry_real(cfg.scene.scene_path, cfg.scene.frame_range)
    else:
        images, depths, intrinsics, extrinsics = meta_camera_geometry(cfg.scene.scene_path, cfg.scene.remove_background_bool)

    N, H, W, C = images.shape

    model = NeRFNetwork(
        # renderer
        intrinsics=intrinsics,
        extrinsics=extrinsics,

        # net
        N=N
    ).to('cuda')

    model.load_state_dict(torch.load(cfg.nets.load_path))

    inferencer = Inferencer(model=model)

    image, invdepth = inferencer.render_image(H, W, model.intrinsics[cfg.inference.image_num], model.extrinsics[cfg.inference.image_num])
    invdepth_thesh = inferencer.render_invdepth_thresh(H, W, model.intrinsics[cfg.inference.image_num], model.extrinsics[cfg.inference.image_num])
    output = np.concatenate([image.cpu().numpy()[..., np.array([2, 1, 0], dtype=int)], color_depthmap(invdepth.cpu().numpy(), 4, 0), color_depthmap(invdepth_thesh.cpu().numpy(), 4, 0)], axis=0).transpose(1, 0, 2)

    # cv2.imwrite(f'./data/demo_clip2/{i:03d}.png', np.uint8(disp[..., np.array([2, 1, 0], dtype=int)] * 255))

    plt.imshow(output)
    plt.show()