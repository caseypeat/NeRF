import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as m
import matplotlib.pyplot as plt

from matplotlib import cm

from tqdm import tqdm


# @torch.no_grad()
def render_image(renderer, n, h, w, K, E, n_rays, cos_anneal_ratio):

    # _f : flattened
    # _b : batched

    # renderer.eval()

    n_f = torch.reshape(n, (-1,))
    h_f = torch.reshape(h, (-1,))
    w_f = torch.reshape(w, (-1,))

    K_f = torch.reshape(K, (-1, 3, 3))
    E_f = torch.reshape(E, (-1, 4, 4))

    image_f = torch.zeros((*h_f.shape, 3), device='cpu')
    invdepth_f = torch.zeros(h_f.shape, device='cpu')

    for a in tqdm(range(0, len(n_f), n_rays)):
        b = min(len(n_f), a+n_rays)

        n_fb = n_f[a:b]
        h_fb = h_f[a:b]
        w_fb = w_f[a:b]

        K_fb = K_f[a:b]
        E_fb = E_f[a:b]

        color_bg = torch.ones(3, device='cuda') # [3], fixed white background

        image_fb, _, _, aux_outputs_fb = renderer.render(n_fb, h_fb, w_fb, K_fb, E_fb, cos_anneal_ratio, bg_color=color_bg)

        image_f[a:b] = image_fb.detach().cpu()
        invdepth_f[a:b] = aux_outputs_fb['invdepth'].cpu()

    image = torch.reshape(image_f, (*h.shape, 3))
    invdepth = torch.reshape(invdepth_f, h.shape)

    return image, invdepth


class ImageInference(object):
    def __init__(self, renderer, dataloader, n_rays, image_num):
        self.renderer = renderer
        self.dataloader = dataloader

        self.n_rays = n_rays
        self.image_num = image_num

    def __call__(self, cos_anneal_ratio, image_num=None):
        image_num = self.image_num if image_num is None else image_num 
        n, h, w, K, E, _, _, _ = self.dataloader.get_image_batch(image_num, device='cuda')
        return render_image(self.renderer, n, h, w, K, E, self.n_rays, cos_anneal_ratio)