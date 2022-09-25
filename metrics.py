import torch
import torchmetrics
import math as m

from torch.nn import functional as F

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from losses import psnr
from rotation import rot2euler


class MetricWrapper(object):
    pass



class SSIMWrapper(MetricWrapper):
    def __init__(self):
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()

    def __call__(self, pred, target):
        pred_bchw = torch.clamp(torch.Tensor(pred.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)

        target_bchw = torch.clamp(target.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)

        return self.lpips(pred_bchw, target_bchw)


class LPIPSWrapper(MetricWrapper):
    def __init__(self):
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to('cuda')

    def __call__(self, pred, target):
        pred_bchw = torch.clamp(torch.Tensor(pred.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        pred_scale = torch.clamp(pred_bchw * 2 - 1, min=-1, max=1)

        target_bchw = torch.clamp(target.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        target_scale = torch.clamp(target_bchw * 2 - 1, min=-1, max=1)

        return self.scale(pred_scale, target_scale)
        

class PSNRWrapper(MetricWrapper):
    def __call__(self, pred, target):
        pred_bchw = torch.clamp(torch.Tensor(pred.copy()).permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        target_bchw = torch.clamp(target.permute(2, 0, 1)[None, ...].to('cuda'), min=0, max=1)
        return psnr(pred, target)


@torch.no_grad()
def pose_inv_error(pred, target):
    error = target @ pred
    error_euler = rot2euler(error[:3, :3], radians=True)
    error_rot = torch.norm(error_euler)
    error_trans = torch.norm(error[:3, 3])
    return error_rot, error_trans


class RotationInvErrorWrapper(MetricWrapper):
    def __init__(self, radians):
        self.radians = radians

    def __call__(self, pred, target):
        error = target @ pred
        error_euler = rot2euler(error[:3, :3], self.radians)
        error_rot = torch.norm(error_euler)
        return error_rot


class TranslationInvErrorWrapper(MetricWrapper):
    def __call__(self, pred, target):
        error = target @ pred
        error_trans = torch.norm(error[:3, 3])
        return error_trans