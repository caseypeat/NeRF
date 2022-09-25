import numpy as np
import torch

# https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/lie_group_helper.py
def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def vec2skew_vector(v):
    zero = torch.zeros((v.shape[0], 1), dtype=torch.float32, device=v.device)  # (N, 1)
    skew_v0 = torch.cat([ zero,    -v[:, 2:3],   v[:, 1:2]], dim=1)  # (N, 3, 1)
    skew_v1 = torch.cat([ v[:, 2:3],   zero,    -v[:, 0:1]], dim=1)
    skew_v2 = torch.cat([-v[:, 1:2],   v[:, 0:1],   zero], dim=1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (N, 3, 3)
    return skew_v  # (N, 3, 3)


def Exp_vector(r):
    skew_r = vec2skew_vector(r)  # (N, 3, 3)
    norm_r = torch.norm(r, dim=1) + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)[None, ...].expand(r.shape[0], -1, -1)
    R = eye + (torch.sin(norm_r) / norm_r)[..., None, None] * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2)[..., None, None] * (skew_r @ skew_r)
    return R


# https://github.com/alisterburt/eulerangles/blob/master/eulerangles/math/rotation_matrix_to_eulers.py
def matrix2xyz_extrinsic(rotation_matrices: np.ndarray) -> np.ndarray:
    """
    Rz(k3) @ Ry(k2) @ Rx(k1) = [[c2c3, s1s2c3-c1s3, c1s2c3+s1s3],
                                [c2s3, s1s2s3+c1c3, c1s2s3-s1c3],
                                [-s2, s1c2, c1c2]]
    """
    rotation_matrices = rotation_matrices.reshape((-1, 3, 3))
    angles_radians = np.zeros((rotation_matrices.shape[0], 3))

    # Angle 2 can be taken directly from matrices
    angles_radians[:, 1] = -np.arcsin(rotation_matrices[:, 2, 0])

    # Gimbal lock case (c2 = 0)
    tolerance = 1e-4

    # Find indices where this is the case
    gimbal_idx = np.abs(rotation_matrices[:, 0, 0]) < tolerance

    # Calculate angle 1 and set angle 3 = 0 for those indices
    r23 = rotation_matrices[gimbal_idx, 1, 2]
    r22 = rotation_matrices[gimbal_idx, 1, 1]
    angles_radians[gimbal_idx, 0] = np.arctan2(-r23, r22)
    angles_radians[gimbal_idx, 2] = 0

    # Normal case (s2 > 0)
    idx = np.invert(gimbal_idx)
    r32 = rotation_matrices[idx, 2, 1]
    r33 = rotation_matrices[idx, 2, 2]
    r21 = rotation_matrices[idx, 1, 0]
    r11 = rotation_matrices[idx, 0, 0]
    angles_radians[idx, 0] = np.arctan2(r32, r33)
    angles_radians[idx, 2] = np.arctan2(r21, r11)

    # convert to degrees
    euler_angles = np.rad2deg(angles_radians)

    return angles_radians
    # return euler_angles


def rot2euler(rotation_matrices:torch.Tensor, radians:bool=True):
    rotation_matrices = rotation_matrices.reshape((-1, 3, 3))
    angles_radians = rotation_matrices.new_zeros((rotation_matrices.shape[0], 3))

    # Angle 2 can be taken directly from matrices
    angles_radians[:, 1] = -torch.arcsin(rotation_matrices[:, 2, 0])

    # Gimbal lock case (c2 = 0)
    tolerance = 1e-4

    # Find indices where this is the case
    gimbal_idx = torch.abs(rotation_matrices[:, 0, 0]) < tolerance

    # Calculate angle 1 and set angle 3 = 0 for those indices
    r23 = rotation_matrices[gimbal_idx, 1, 2]
    r22 = rotation_matrices[gimbal_idx, 1, 1]
    angles_radians[gimbal_idx, 0] = torch.arctan2(-r23, r22)
    angles_radians[gimbal_idx, 2] = 0

    # Normal case (s2 > 0)
    idx = ~gimbal_idx
    r32 = rotation_matrices[idx, 2, 1]
    r33 = rotation_matrices[idx, 2, 2]
    r21 = rotation_matrices[idx, 1, 0]
    r11 = rotation_matrices[idx, 0, 0]
    angles_radians[idx, 0] = torch.arctan2(r32, r33)
    angles_radians[idx, 2] = torch.arctan2(r21, r11)

    if radians:
        return angles_radians
    else:
        return torch.rad2deg(angles_radians)