import numpy as np
import rotation as rot

def calculate_xyz_error(transform_pred, transform_target):
    xyz_pred = transform_pred[:3, 3].detach().cpu().numpy()
    xyz_true = transform_pred[:3, 3].detach().cpu().numpy()
    xyz_error = (xyz_pred - xyz_true)
    return xyz_error

def calculate_rot_error(transform_pred, transform_target):
    rot_init = transform_pred[:3, :3].detach().cpu().numpy()
    rot_pred = transform_pred[:3, :3].detach().cpu().numpy()
    rot_true = np.eye(3)  # no adjustment to rotation in synthetic data
    rot_error = np.linalg.norm(rot.matrix2xyz_extrinsic(rot_pred @ np.linalg.inv(rot_true)))
    return rot_error