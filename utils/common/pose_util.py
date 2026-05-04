"""
Pose representation utilities for 6DoF transformations.

Shared by both ArUco (support line) and HaMeR (main line) pipelines.
All poses use the convention: [x, y, z, rx, ry, rz] where rotation is axis-angle.
Homogeneous matrices are 4x4 SE(3).

Source: Adapted from UMI umi/common/pose_util.py (MIT License)
"""

import numpy as np
import scipy.spatial.transform as st


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert pose [pos3 + axis_angle3] to 4x4 homogeneous matrix.

    Supports batched input: (6,) -> (4,4) or (T, 6) -> (T, 4, 4).
    """
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])

    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def mat_to_pose(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous matrix to pose [pos3 + axis_angle3].

    Inverse of pose_to_mat. Supports batched input.
    """
    pos = mat[..., :3, 3]
    rot = st.Rotation.from_matrix(mat[..., :3, :3])

    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=np.float64)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def rvec_tvec_to_pose(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV rvec/tvec pair to pose [pos3 + axis_angle3].

    This is the direct output format of cv2.aruco.estimatePoseSingleMarkers.
    rvec is already axis-angle (Rodrigues), so just concatenate.
    """
    return np.concatenate([tvec.flatten(), rvec.flatten()])


def rvec_tvec_to_mat(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV rvec/tvec to 4x4 homogeneous matrix.

    # T_camera_marker: transforms points from marker frame to camera frame
    """
    return pose_to_mat(rvec_tvec_to_pose(rvec, tvec))


def invert_transform(mat: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid body transform efficiently.

    For rigid transforms: T^{-1} = [R^T | -R^T @ t]
    Much faster than np.linalg.inv for SE(3) matrices.
    """
    R = mat[..., :3, :3]
    t = mat[..., :3, 3:]
    R_inv = np.swapaxes(R, -2, -1)  # R^T
    t_inv = -R_inv @ t

    result = np.zeros_like(mat)
    result[..., :3, :3] = R_inv
    result[..., :3, 3:] = t_inv
    result[..., 3, 3] = 1
    return result


def transform_pose(tx: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to a pose.

    tx: T_new_old (4x4), pose: T_old_obj (6D) -> result: T_new_obj (6D)
    """
    pose_mat = pose_to_mat(pose)
    return mat_to_pose(tx @ pose_mat)


# ---- rot6d for training (used by generate_dataset.py later) ----

def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Extract first two rows of rotation matrix as 6D rotation representation.

    Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    6D is continuous and better for neural network training than axis-angle or quaternion.

    Input: (..., 3, 3) rotation matrix. Output: (..., 6).
    """
    first_two_rows = mat[..., :2, :].copy()
    return first_two_rows.reshape(mat.shape[:-2] + (6,))


def mat_to_pose9d(mat: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to 10D pose [pos3 + rot6d].

    Used as the training representation in UMI's Diffusion Policy.
    """
    pos = mat[..., :3, 3]
    rot6d = mat_to_rot6d(mat[..., :3, :3])
    return np.concatenate([pos, rot6d], axis=-1)
