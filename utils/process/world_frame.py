"""
Cam-frame → World-frame transformation primitives.

Pipeline position: shared helper for any line that has per-frame
T_world_cam. Consumed by 02_process.py (per-line) to lift HaMeR /
hand-tracking output (cam-frame wrist + MANO 21 joints + wrist quat)
into the world frame for the v2 dataset schema (see
docs/Tech_Survey_World_Frame_v2.md §7).

This module CONSUMES T_world_cam — it does not produce it. The
producers live in `utils/cam_pose/` (pluggable backends: ArUco/ChArUco
PnP today, SLAM tomorrow) and `utils/iphone/r3d_reader.py` (ARKit
passthrough).

Math (per frame t):
  T_world_cam[t] = [[R_world_cam, t_world_cam],
                    [0,           1          ]]      (4×4 SE(3))

  Position    : world_pt    = R_world_cam · cam_pt + t_world_cam
  Orientation : R_world_obj = R_world_cam · R_cam_obj

Conventions:
  - T_world_cam: shape (T, 4, 4) float, world is gravity-aligned per source
  - Rotations as scipy xyzw quaternions (matches OPC stage-2 schema)
  - NaN propagation: any non-finite input row → NaN output row
    (no silent zero substitution — masking is the consumer's job)

Why generic / device-agnostic:
  iPhone-line gets T_world_cam from ARKit (utils/iphone/r3d_reader);
  335-line gets it from a pluggable cam_pose source (utils/cam_pose,
  default ArUco ChArUco PnP). Both feed this same module. Adding a new
  device or new pose source only requires producing T_world_cam in the
  same shape — no math changes here.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _validate_T_world_cam(T_world_cam: np.ndarray, expected_T: int) -> None:
    if T_world_cam.ndim != 3 or T_world_cam.shape[1:] != (4, 4):
        raise ValueError(
            f"T_world_cam must be (T, 4, 4), got {T_world_cam.shape}"
        )
    if T_world_cam.shape[0] != expected_T:
        raise ValueError(
            f"T_world_cam length {T_world_cam.shape[0]} != "
            f"input length {expected_T}"
        )


def transform_points_cam_to_world(
    points_cam: np.ndarray,
    T_world_cam: np.ndarray,
) -> np.ndarray:
    """Lift per-frame cam-frame 3D points into world frame.

    Args:
        points_cam: (T, 3) for one point per frame OR (T, K, 3) for K
            points per frame (e.g. MANO 21 joints). May contain NaN
            rows; those propagate.
        T_world_cam: (T, 4, 4) per-frame cam→world SE(3).

    Returns:
        Same-shape array in world frame; NaN where input was NaN.

    Math (vectorized per t):
        world[t]    = R_world_cam[t] @ cam[t]    + t_world_cam[t]      # (T,3) case
        world[t,k]  = R_world_cam[t] @ cam[t,k]  + t_world_cam[t]      # (T,K,3) case
    """
    if points_cam.ndim not in (2, 3):
        raise ValueError(
            f"points_cam must be (T, 3) or (T, K, 3), got {points_cam.shape}"
        )
    if points_cam.shape[-1] != 3:
        raise ValueError(
            f"points_cam last dim must be 3 (xyz), got {points_cam.shape}"
        )
    _validate_T_world_cam(T_world_cam, points_cam.shape[0])

    R = T_world_cam[:, :3, :3].astype(np.float64)   # (T, 3, 3)
    t = T_world_cam[:, :3,  3].astype(np.float64)   # (T, 3)
    cam = points_cam.astype(np.float64)

    if cam.ndim == 2:
        # (T, 3) → world[t] = R[t] @ cam[t] + t[t]
        return np.einsum("tij,tj->ti", R, cam) + t
    # (T, K, 3) → world[t, k] = R[t] @ cam[t, k] + t[t]
    return np.einsum("tij,tkj->tki", R, cam) + t[:, None, :]


def transform_quats_cam_to_world(
    quat_xyzw_cam: np.ndarray,
    T_world_cam: np.ndarray,
) -> np.ndarray:
    """Lift per-frame cam-frame xyzw quats into world frame.

    Args:
        quat_xyzw_cam: (T, 4) scipy xyzw. NaN rows propagate.
        T_world_cam: (T, 4, 4) per-frame cam→world SE(3).

    Returns:
        (T, 4) world-frame xyzw quats; NaN where input row was NaN.

    Math (per valid t):
        R_cam_obj   = quat_to_matrix(q_cam[t])
        R_world_obj = R_world_cam[t] @ R_cam_obj
        q_world[t]  = matrix_to_quat(R_world_obj)
    """
    if quat_xyzw_cam.ndim != 2 or quat_xyzw_cam.shape[1] != 4:
        raise ValueError(
            f"quat_xyzw_cam must be (T, 4), got {quat_xyzw_cam.shape}"
        )
    _validate_T_world_cam(T_world_cam, quat_xyzw_cam.shape[0])

    out = np.full_like(quat_xyzw_cam, np.nan, dtype=np.float64)
    valid = np.isfinite(quat_xyzw_cam).all(axis=1)
    if not np.any(valid):
        return out

    # scipy.from_quat normalizes; we don't want side effects on NaN rows so
    # we slice valid rows only.
    R_cam_obj = Rotation.from_quat(quat_xyzw_cam[valid]).as_matrix()    # (Nv, 3, 3)
    R_world_cam_v = T_world_cam[valid, :3, :3].astype(np.float64)        # (Nv, 3, 3)
    R_world_obj = np.einsum("nij,njk->nik", R_world_cam_v, R_cam_obj)
    out[valid] = Rotation.from_matrix(R_world_obj).as_quat()
    return out
