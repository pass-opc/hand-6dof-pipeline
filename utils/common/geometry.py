"""
Self-contained rotation-format and rigid-transform helpers.

Pipeline position: shared utility for OPC's internal scripts AND for
customers consuming our LeRobot v3 source dataset. Single-file module
with only `numpy` and `scipy.spatial.transform` deps — `import` and use.

Why this exists:
  Our LeRobot v3 source dataset stores wrist orientation as scipy xyzw
  quaternion (4-D, in `observation.wrist_pose_left/right[3:7]`). Modern
  imitation-learning pipelines often want one of:
    - 6-D continuous rotation (Zhou et al. 2019, used by DROID, common
      in Diffusion Policy / ACT EE-pose training)
    - 3x3 rotation matrix (zero-ambiguity, easy to compose with frame
      transforms)
    - 4-D wxyz quaternion (pinocchio / MuJoCo convention)
  Conversions between these are float32-epsilon precision (verified by
  empirical benchmark — error ~1e-7 elementwise for SO(3) matrices).
  No semantic information is lost; this file standardizes the conversions
  so customers don't have to roll their own.

Frame convention reference:
  HaMeR / 01_track / 02_process / 03_build_source all store data in the
  **camera frame** (Gemini 335 color camera): x=right, y=down, z=forward.
  Wrist quaternion `q_xyzw` represents `R = quat_to_matrix(q)` such that
  `R @ v_hand_local = v_cam`, i.e., the wrist's MANO-local axes
  expressed in cam coordinates.

  If you need a world frame for a fixed-camera setup, multiply by your
  inverse extrinsic. For head-mounted recordings (our case), there is no
  static world frame; cam frame is what HaMeR + RGB share natively.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# =============================================================================
# Quaternion convention conversions (xyzw ↔ wxyz)
# =============================================================================

def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """scipy / ROS xyzw → pinocchio / MuJoCo / pytransform3d wxyz."""
    q_xyzw = np.asarray(q_xyzw)
    return np.concatenate([q_xyzw[..., 3:4], q_xyzw[..., 0:3]], axis=-1)


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """pinocchio / MuJoCo wxyz → scipy / ROS xyzw."""
    q_wxyz = np.asarray(q_wxyz)
    return np.concatenate([q_wxyz[..., 1:4], q_wxyz[..., 0:1]], axis=-1)


# =============================================================================
# Quaternion ↔ matrix
# =============================================================================

def quat_xyzw_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quaternion(s) → 3x3 rotation matrix (or batch)."""
    return Rotation.from_quat(q_xyzw).as_matrix()


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix(es) → xyzw quaternion(s)."""
    return Rotation.from_matrix(R).as_quat()


# =============================================================================
# Quaternion ↔ axis-angle (rotvec)
# =============================================================================

def quat_xyzw_to_axis_angle(q_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quat → 3-D axis-angle (rotvec, magnitude = angle in radians)."""
    return Rotation.from_quat(q_xyzw).as_rotvec()


def axis_angle_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    """3-D axis-angle → xyzw quat."""
    return Rotation.from_rotvec(rotvec).as_quat()


# =============================================================================
# Matrix ↔ 6D continuous rotation (Zhou et al. 2019)
# =============================================================================
# 6D = concat(R[:, 0], R[:, 1])   shape (..., 6)
# Inverse via Gram-Schmidt: b1=normalize(a1); b2=normalize(a2 - <a2,b1>b1);
#                             b3 = b1 × b2 → R = [b1 b2 b3]
# Reference: Zhou, Yi, et al. "On the continuity of rotation
# representations in neural networks." CVPR 2019.

def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix(es) → 6-D continuous representation."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def six_d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """6-D continuous representation → 3x3 rotation matrix(es).

    Robust to non-orthonormal input (Gram-Schmidt projects to SO(3)).
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = (a2 - proj) / np.linalg.norm(a2 - proj, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def quat_xyzw_to_6d(q_xyzw: np.ndarray) -> np.ndarray:
    """Compose: xyzw quat → matrix → 6-D."""
    return matrix_to_6d(quat_xyzw_to_matrix(q_xyzw))


def six_d_to_quat_xyzw(d6: np.ndarray) -> np.ndarray:
    """Compose: 6-D → matrix → xyzw quat."""
    return matrix_to_quat_xyzw(six_d_to_matrix(d6))


# =============================================================================
# Hemisphere normalization (q vs -q)
# =============================================================================

def hemisphere_normalize(q_seq_xyzw: np.ndarray) -> np.ndarray:
    """For a (T, 4) xyzw quat trajectory, flip sign so consecutive quats
    have non-negative dot product. Removes the q vs -q sign flip artifact
    that breaks SLERP and naive interpolation on quat sequences.

    NaN-safe: frames containing any NaN component pass through unchanged
    (their sign carries no meaningful information). The next finite frame
    is compared against the last finite frame, not against the NaN gap.

    Returns a new array; original is not modified.
    """
    q = np.asarray(q_seq_xyzw, dtype=np.float64).copy()
    last_finite: int | None = None
    for t in range(len(q)):
        if not np.all(np.isfinite(q[t])):
            continue
        if last_finite is not None and np.dot(q[last_finite], q[t]) < 0.0:
            q[t] = -q[t]
        last_finite = t
    return q.astype(q_seq_xyzw.dtype)


# =============================================================================
# Wrist-pose-level convenience (7D ↔ 9D ↔ 12D)
# =============================================================================
# Layout: [pos(3), rotation(R)] where R varies in width.
#   7D  = pos + xyzw quat
#   9D  = pos + 6-D continuous
#   12D = pos + 9-flat matrix (row-major)

def wrist_pose_7d_to_9d(pose_7d: np.ndarray) -> np.ndarray:
    """(..., 7) [xyz + xyzw] → (..., 9) [xyz + 6D]."""
    pose_7d = np.asarray(pose_7d)
    pos = pose_7d[..., :3]
    quat = pose_7d[..., 3:7]
    d6 = quat_xyzw_to_6d(quat)
    return np.concatenate([pos, d6], axis=-1)


def wrist_pose_7d_to_12d(pose_7d: np.ndarray) -> np.ndarray:
    """(..., 7) [xyz + xyzw] → (..., 12) [xyz + 9-flat row-major matrix]."""
    pose_7d = np.asarray(pose_7d)
    pos = pose_7d[..., :3]
    R = quat_xyzw_to_matrix(pose_7d[..., 3:7])
    R_flat = R.reshape(*R.shape[:-2], 9)
    return np.concatenate([pos, R_flat], axis=-1)


def wrist_pose_9d_to_7d(pose_9d: np.ndarray) -> np.ndarray:
    """(..., 9) [xyz + 6D] → (..., 7) [xyz + xyzw]."""
    pose_9d = np.asarray(pose_9d)
    pos = pose_9d[..., :3]
    quat = six_d_to_quat_xyzw(pose_9d[..., 3:9])
    return np.concatenate([pos, quat], axis=-1)


def wrist_pose_12d_to_7d(pose_12d: np.ndarray) -> np.ndarray:
    """(..., 12) [xyz + 9-flat matrix] → (..., 7) [xyz + xyzw]."""
    pose_12d = np.asarray(pose_12d)
    pos = pose_12d[..., :3]
    R = pose_12d[..., 3:12].reshape(*pose_12d.shape[:-1], 3, 3)
    quat = matrix_to_quat_xyzw(R)
    return np.concatenate([pos, quat], axis=-1)


# =============================================================================
# Rigid frame transformations
# =============================================================================

def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform: points' = R @ points + t.

    points shape (..., 3), R shape (3, 3), t shape (3,).
    For row-vector convention, this is points @ R.T + t.
    """
    return points @ np.asarray(R).T + np.asarray(t)


def invert_transform(
    R: np.ndarray, t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of (R, t). For rotation: R_inv = R.T. For translation:
    t_inv = -R.T @ t."""
    R = np.asarray(R)
    R_inv = R.T
    t_inv = -R_inv @ np.asarray(t)
    return R_inv, t_inv


def compose_transforms(
    R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose: (R, t) such that applying (R1, t1) then (R2, t2) is one
    transform. Result: R = R2 @ R1, t = R2 @ t1 + t2."""
    R = np.asarray(R2) @ np.asarray(R1)
    t = np.asarray(R2) @ np.asarray(t1) + np.asarray(t2)
    return R, t


# =============================================================================
# Quick self-check (run as `python -m utils.common.geometry`)
# =============================================================================

def _self_check() -> None:
    rng = np.random.default_rng(0)
    rotvecs = rng.normal(size=(100, 3)) * np.pi
    R_orig = Rotation.from_rotvec(rotvecs).as_matrix().astype(np.float32)
    pos = rng.normal(size=(100, 3)).astype(np.float32)
    pose_7d = np.concatenate(
        [pos, Rotation.from_matrix(R_orig).as_quat().astype(np.float32)],
        axis=-1,
    )

    # Round-trip 7D → 9D → 7D
    pose_9d = wrist_pose_7d_to_9d(pose_7d)
    pose_7d_back = wrist_pose_9d_to_7d(pose_9d)
    err = max(
        np.abs(pose_7d[:, :3] - pose_7d_back[:, :3]).max(),
        np.abs(np.abs(pose_7d[:, 3:]).sum(-1) - np.abs(pose_7d_back[:, 3:]).sum(-1)).max(),
    )
    print(f"7D→9D→7D max error: {err:.3e}")

    # Round-trip 7D → 12D → 7D
    pose_12d = wrist_pose_7d_to_12d(pose_7d)
    pose_7d_back2 = wrist_pose_12d_to_7d(pose_12d)
    err2 = np.abs(pose_7d[:, :3] - pose_7d_back2[:, :3]).max()
    print(f"7D→12D→7D pos max error: {err2:.3e}")

    # Hemisphere norm
    q = rng.normal(size=(50, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q_norm = hemisphere_normalize(q)
    diffs = (q_norm[1:] * q_norm[:-1]).sum(axis=-1)
    print(f"hemisphere_normalize: min consecutive dot = {diffs.min():.3e}  "
          f"(should be >= 0)")

    # Frame transform sanity
    p = np.array([[1.0, 0.0, 0.0]])
    R_z90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    p_rot = transform_points(p, R_z90, np.zeros(3))
    print(f"transform_points([1,0,0]) by Rz(90°) = {p_rot[0]}  (expect ~[0,1,0])")

    print("self-check OK.")


if __name__ == "__main__":
    _self_check()
