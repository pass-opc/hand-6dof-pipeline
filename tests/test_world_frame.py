"""
Tests for utils/process/world_frame.py — cam-frame → world-frame primitives.

Coverage:
  - identity: T_world_cam = I → output equals input
  - pure translation: cam pose offset → output translates by that amount
  - pure rotation: cam rotated 90° → points + quats rotate consistently
  - composed: rotation + translation
  - shape support: (T, 3) wrist + (T, K, 3) MANO joints
  - NaN propagation: any non-finite input row → NaN row out, no crash
  - shape validation: bad shape → ValueError
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from utils.process.world_frame import (
    transform_points_cam_to_world,
    transform_quats_cam_to_world,
)


# ---------------- helpers ----------------

def _identity_T(T: int) -> np.ndarray:
    """T × identity SE(3)."""
    return np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))


def _make_T_world_cam(R_world_cam: np.ndarray, t_world_cam: np.ndarray, T: int) -> np.ndarray:
    """Repeat one (3,3) rot + (3,) trans across T frames."""
    out = _identity_T(T)
    out[:, :3, :3] = R_world_cam
    out[:, :3, 3] = t_world_cam
    return out


# ---------------- transform_points_cam_to_world ----------------

class TestTransformPoints:
    def test_identity_T_world_cam_returns_input_for_2D(self):
        pts = np.random.RandomState(0).randn(5, 3)
        out = transform_points_cam_to_world(pts, _identity_T(5))
        assert np.allclose(out, pts)

    def test_identity_T_world_cam_returns_input_for_3D(self):
        pts = np.random.RandomState(0).randn(5, 21, 3)
        out = transform_points_cam_to_world(pts, _identity_T(5))
        assert np.allclose(out, pts)

    def test_pure_translation(self):
        pts = np.zeros((3, 3))                  # 3 frames, point at origin
        T_wc = _make_T_world_cam(np.eye(3), np.array([1.0, 2.0, 3.0]), 3)
        out = transform_points_cam_to_world(pts, T_wc)
        assert np.allclose(out, np.tile([1.0, 2.0, 3.0], (3, 1)))

    def test_pure_rotation_90deg_about_z(self):
        # Rotation about world Z by +90°: cam +X → world +Y
        R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        T_wc = _make_T_world_cam(R, np.zeros(3), 1)
        pts = np.array([[[1.0, 0.0, 0.0]]])     # (1, 1, 3): cam +X
        out = transform_points_cam_to_world(pts, T_wc)
        assert np.allclose(out, [[[0.0, 1.0, 0.0]]], atol=1e-10)

    def test_translation_independent_of_rotation_axis(self):
        # General SE(3): rotation + translation
        R = Rotation.from_euler("xyz", [30, 45, 60], degrees=True).as_matrix()
        t = np.array([0.1, -0.2, 0.5])
        T_wc = _make_T_world_cam(R, t, 1)
        pts = np.array([[1.0, 2.0, 3.0]])       # (1, 3)
        out = transform_points_cam_to_world(pts, T_wc)
        # Hand-compute: world = R @ [1,2,3] + t
        expected = R @ np.array([1.0, 2.0, 3.0]) + t
        assert np.allclose(out[0], expected)

    def test_per_frame_independence(self):
        # Each frame uses its own T_world_cam — different cam poses
        T_wc = _identity_T(2)
        T_wc[1, :3, 3] = [10.0, 0.0, 0.0]       # frame 1 has +10 in X
        pts = np.array([[1.0, 2.0, 3.0],         # frame 0
                        [1.0, 2.0, 3.0]])        # frame 1
        out = transform_points_cam_to_world(pts, T_wc)
        assert np.allclose(out[0], [1.0, 2.0, 3.0])
        assert np.allclose(out[1], [11.0, 2.0, 3.0])

    def test_nan_propagates_2D(self):
        pts = np.array([[1.0, 2.0, 3.0],
                        [np.nan, np.nan, np.nan]])
        out = transform_points_cam_to_world(pts, _identity_T(2))
        assert np.allclose(out[0], [1.0, 2.0, 3.0])
        assert np.all(np.isnan(out[1]))

    def test_nan_propagates_3D_per_joint(self):
        pts = np.zeros((2, 21, 3))
        pts[0, 5, :] = np.nan       # frame 0, joint 5 missing
        out = transform_points_cam_to_world(pts, _identity_T(2))
        assert np.all(np.isnan(out[0, 5, :]))
        assert np.allclose(out[0, 0, :], [0, 0, 0])  # other joints untouched
        assert np.allclose(out[1], 0.0)

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="must be"):
            transform_points_cam_to_world(np.zeros((5,)), _identity_T(5))
        with pytest.raises(ValueError, match="last dim must be 3"):
            transform_points_cam_to_world(np.zeros((5, 4)), _identity_T(5))
        with pytest.raises(ValueError, match="length"):
            transform_points_cam_to_world(np.zeros((5, 3)), _identity_T(4))


# ---------------- transform_quats_cam_to_world ----------------

class TestTransformQuats:
    def test_identity_returns_input(self):
        q = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
        q_arr = np.tile(q, (3, 1))
        out = transform_quats_cam_to_world(q_arr, _identity_T(3))
        # Allow sign-flip (q and -q represent same rotation)
        for i in range(3):
            R_in = Rotation.from_quat(q_arr[i]).as_matrix()
            R_out = Rotation.from_quat(out[i]).as_matrix()
            assert np.allclose(R_in, R_out, atol=1e-10)

    def test_compose_world_cam_with_cam_obj(self):
        # cam→world rotation = +90° about Z
        # cam→obj rotation = +90° about X
        # Expected world→obj = R_world_cam @ R_cam_obj
        R_wc = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        R_co = Rotation.from_euler("x", 90, degrees=True).as_matrix()
        T_wc = _make_T_world_cam(R_wc, np.zeros(3), 1)
        q_co = Rotation.from_matrix(R_co).as_quat()[None, :]    # (1, 4)
        out = transform_quats_cam_to_world(q_co, T_wc)
        R_out = Rotation.from_quat(out[0]).as_matrix()
        R_expected = R_wc @ R_co
        assert np.allclose(R_out, R_expected, atol=1e-10)

    def test_translation_does_not_affect_quat(self):
        # Quat composition doesn't depend on translation
        q = Rotation.from_euler("y", 45, degrees=True).as_quat()
        q_arr = q[None, :]
        T_wc = _make_T_world_cam(np.eye(3), np.array([100, 200, 300]), 1)
        out = transform_quats_cam_to_world(q_arr, T_wc)
        # Identity rotation + any translation → out should equal in (mod sign)
        R_in = Rotation.from_quat(q).as_matrix()
        R_out = Rotation.from_quat(out[0]).as_matrix()
        assert np.allclose(R_in, R_out, atol=1e-10)

    def test_nan_propagates(self):
        q1 = Rotation.from_euler("z", 30, degrees=True).as_quat()
        q_arr = np.array([q1, [np.nan, np.nan, np.nan, np.nan]])
        out = transform_quats_cam_to_world(q_arr, _identity_T(2))
        assert np.all(np.isfinite(out[0]))
        assert np.all(np.isnan(out[1]))

    def test_all_nan_returns_all_nan(self):
        q_arr = np.full((3, 4), np.nan)
        out = transform_quats_cam_to_world(q_arr, _identity_T(3))
        assert np.all(np.isnan(out))

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="must be"):
            transform_quats_cam_to_world(np.zeros((5,)), _identity_T(5))
        with pytest.raises(ValueError, match="must be"):
            transform_quats_cam_to_world(np.zeros((5, 3)), _identity_T(5))
        with pytest.raises(ValueError, match="length"):
            transform_quats_cam_to_world(np.zeros((5, 4)), _identity_T(4))


# ---------------- end-to-end consistency ----------------

class TestEndToEndConsistency:
    """Position transform of (R, t) cam-frame point should equal the
    transform of orientation-only data composed with translation. Verifies
    points and quats use the same SE(3) convention."""

    def test_compose_rotation_position_consistent(self):
        # If we transform a point at the origin with arbitrary T_world_cam,
        # we get t_world_cam back. If we transform the identity quaternion,
        # we get the rotation part back.
        R = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
        t = np.array([0.5, -0.3, 0.7])
        T_wc = _make_T_world_cam(R, t, 1)

        # Position: cam-frame origin → world-frame t
        out_pos = transform_points_cam_to_world(np.zeros((1, 3)), T_wc)
        assert np.allclose(out_pos[0], t)

        # Orientation: cam-frame identity quat → world-frame R
        q_id = np.array([[0, 0, 0, 1]], dtype=np.float64)   # xyzw identity
        out_q = transform_quats_cam_to_world(q_id, T_wc)
        R_out = Rotation.from_quat(out_q[0]).as_matrix()
        assert np.allclose(R_out, R, atol=1e-10)
