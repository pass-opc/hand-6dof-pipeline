"""
Tests for utils/r3d_reader.py

Focus:
  1. read_poses — shape, rotation orthogonality, [qx,qy,qz,qw] order
  2. Landscape rotation compensation: world→camera pose must follow the
     CCW 90° image rotation applied by iter_r3d_frames
  3. read_iphone_intrinsics — portrait native, landscape swap
  4. Error paths — missing poses, wrong shape

No real .r3d file is needed; synthetic ZIP archives are built per-test.

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_r3d_reader.py -v
"""

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.r3d_reader import (
    needs_rotation,
    read_iphone_intrinsics,
    read_poses,
)


# ============================================================
# Helpers
# ============================================================

def _write_r3d(
    tmp_path: Path,
    poses,
    w: int = 480,
    h: int = 640,
    K=None,
    name: str = "t.r3d",
) -> Path:
    """Minimal .r3d ZIP (no rgbd frames — only metadata exercised here)."""
    r3d = tmp_path / name
    if K is None:
        # column-major 3x3: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        K = [500.0, 0, 0, 0, 500.0, 0, w / 2.0, h / 2.0, 1]
    meta = {"w": w, "h": h, "K": K}
    if poses is not None:
        meta["poses"] = poses
    with zipfile.ZipFile(r3d, "w") as zf:
        zf.writestr("metadata", json.dumps(meta))
    return r3d


def _identity_pose():
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # [qx, qy, qz, qw, tx, ty, tz]


# ============================================================
# 1. needs_rotation
# ============================================================

class TestNeedsRotation:

    def test_portrait_no_rotate(self):
        assert needs_rotation({"w": 480, "h": 640}) is False

    def test_landscape_rotate(self):
        assert needs_rotation({"w": 640, "h": 480}) is True

    def test_square_no_rotate(self):
        # not strictly landscape — leave untouched
        assert needs_rotation({"w": 480, "h": 480}) is False


# ============================================================
# 2. read_poses — shape, identity, quaternion order
# ============================================================

class TestReadPosesBasic:

    def test_identity_shape_and_values(self, tmp_path):
        """Identity quaternion + zero translation → identity T."""
        r3d = _write_r3d(tmp_path, poses=[_identity_pose()])
        T = read_poses(r3d)
        assert T.shape == (1, 4, 4)
        assert T.dtype == np.float64
        np.testing.assert_allclose(T[0], np.eye(4), atol=1e-12)

    def test_multi_frame_shape(self, tmp_path):
        r3d = _write_r3d(tmp_path, poses=[_identity_pose()] * 7)
        T = read_poses(r3d)
        assert T.shape == (7, 4, 4)

    def test_translation_preserved(self, tmp_path):
        pose = [0.0, 0.0, 0.0, 1.0, 0.1, -0.2, 0.3]  # identity R, offset t
        r3d = _write_r3d(tmp_path, poses=[pose])
        T = read_poses(r3d)
        np.testing.assert_allclose(T[0, :3, 3], [0.1, -0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(T[0, :3, :3], np.eye(3), atol=1e-12)

    def test_rotation_is_orthonormal(self, tmp_path):
        """All per-frame rotations must satisfy R @ R.T == I."""
        rng = np.random.default_rng(0)
        poses = []
        for _ in range(10):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            poses.append([q[0], q[1], q[2], q[3], 0.0, 0.0, 0.0])
        r3d = _write_r3d(tmp_path, poses=poses)
        T = read_poses(r3d)
        for i in range(10):
            R = T[i, :3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_quaternion_xyzw_order(self, tmp_path):
        """Record3D stores [qx,qy,qz,qw]; verify we don't swap it."""
        # 90° rotation around world X → scipy quat = [sin(45°), 0, 0, cos(45°)]
        s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
        pose = [s, 0.0, 0.0, c, 0.0, 0.0, 0.0]
        r3d = _write_r3d(tmp_path, poses=[pose])
        T = read_poses(r3d)
        expected = Rotation.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
        np.testing.assert_allclose(T[0, :3, :3], expected, atol=1e-10)


# ============================================================
# 3. Landscape rotation compensation
# ============================================================

class TestReadPosesLandscape:

    def test_portrait_no_compensation(self, tmp_path):
        """When w <= h, the pose is returned unmodified."""
        pose = _identity_pose()
        r3d = _write_r3d(tmp_path, poses=[pose], w=480, h=640)
        T = read_poses(r3d)
        np.testing.assert_allclose(T[0, :3, :3], np.eye(3), atol=1e-12)

    def test_landscape_applies_ccw90_camera_adjustment(self, tmp_path):
        """Landscape recordings: T_world_cam must get a right-multiply by
        R_raw_from_port so that the pose matches the CCW 90° image rotation
        applied in iter_r3d_frames.

            R_raw_from_port = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
            R_wc_port       = R_wc_raw @ R_raw_from_port
        """
        pose = _identity_pose()  # R_wc_raw = I
        r3d = _write_r3d(tmp_path, poses=[pose], w=640, h=480)
        T = read_poses(r3d)

        R_raw_from_port = np.array(
            [[0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(T[0, :3, :3], R_raw_from_port, atol=1e-12)
        # Translation is unchanged by the right-multiply
        np.testing.assert_allclose(T[0, :3, 3], [0, 0, 0], atol=1e-12)

    def test_landscape_translation_preserved(self, tmp_path):
        pose = [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.5]
        r3d = _write_r3d(tmp_path, poses=[pose], w=640, h=480)
        T = read_poses(r3d)
        np.testing.assert_allclose(T[0, :3, 3], [0.5, 1.0, 1.5], atol=1e-12)


# ============================================================
# 4. Error paths
# ============================================================

class TestReadPosesErrors:

    def test_missing_poses_raises(self, tmp_path):
        r3d = _write_r3d(tmp_path, poses=None)
        with pytest.raises(ValueError, match="poses"):
            read_poses(r3d)

    def test_wrong_shape_raises(self, tmp_path):
        # 6 floats instead of 7
        r3d = _write_r3d(tmp_path, poses=[[0.0] * 6])
        with pytest.raises(ValueError, match="shape"):
            read_poses(r3d)


# ============================================================
# 5. read_iphone_intrinsics
# ============================================================

class TestIntrinsics:

    def test_portrait_native(self):
        """When not rotated, K is unchanged."""
        metadata = {
            "w": 480,
            "h": 640,
            "K": [500.0, 0, 0, 0, 500.0, 0, 240.0, 320.0, 1],
        }
        K = read_iphone_intrinsics(metadata)
        np.testing.assert_allclose(
            K,
            [[500.0, 0, 240.0], [0, 500.0, 320.0], [0, 0, 1]],
            atol=1e-10,
        )

    def test_landscape_swap(self):
        """Landscape → CCW 90° rotation swaps fx/fy and remaps principal point."""
        W_orig = 640
        fx, fy, cx, cy = 500.0, 510.0, 300.0, 240.0
        metadata = {
            "w": W_orig, "h": 480,
            "K": [fx, 0, 0, 0, fy, 0, cx, cy, 1],
        }
        K = read_iphone_intrinsics(metadata)
        # Expected after CCW 90°: fx_new=fy, fy_new=fx, cx_new=cy, cy_new=W-1-cx
        np.testing.assert_allclose(K[0, 0], fy, atol=1e-10)
        np.testing.assert_allclose(K[1, 1], fx, atol=1e-10)
        np.testing.assert_allclose(K[0, 2], cy, atol=1e-10)
        np.testing.assert_allclose(K[1, 2], W_orig - 1 - cx, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
