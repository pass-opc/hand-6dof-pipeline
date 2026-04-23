"""
Tests for scripts/04_replay_on_arm.py

Covers:
  1. compute_T_arm_world — default translation, world Y→arm Z, rotate_deg,
     flip, orthonormality
  2. retarget_trajectory — position/rotation transform, gripper mapping
  3. smooth_joint_trajectory — jump clamping + moving average
  4. workspace_check — pass/fail conditions
  5. IK + wrist_roll (skipped when URDF absent)

The legacy load_calibration function is gone; workspace placement is now
parametrised via CLI flags that feed compute_T_arm_world.

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_replay_on_arm.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

spec = importlib.util.spec_from_file_location(
    "replay_on_arm",
    Path(__file__).resolve().parent.parent / "scripts" / "04_replay_on_arm.py",
)
replay_on_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_on_arm)

compute_T_arm_world = replay_on_arm.compute_T_arm_world
retarget_trajectory = replay_on_arm.retarget_trajectory
compute_ik_trajectory = replay_on_arm.compute_ik_trajectory
smooth_joint_trajectory = replay_on_arm.smooth_joint_trajectory
workspace_check = replay_on_arm.workspace_check
build_ik_chain = replay_on_arm.build_ik_chain
get_revolute_indices = replay_on_arm.get_revolute_indices
_find_wrist_roll_index = replay_on_arm._find_wrist_roll_index
_extract_wrist_roll_from_orientation = replay_on_arm._extract_wrist_roll_from_orientation

URDF_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "so101_new_calib.urdf"
)


# ============================================================
# 1. compute_T_arm_world
# ============================================================
class TestComputeTArmWorld:

    def test_default_translation(self):
        """Default: +X 0.30 m from arm base_link, table at base height."""
        T = compute_T_arm_world()
        np.testing.assert_allclose(T[:3, 3], [0.30, 0.0, 0.0], atol=1e-12)

    def test_custom_translation(self):
        T = compute_T_arm_world(distance=0.5, table_height=0.1)
        np.testing.assert_allclose(T[:3, 3], [0.5, 0.0, 0.1], atol=1e-12)

    def test_world_y_maps_to_arm_z(self):
        """World +Y (gravity-up) must land on arm +Z after the rotation part."""
        T = compute_T_arm_world()  # no flip, no rotate_deg
        v_world = np.array([0.0, 1.0, 0.0])
        v_arm = T[:3, :3] @ v_world
        np.testing.assert_allclose(v_arm, [0.0, 0.0, 1.0], atol=1e-12)

    def test_world_z_maps_to_arm_neg_y(self):
        """Consistency: world +Z must go to arm -Y with default rotate_deg=0."""
        T = compute_T_arm_world()
        v = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-12)

    def test_rotate_deg_90_around_arm_z(self):
        """rotate_deg=90 rotates the mapped frame 90° around arm +Z."""
        # world +X → arm +X with rotate=0; with rotate=90, → arm +Y
        T = compute_T_arm_world(rotate_deg=90.0)
        v = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-10)

    def test_flip_mirrors_world_x(self):
        """flip=True mirrors world +X (useful for right→left hand retargeting)."""
        T_flip = compute_T_arm_world(flip=True)
        v = T_flip[:3, :3] @ np.array([1.0, 0.0, 0.0])
        # Without flip world +X → arm +X. With flip → arm -X.
        np.testing.assert_allclose(v, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_rotation_orthogonal_no_flip(self):
        T = compute_T_arm_world()
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_flip_gives_reflection(self):
        T = compute_T_arm_world(flip=True)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        # Flip introduces a reflection → det = -1
        assert abs(np.linalg.det(R) + 1.0) < 1e-10

    def test_homogeneous_bottom_row(self):
        T = compute_T_arm_world(distance=0.42, table_height=-0.05,
                                rotate_deg=33.0, flip=True)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)


# ============================================================
# 2. retarget_trajectory
# ============================================================
class TestRetarget:

    def test_identity_transform(self):
        T = np.eye(4)
        positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        rotations = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        grippers = np.array([0.0, 1.0])

        pos_r, rot_r, grip_deg = retarget_trajectory(
            positions, rotations, grippers, T,
        )
        np.testing.assert_allclose(pos_r, positions, atol=1e-10)
        assert grip_deg[0] == 0.0
        assert grip_deg[1] == 100.0

    def test_translation_only(self):
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        positions = np.array([[0.0, 0.0, 0.0]])
        rotations = np.array([[0.0, 0.0, 0.0]])
        grippers = np.array([0.5])

        pos_r, _, _ = retarget_trajectory(
            positions, rotations, grippers, T,
        )
        np.testing.assert_allclose(pos_r[0], [1.0, 2.0, 3.0], atol=1e-10)

    def test_rotation_applied_to_positions(self):
        T = np.eye(4)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        positions = np.array([[1.0, 0.0, 0.0]])
        rotations = np.array([[0.0, 0.0, 0.0]])
        grippers = np.array([0.0])

        pos_r, _, _ = retarget_trajectory(
            positions, rotations, grippers, T,
        )
        np.testing.assert_allclose(pos_r[0], [0.0, 1.0, 0.0], atol=1e-10)

    def test_rotation_applied_to_orientations(self):
        R_rw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T = np.eye(4)
        T[:3, :3] = R_rw
        rotations = np.array([[0.0, 0.0, 0.0]])
        positions = np.array([[0.0, 0.0, 0.0]])
        grippers = np.array([0.0])

        _, rot_r, _ = retarget_trajectory(
            positions, rotations, grippers, T,
        )
        np.testing.assert_allclose(rot_r[0], R_rw, atol=1e-10)

    def test_gripper_mapping(self):
        T = np.eye(4)
        positions = np.array([[0.0, 0.0, 0.0]])
        rotations = np.zeros((1, 3))
        grippers = np.array([0.5])

        _, _, grip_deg = retarget_trajectory(
            positions, rotations, grippers, T,
            gripper_open_deg=10.0, gripper_close_deg=90.0,
        )
        assert abs(grip_deg[0] - 50.0) < 1e-10


# ============================================================
# 3. smooth_joint_trajectory
# ============================================================
class TestSmooth:

    def test_no_jumps_unchanged(self):
        T, N = 50, 5
        angles = np.tile(np.linspace(0, 30, T), (N, 1)).T
        result = smooth_joint_trajectory(angles, max_jump_deg=20.0, window_size=1)
        np.testing.assert_allclose(result, angles, atol=1e-10)

    def test_jump_clamped(self):
        angles = np.array([[0.0], [0.0], [50.0], [0.0], [0.0]])
        result = smooth_joint_trajectory(angles, max_jump_deg=20.0, window_size=1)
        assert abs(result[2, 0]) < 25.0

    def test_moving_average_smooths(self):
        rng = np.random.default_rng(42)
        base = np.linspace(0, 90, 100)
        noisy = base + rng.normal(0, 2, 100)
        angles = noisy.reshape(-1, 1)
        result = smooth_joint_trajectory(angles, max_jump_deg=50.0, window_size=5)
        assert np.std(result[:, 0]) < np.std(angles[:, 0])


# ============================================================
# 4. workspace_check
# ============================================================
class TestWorkspaceCheck:

    def test_good_trajectory_passes(self, capsys):
        T, N = 100, 7
        angles = np.zeros((T, N))
        angles[:, 1] = np.linspace(0, 45, T)
        angles[:, 3] = np.linspace(-20, 20, T)
        gripper = np.linspace(0, 50, T)
        ik_errors = np.zeros(T)
        active_indices = [1, 2, 3, 4, 5]
        ok = workspace_check(angles, gripper, ik_errors, active_indices, 60, 0.5)
        assert ok is True
        assert "All checks passed" in capsys.readouterr().out

    def test_high_ik_error_fails(self, capsys):
        T, N = 50, 7
        angles = np.zeros((T, N))
        gripper = np.zeros(T)
        ik_errors = np.ones(T) * 0.02
        active_indices = [1, 2, 3, 4, 5]
        ok = workspace_check(angles, gripper, ik_errors, active_indices, 60, 0.5)
        assert ok is False
        assert "FAIL" in capsys.readouterr().out


# ============================================================
# 5. IK + wrist_roll with real URDF
# ============================================================
@pytest.mark.skipif(not URDF_PATH.exists(), reason="URDF not found")
class TestIKWithURDF:

    def test_position_only_ik(self):
        chain = build_ik_chain(URDF_PATH)
        positions = np.array([[0.15, -0.05, 0.10]])
        angles_deg, errors = compute_ik_trajectory(
            chain, positions, orientations=None,
        )
        assert errors[0] < 0.001
        assert angles_deg.shape == (1, len(chain.links))

    def test_ik_with_orientation_preserves_position(self):
        chain = build_ik_chain(URDF_PATH)
        positions = np.array([[0.15, -0.05, 0.10]])
        orientations = np.array([np.eye(3)])
        angles_deg, errors = compute_ik_trajectory(
            chain, positions, orientations=orientations,
        )
        assert errors[0] < 0.001

    def test_wrist_roll_index_found(self):
        chain = build_ik_chain(URDF_PATH)
        idx = _find_wrist_roll_index(chain)
        assert idx is not None
        assert chain.links[idx].name == "wrist_roll"

    def test_revolute_indices(self):
        chain = build_ik_chain(URDF_PATH)
        indices = get_revolute_indices(chain)
        assert len(indices) == 5

    def test_continuous_ik_trajectory(self):
        chain = build_ik_chain(URDF_PATH)
        T = 20
        t = np.linspace(0, 1, T)
        positions = np.column_stack([
            0.15 + 0.05 * np.sin(t * np.pi),
            -0.05 * np.ones(T),
            0.10 + 0.05 * np.cos(t * np.pi),
        ])
        angles_deg, errors = compute_ik_trajectory(chain, positions)
        assert errors.max() < 0.001
        diffs = np.abs(np.diff(angles_deg, axis=0))
        assert diffs.max() < 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
