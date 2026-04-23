"""
Tests for utils/pose_util.py

Verifies:
  1. pose_to_mat / mat_to_pose roundtrip consistency
  2. rvec_tvec_to_mat matches OpenCV convention
  3. invert_transform correctness (T @ T_inv = I)
  4. Batched operations work correctly
  5. Cross-check against UMI's pose_util (if available)

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_pose_util.py -v
    # or simply:
    python tests/test_pose_util.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.pose_util import (
    pose_to_mat,
    mat_to_pose,
    rvec_tvec_to_pose,
    rvec_tvec_to_mat,
    invert_transform,
    transform_pose,
    mat_to_rot6d,
    mat_to_pose9d,
)

ATOL = 1e-10  # numerical tolerance for float64 comparisons


def test_pose_mat_roundtrip_identity():
    """Identity pose [0,0,0, 0,0,0] should give 4x4 identity matrix."""
    pose = np.zeros(6)
    mat = pose_to_mat(pose)
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat, np.eye(4), atol=ATOL)

    # Roundtrip: mat -> pose -> mat
    pose_back = mat_to_pose(mat)
    np.testing.assert_allclose(pose_back, pose, atol=ATOL)


def test_pose_mat_roundtrip_translation_only():
    """Pure translation: rotation part should be identity."""
    pose = np.array([0.1, -0.2, 0.3, 0, 0, 0])
    mat = pose_to_mat(pose)

    # Check translation column
    np.testing.assert_allclose(mat[:3, 3], [0.1, -0.2, 0.3], atol=ATOL)
    # Check rotation is identity
    np.testing.assert_allclose(mat[:3, :3], np.eye(3), atol=ATOL)

    # Roundtrip
    pose_back = mat_to_pose(mat)
    np.testing.assert_allclose(pose_back, pose, atol=ATOL)


def test_pose_mat_roundtrip_rotation_only():
    """Pure rotation: 90° around z-axis."""
    angle = np.pi / 2
    pose = np.array([0, 0, 0, 0, 0, angle])  # axis-angle: z-axis * 90°
    mat = pose_to_mat(pose)

    # Expected rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    np.testing.assert_allclose(mat[:3, :3], expected_R, atol=ATOL)
    np.testing.assert_allclose(mat[:3, 3], [0, 0, 0], atol=ATOL)

    # Roundtrip
    pose_back = mat_to_pose(mat)
    np.testing.assert_allclose(pose_back, pose, atol=ATOL)


def test_pose_mat_roundtrip_full():
    """Full 6DoF: random but reproducible pose."""
    rng = np.random.default_rng(42)
    pose = rng.standard_normal(6) * 0.5  # moderate values

    mat = pose_to_mat(pose)
    pose_back = mat_to_pose(mat)
    mat_back = pose_to_mat(pose_back)

    np.testing.assert_allclose(pose_back, pose, atol=1e-8)
    np.testing.assert_allclose(mat_back, mat, atol=1e-8)


def test_pose_mat_batched():
    """Batched input: (T, 6) -> (T, 4, 4) and back."""
    rng = np.random.default_rng(123)
    poses = rng.standard_normal((10, 6)) * 0.3

    mats = pose_to_mat(poses)
    assert mats.shape == (10, 4, 4)

    poses_back = mat_to_pose(mats)
    assert poses_back.shape == (10, 6)
    np.testing.assert_allclose(poses_back, poses, atol=1e-8)


def test_rvec_tvec_to_pose():
    """rvec/tvec concatenation matches expected format."""
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([1.0, 2.0, 3.0])

    pose = rvec_tvec_to_pose(rvec, tvec)

    # pose = [tvec, rvec] = [x, y, z, rx, ry, rz]
    np.testing.assert_allclose(pose[:3], tvec, atol=ATOL)
    np.testing.assert_allclose(pose[3:], rvec, atol=ATOL)


def test_rvec_tvec_to_mat():
    """rvec/tvec -> mat should match pose_to_mat(rvec_tvec_to_pose(...))."""
    rvec = np.array([0.5, -0.3, 0.1])
    tvec = np.array([0.1, 0.2, 0.3])

    mat_direct = rvec_tvec_to_mat(rvec, tvec)
    mat_via_pose = pose_to_mat(rvec_tvec_to_pose(rvec, tvec))

    np.testing.assert_allclose(mat_direct, mat_via_pose, atol=ATOL)


def test_invert_transform_identity():
    """Inverse of identity is identity."""
    I = np.eye(4)
    I_inv = invert_transform(I)
    np.testing.assert_allclose(I_inv, I, atol=ATOL)


def test_invert_transform_correctness():
    """T @ T_inv should equal identity for any rigid transform."""
    rng = np.random.default_rng(99)
    pose = rng.standard_normal(6) * 0.5
    T = pose_to_mat(pose)
    T_inv = invert_transform(T)

    product = T @ T_inv
    np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    # Also check the other direction
    product2 = T_inv @ T
    np.testing.assert_allclose(product2, np.eye(4), atol=1e-10)


def test_invert_transform_vs_numpy():
    """Our fast SE(3) inverse should match np.linalg.inv."""
    rng = np.random.default_rng(77)
    pose = rng.standard_normal(6)
    T = pose_to_mat(pose)

    our_inv = invert_transform(T)
    np_inv = np.linalg.inv(T)

    np.testing.assert_allclose(our_inv, np_inv, atol=1e-10)


def test_invert_transform_batched():
    """Batched inversion: (N, 4, 4)."""
    rng = np.random.default_rng(55)
    poses = rng.standard_normal((5, 6))
    mats = pose_to_mat(poses)

    mats_inv = invert_transform(mats)
    assert mats_inv.shape == (5, 4, 4)

    for i in range(5):
        product = mats[i] @ mats_inv[i]
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


def test_transform_pose():
    """Chained transform: T_A_C = T_A_B @ T_B_C."""
    rng = np.random.default_rng(11)
    pose_B_C = rng.standard_normal(6) * 0.3
    T_A_B = pose_to_mat(rng.standard_normal(6) * 0.3)

    result = transform_pose(T_A_B, pose_B_C)

    # Manual: convert to mat, multiply, convert back
    expected_mat = T_A_B @ pose_to_mat(pose_B_C)
    expected = mat_to_pose(expected_mat)

    np.testing.assert_allclose(result, expected, atol=1e-8)


def test_mat_to_rot6d_shape():
    """rot6d is first two rows of rotation matrix, flattened."""
    mat = pose_to_mat(np.array([0, 0, 0, 0.1, 0.2, 0.3]))
    R = mat[:3, :3]  # extract 3x3 rotation from 4x4
    rot6d = mat_to_rot6d(R)
    assert rot6d.shape == (6,)

    # Should be R[:2, :].flatten() = first two rows of R
    expected = R[:2, :].flatten()
    np.testing.assert_allclose(rot6d, expected, atol=ATOL)


def test_mat_to_pose9d():
    """pose10d = [pos3, rot6d6] = 9D total.

    Note: UMI names this "pose10d" but the actual output is 9D (pos3 + rot6d6).
    The name is inherited from UMI for consistency.
    """
    pose6 = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    mat = pose_to_mat(pose6)
    pose_out = mat_to_pose9d(mat)

    # pos(3) + rot6d(6) = 9D
    assert pose_out.shape == (9,), f"Expected (9,), got {pose_out.shape}"
    np.testing.assert_allclose(pose_out[:3], [1.0, 2.0, 3.0], atol=ATOL)


# =============================================================================
# Cross-check with UMI (optional, only runs if UMI code is accessible)
# =============================================================================

def test_cross_check_with_umi():
    """Compare our output with UMI's pose_util for the same input."""
    umi_path = Path(__file__).resolve().parents[2] / "UMI"
    if not (umi_path / "umi" / "common" / "pose_util.py").exists():
        print("  [SKIP] UMI code not found, skipping cross-check")
        return

    sys.path.insert(0, str(umi_path))
    from umi.common.pose_util import (
        pose_to_mat as umi_pose_to_mat,
        mat_to_pose as umi_mat_to_pose,
        mat_to_pose10d as umi_mat_to_pose9d,  # UMI calls it "10d" but it's actually 9D
    )

    rng = np.random.default_rng(42)
    for _ in range(20):
        pose = rng.standard_normal(6) * 0.5

        our_mat = pose_to_mat(pose)
        umi_mat = umi_pose_to_mat(pose)
        np.testing.assert_allclose(our_mat, umi_mat, atol=1e-10,
                                   err_msg="pose_to_mat mismatch with UMI")

        our_pose = mat_to_pose(our_mat)
        umi_pose = umi_mat_to_pose(umi_mat)
        np.testing.assert_allclose(our_pose, umi_pose, atol=1e-10,
                                   err_msg="mat_to_pose mismatch with UMI")

        our_10d = mat_to_pose9d(our_mat)
        umi_10d = umi_mat_to_pose9d(umi_mat)
        np.testing.assert_allclose(our_10d, umi_10d, atol=1e-10,
                                   err_msg="mat_to_pose9d mismatch with UMI")

    print("  [PASS] All 20 random poses match UMI output exactly")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {name}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
