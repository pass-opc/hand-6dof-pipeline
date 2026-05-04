"""
Smoke tests for the SO-arm 101 retarget backend.

Scope: confirm the mink-based retarget chain (cam-frame pinch midpoint
+ wrist orientation → 6-joint qpos via FrameTask soft IK) produces
well-formed output and respects the basic protocol contract. IK error
budget is NOT pinned here — pos / rot errors are 5-DoF-vs-6-DoF
trade-offs and live in qpos meta `extras` for per-batch inspection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retarget import RetargetResult, get_backend
from retarget.loader import load_npz_source
from retarget.so101 import JOINT_NAMES, So101Backend, _R_ARM_CAM_OPENCV


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_IPHONE_BATCH = (
    _PROJECT_ROOT / "output" / "iphone" / "pour_coffee_bean" / "02_processed"
)


# -------- backend protocol --------

def test_registry_resolves_so101_to_backend():
    cls = get_backend("so101", "mujoco")
    assert cls is So101Backend
    cls_real = get_backend("so101", "real")
    assert cls_real is So101Backend


def test_required_keys_includes_per_hand_fields():
    keys = So101Backend.required_keys("right")
    # No wrist_cam (we use joints_cam[4..8] pinch midpoint instead).
    assert "right_joints_cam" in keys
    assert "right_wrist_quat_cam" in keys
    assert "right_confidence" in keys
    assert "right_quality_passed" in keys
    assert "right_trim_first" in keys
    assert "right_trim_last" in keys


def test_joint_names_canonical_order():
    b = So101Backend(robot="so101", hand="right")
    assert b.joint_names == [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    assert b.joint_names == JOINT_NAMES
    assert b.n_joints == 6


def test_backend_rejects_wrong_robot_or_hand():
    with pytest.raises(ValueError, match="so101"):
        So101Backend(robot="koch", hand="right")
    with pytest.raises(ValueError, match="hand"):
        So101Backend(robot="so101", hand="middle")


# -------- cam→arm rotation invariants --------

def test_R_arm_cam_is_orthonormal_right_handed():
    R = _R_ARM_CAM_OPENCV
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_R_arm_cam_maps_axes_correctly():
    R = _R_ARM_CAM_OPENCV
    # cam +Z (forward, optical axis) → arm +X (forward)
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])
    # cam +Y (down) → arm -Z (down is -up)
    assert np.allclose(R @ [0, 1, 0], [0, 0, -1])
    # cam +X (right) → arm -Y (right is -left)
    assert np.allclose(R @ [1, 0, 0], [0, -1, 0])


# -------- end-to-end on real iPhone-line episode --------

@pytest.fixture(scope="module")
def iphone_ep02_source():
    npz = _IPHONE_BATCH / "pour_coffee_bean_episode02" / "pour_coffee_bean_episode02.processed.npz"
    if not npz.exists():
        pytest.skip(f"iPhone batch not present: {npz}")
    return load_npz_source(npz)


@pytest.fixture(scope="module")
def iphone_ep02_result(iphone_ep02_source):
    backend = So101Backend(robot="so101", hand="right")
    return backend.retarget_episode(iphone_ep02_source, "right")


def test_e2e_shape_and_validity(iphone_ep02_source, iphone_ep02_result):
    res = iphone_ep02_result
    assert isinstance(res, RetargetResult)
    first, last = iphone_ep02_source.trim_range("right")
    T_trim = last - first
    assert res.qpos.shape == (T_trim, 6)
    assert res.qpos_valid.shape == (T_trim,)
    assert int(res.qpos_valid.sum()) > 0


def test_e2e_extras_have_diagnostic_fields(iphone_ep02_result):
    e = iphone_ep02_result.extras
    assert e["ik_engine"] == "mink"
    assert "ik_pos_err_mean_mm" in e
    assert "ik_rot_err_mean_deg" in e
    assert "workspace_scale_used" in e
    assert "cam_pos_arm" in e
    assert "cam_quat_arm_xyzw" in e


def test_e2e_wrist_flex_actually_moves(iphone_ep02_result):
    """Regression check: legacy ikpy 4-joint position IK left wrist_flex
    stuck near 0 due to redundant DoF. mink weighted soft IK should
    actually use wrist_flex when the source motion has tilt."""
    qv = iphone_ep02_result.qpos[iphone_ep02_result.qpos_valid]
    wf_idx = JOINT_NAMES.index("wrist_flex")
    wf_range_deg = float(np.degrees(qv[:, wf_idx].max() - qv[:, wf_idx].min()))
    assert wf_range_deg > 5.0, (
        f"wrist_flex range {wf_range_deg:.1f}° suspiciously small; "
        f"check IK is actually using wrist orient."
    )
