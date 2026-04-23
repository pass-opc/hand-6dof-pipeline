"""
Tests for scripts/02_process.py (processing layer).

Covers:
  1. _validate_input — coord_frame='camera' + dual-hand + required keys
  2. trim_hand — slices all per-frame arrays; handles all-NaN case
  3. world_transform — identity pass-through; translation; rotation
  4. center_to_anchor — median-of-first-N shift, NaN preserved, joints_cam untouched
  5. quality_check — detection rate / max gap / duration rejections
  6. normalize_gripper — min/max mapping, flat signal → zeros
  7. build_states_and_actions — shape, dtype, tail-holds-pose
  8. process_hand end-to-end — all-NaN hand, good trim
  9. process_episode — dual-hand output, output schema

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_process.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "process_mod",
    str(Path(__file__).resolve().parent.parent / "scripts" / "02_process.py"),
)
_proc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proc)

ProcessConfig = _proc.ProcessConfig
_validate_input = _proc._validate_input
_extract_hand_cam = _proc._extract_hand_cam
trim_hand = _proc.trim_hand
world_transform = _proc.world_transform
center_to_anchor = _proc.center_to_anchor
quality_check = _proc.quality_check
normalize_gripper = _proc.normalize_gripper
build_states_and_actions = _proc.build_states_and_actions
process_hand = _proc.process_hand
process_episode = _proc.process_episode


# ============================================================
# Helpers
# ============================================================

def _make_cam_hand(n: int, valid_mask: np.ndarray | None = None,
                   base_pos=(0.1, 0.2, 0.5)):
    """Build a cam-frame single-hand dict with per-frame arrays."""
    wrist_cam = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        wrist_cam[i] = [base_pos[0] + i * 0.01,
                        base_pos[1] + i * 0.01,
                        base_pos[2] + i * 0.01]
    wrist_rot_cam = np.zeros((n, 3), dtype=np.float64)
    joints_cam = np.zeros((n, 21, 3), dtype=np.float64)
    for t in range(n):
        for j in range(21):
            joints_cam[t, j] = [j * 0.01, 0.0, 0.5 + t * 0.01]
    bbox = np.tile([0.0, 0.0, 10.0, 10.0], (n, 1)).astype(np.float64)
    gripper_width = np.full(n, 0.05, dtype=np.float64)
    confidence = np.full(n, 0.9, dtype=np.float64)

    if valid_mask is not None:
        invalid = ~valid_mask
        wrist_cam[invalid] = np.nan
        wrist_rot_cam[invalid] = np.nan
        joints_cam[invalid] = np.nan
        bbox[invalid] = np.nan
        gripper_width[invalid] = np.nan
        confidence[invalid] = 0.0

    return {
        "wrist_cam":     wrist_cam,
        "wrist_rot_cam": wrist_rot_cam,
        "joints_cam":    joints_cam,
        "bbox":          bbox,
        "gripper_width": gripper_width,
        "confidence":    confidence,
    }


def _make_tracking(
    n: int = 10,
    hand: str = "right",
    valid_mask: np.ndarray | None = None,
    coord_frame: str = "camera",
    T_world_cam: np.ndarray | None = None,
    base_pos=(0.1, 0.2, 0.5),
) -> dict:
    """Build a dual-hand cam-frame tracking dict (01 output shape)."""
    if hand == "right":
        left = _make_cam_hand(n, valid_mask=np.zeros(n, dtype=bool))
        right = _make_cam_hand(n, valid_mask=valid_mask, base_pos=base_pos)
    else:
        left = _make_cam_hand(n, valid_mask=valid_mask, base_pos=base_pos)
        right = _make_cam_hand(n, valid_mask=np.zeros(n, dtype=bool))

    if T_world_cam is None:
        T_world_cam = np.tile(np.eye(4), (n, 1, 1))

    t = {
        "timestamps":   np.arange(n, dtype=np.float64) / 30.0,
        "T_world_cam":  T_world_cam,
        "K":            np.eye(3),
        "left_hand":    left,
        "right_hand":   right,
        "source":       "mock",
        "episode_name": "ep0",
    }
    if coord_frame is not None:
        t["coord_frame"] = coord_frame
    return t


# ============================================================
# 1. _validate_input
# ============================================================

class TestValidateInput:

    def test_accepts_camera_frame_dual_hand(self):
        t = _make_tracking()
        _validate_input(t, "ep0")  # no raise

    def test_rejects_missing_coord_frame(self):
        t = _make_tracking(coord_frame=None)
        with pytest.raises(ValueError, match="coord_frame"):
            _validate_input(t, "ep0")

    def test_rejects_wrong_coord_frame(self):
        t = _make_tracking(coord_frame="world")
        with pytest.raises(ValueError, match="coord_frame"):
            _validate_input(t, "ep0")

    def test_rejects_missing_hand_key(self):
        t = _make_tracking()
        del t["right_hand"]
        with pytest.raises(ValueError, match="dual-hand"):
            _validate_input(t, "ep0")

    def test_rejects_missing_K(self):
        t = _make_tracking()
        del t["K"]
        with pytest.raises(ValueError, match="K"):
            _validate_input(t, "ep0")


# ============================================================
# 2. trim_hand
# ============================================================

class TestTrimHand:

    def test_no_boundary_nan_preserves_length(self):
        t = _make_tracking(n=5, hand="right")
        cam = _extract_hand_cam(t, "right")
        trimmed, slc = trim_hand(cam)
        assert slc == (0, 5)
        assert len(trimmed["wrist_cam"]) == 5

    def test_trims_leading_and_trailing(self):
        valid = np.array([False, False, True, True, True, True, False])
        t = _make_tracking(n=7, hand="right", valid_mask=valid)
        cam = _extract_hand_cam(t, "right")
        trimmed, slc = trim_hand(cam)
        assert slc == (2, 6)
        assert len(trimmed["wrist_cam"]) == 4
        # T_world_cam / timestamps must be sliced the same way
        assert len(trimmed["T_world_cam"]) == 4
        assert len(trimmed["timestamps"]) == 4
        # First frame in trimmed slice must be valid (original index 2)
        assert not np.isnan(trimmed["wrist_cam"][0, 0])

    def test_all_nan_returns_none(self):
        valid = np.zeros(4, dtype=bool)
        t = _make_tracking(n=4, hand="right", valid_mask=valid)
        cam = _extract_hand_cam(t, "right")
        assert trim_hand(cam) is None


# ============================================================
# 3. world_transform
# ============================================================

class TestWorldTransform:

    def test_identity_pose_passthrough(self):
        t = _make_tracking(n=3, hand="right", base_pos=(0.1, -0.2, 0.7))
        cam = _extract_hand_cam(t, "right")
        out = world_transform(cam)
        # Identity T_world_cam → world == cam
        np.testing.assert_allclose(
            out["eef_pos"][0], [0.1, -0.2, 0.7], atol=1e-10,
        )

    def test_translation_pose(self):
        """t_wc = (1, 2, 3): wrist_world = wrist_cam + (1, 2, 3)."""
        n = 2
        T_wc = np.tile(np.eye(4), (n, 1, 1))
        T_wc[:, :3, 3] = [1.0, 2.0, 3.0]
        t = _make_tracking(n=n, hand="right", T_world_cam=T_wc,
                           base_pos=(0.1, 0.2, 0.5))
        cam = _extract_hand_cam(t, "right")
        out = world_transform(cam)
        np.testing.assert_allclose(
            out["eef_pos"][0],
            [0.1 + 1.0, 0.2 + 2.0, 0.5 + 3.0], atol=1e-10,
        )

    def test_rotation_pose(self):
        """90° around world Z: cam (1, 0, z) → world (0, 1, z)."""
        n = 2
        R_wc = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
        T_wc = np.tile(np.eye(4), (n, 1, 1))
        T_wc[:, :3, :3] = R_wc
        t = _make_tracking(n=n, hand="right", T_world_cam=T_wc,
                           base_pos=(1.0, 0.0, 0.5))
        cam = _extract_hand_cam(t, "right")
        out = world_transform(cam)
        np.testing.assert_allclose(
            out["eef_pos"][0], [0.0, 1.0, 0.5], atol=1e-10,
        )

    def test_nan_preserved(self):
        valid = np.array([False, True, True])
        t = _make_tracking(n=3, hand="right", valid_mask=valid)
        cam = _extract_hand_cam(t, "right")
        out = world_transform(cam)
        assert np.all(np.isnan(out["eef_pos"][0]))
        assert not np.any(np.isnan(out["eef_pos"][1]))


# ============================================================
# 4. center_to_anchor (median-of-first-N)
# ============================================================

class TestCenterToAnchor:

    def test_three_axis_shift_median_of_available(self):
        # 5 frames all valid, positions +0.01/frame → median = frame 2
        t = _make_tracking(n=5, hand="right", base_pos=(1.0, 2.0, 3.0))
        cam = _extract_hand_cam(t, "right")
        w = world_transform(cam)
        out, origin = center_to_anchor(w, window_n=10)

        # Median of 5 linear frames = middle frame = base + 0.02
        np.testing.assert_allclose(origin, [1.02, 2.02, 3.02], atol=1e-10)
        # Frame 2 (the median) centered at 0
        np.testing.assert_allclose(out["eef_pos"][2], [0, 0, 0], atol=1e-10)
        # Frame 0 is 2 steps before median → -0.02
        np.testing.assert_allclose(
            out["eef_pos"][0], [-0.02, -0.02, -0.02], atol=1e-10,
        )

    def test_outlier_in_window_is_ignored_by_median(self):
        # 11 valid frames, frame 0 is a huge outlier, rest linear.
        # Median of 10 (window_n=10) tolerates 1 outlier easily.
        t = _make_tracking(n=11, hand="right", base_pos=(1.0, 2.0, 3.0))
        cam = _extract_hand_cam(t, "right")
        w = world_transform(cam)
        # Inject an outlier at frame 0
        w["eef_pos"][0] = [99.0, 99.0, 99.0]
        out, origin = center_to_anchor(w, window_n=10)

        # Inliers in window: frames 1..9 (positions base+0.01..base+0.09)
        # Plus the outlier [99,99,99]. Median of these 10 = midpoint of
        # sorted values; robust to the single outlier.
        # Expected: around base + 0.05 (midway of inliers)
        assert origin[0] < 2.0, f"Outlier leaked into anchor: {origin}"

    def test_nan_prefix_preserved(self):
        # Only frames 2,3,4 valid → window reduces to 3 frames
        # Positions: base+0.02, base+0.03, base+0.04 → median = base+0.03
        valid = np.array([False, False, True, True, True])
        t = _make_tracking(n=5, hand="right", valid_mask=valid,
                           base_pos=(10.0, 20.0, 30.0))
        cam = _extract_hand_cam(t, "right")
        w = world_transform(cam)
        out, origin = center_to_anchor(w, window_n=10)

        np.testing.assert_allclose(
            origin, [10.03, 20.03, 30.03], atol=1e-10,
        )
        # Frame 3 (the median) is centered at 0
        np.testing.assert_allclose(out["eef_pos"][3], [0, 0, 0], atol=1e-10)
        assert np.all(np.isnan(out["eef_pos"][:2]))

    def test_joints_cam_unchanged(self):
        t = _make_tracking(n=3, hand="right", base_pos=(5.0, 5.0, 5.0))
        cam = _extract_hand_cam(t, "right")
        w = world_transform(cam)
        joints_before = w["joints_cam"].copy()
        out, _ = center_to_anchor(w, window_n=10)
        np.testing.assert_array_equal(out["joints_cam"], joints_before)

    def test_all_nan_returns_zero_origin(self):
        valid = np.zeros(3, dtype=bool)
        t = _make_tracking(n=3, hand="right", valid_mask=valid)
        cam = _extract_hand_cam(t, "right")
        w = world_transform(cam)
        out, origin = center_to_anchor(w, window_n=10)
        np.testing.assert_allclose(origin, [0, 0, 0], atol=1e-10)


# ============================================================
# 5. quality_check
# ============================================================

class TestQualityCheck:

    def _make_for_quality(self, n=100, valid_mask=None):
        """Shortcut: build the intermediate dict quality_check expects."""
        eef_pos = np.zeros((n, 3), dtype=np.float64)
        if valid_mask is not None:
            eef_pos[~valid_mask] = np.nan
        return {"eef_pos": eef_pos}

    def test_clean_passes(self):
        data = self._make_for_quality(n=300)
        config = ProcessConfig(fps=60, min_detect_rate=0.5,
                               max_consecutive_gap=30, min_duration_s=2.0)
        ok, reason = quality_check(data, config)
        assert ok, f"Expected pass, got: {reason}"

    def test_low_detect_rate_fails(self):
        valid = np.zeros(100, dtype=bool)
        valid[:10] = True
        data = self._make_for_quality(n=100, valid_mask=valid)
        config = ProcessConfig(fps=60, min_detect_rate=0.5,
                               max_consecutive_gap=200, min_duration_s=0.1)
        ok, reason = quality_check(data, config)
        assert not ok
        assert "detection rate" in reason

    def test_large_gap_fails(self):
        valid = np.ones(200, dtype=bool)
        valid[50:150] = False  # 100-frame gap
        data = self._make_for_quality(n=200, valid_mask=valid)
        config = ProcessConfig(fps=60, min_detect_rate=0.2,
                               max_consecutive_gap=30, min_duration_s=0.1)
        ok, reason = quality_check(data, config)
        assert not ok
        assert "gap" in reason

    def test_short_duration_fails(self):
        data = self._make_for_quality(n=30)  # 0.5 s @ 60 fps
        config = ProcessConfig(fps=60, min_detect_rate=0.5,
                               max_consecutive_gap=100, min_duration_s=2.0)
        ok, reason = quality_check(data, config)
        assert not ok
        assert "duration" in reason


# ============================================================
# 6. normalize_gripper
# ============================================================

class TestNormalizeGripper:

    def test_basic_range(self):
        gw = np.array([0.0, 0.5, 1.0])
        out = normalize_gripper(gw)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-6)
        assert out.dtype == np.float32

    def test_scaled_range(self):
        gw = np.array([0.02, 0.04, 0.08])  # min=0.02 max=0.08
        out = normalize_gripper(gw)
        np.testing.assert_allclose(
            out, [0.0, (0.04 - 0.02) / 0.06, 1.0], atol=1e-6,
        )

    def test_flat_signal_returns_zeros(self):
        gw = np.array([0.05, 0.05, 0.05])
        out = normalize_gripper(gw)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])


# ============================================================
# 7. build_states_and_actions
# ============================================================

class TestStatesAndActions:

    def test_shape_and_dtype(self):
        T = 5
        pos = np.arange(T * 3, dtype=np.float64).reshape(T, 3)
        rot = pos * 0.1
        grip = np.linspace(0, 1, T).astype(np.float32)
        states, actions = build_states_and_actions(pos, rot, grip)
        assert states.shape == (T, 7)
        assert actions.shape == (T, 7)
        assert states.dtype == np.float32

    def test_action_equals_next_state(self):
        T = 4
        pos = np.arange(T * 3, dtype=np.float64).reshape(T, 3)
        rot = np.zeros((T, 3))
        grip = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
        states, actions = build_states_and_actions(pos, rot, grip)
        np.testing.assert_allclose(actions[:-1], states[1:], atol=1e-6)

    def test_tail_holds_last(self):
        T = 3
        pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
        rot = np.zeros((T, 3))
        grip = np.zeros(T, dtype=np.float32)
        states, actions = build_states_and_actions(pos, rot, grip)
        np.testing.assert_allclose(actions[-1], states[-1], atol=1e-6)


# ============================================================
# 8. process_hand end-to-end
# ============================================================

class TestProcessHand:

    def test_all_nan_hand(self):
        t = _make_tracking(n=5, hand="right",
                           valid_mask=np.zeros(5, dtype=bool))
        cam = _extract_hand_cam(t, "right")
        config = ProcessConfig(fps=60, min_duration_s=0.01,
                               enable_filter=False)
        out = process_hand(cam, config, "right")
        assert out["quality_passed"] is False
        assert out["trim_slice"] is None

    def test_good_hand_passes(self):
        """Simple 120-frame all-valid run @ 60 fps — should pass."""
        n = 120
        t = _make_tracking(n=n, hand="right")
        cam = _extract_hand_cam(t, "right")
        config = ProcessConfig(fps=60, min_detect_rate=0.5,
                               max_consecutive_gap=30,
                               min_duration_s=1.0,
                               enable_filter=False)
        out = process_hand(cam, config, "right")
        assert out["quality_passed"] is True
        assert out["state"].shape == (n, 7)
        assert out["action"].shape == (n, 7)
        assert out["trim_slice"] == (0, n)
        # center_to_anchor (default window_n=10): origin = median of frames
        # 0..9 at +0.01/frame = base_pos + 0.145 (mean of sorted idx 4,5).
        # Frame 0 sits at base - origin = -0.045 on every axis.
        np.testing.assert_allclose(
            out["center_offset_world"], [0.145, 0.245, 0.545], atol=1e-10,
        )
        np.testing.assert_allclose(
            out["wrist_world"][0], [-0.045, -0.045, -0.045], atol=1e-10,
        )


# ============================================================
# 9. process_episode end-to-end
# ============================================================

class TestProcessEpisode:

    def test_dual_hand_output_schema(self):
        n = 120
        t = _make_tracking(n=n, hand="right")
        # Make left hand also valid so both pass
        t["left_hand"] = _make_cam_hand(n, base_pos=(-0.1, 0.2, 0.5))

        config = ProcessConfig(fps=60, min_detect_rate=0.5,
                               max_consecutive_gap=30,
                               min_duration_s=1.0,
                               enable_filter=False)
        out = process_episode(t, config)

        assert out["coord_frame"] == "episode_local"
        assert "timestamps" in out
        assert "T_world_cam" in out
        assert "K" in out
        assert out["episode_name"] == "ep0"
        for hand in ("left_hand", "right_hand"):
            h = out[hand]
            assert h["quality_passed"] is True
            assert h["state"].shape == (n, 7)
            assert h["action"].shape == (n, 7)
            assert "center_offset_world" in h
            assert "trim_slice" in h


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
