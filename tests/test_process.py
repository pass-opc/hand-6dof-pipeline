"""
Tests for scripts/02_process.py + utils/process/core.py (raw-first processing).

Covers:
  1. utils/process/core.py:
     - trim_leading_trailing_invalid (confidence-based, IoU-based)
     - quality_check (detection rate, gap, duration)
     - rotation_jump_diagnostic (hemisphere-aware quat dot)
     - process_hand top-level orchestration
  2. scripts/02_process.py:
     - _axis_angle_to_quat_hemisphere conversion + hemisphere continuity
     - npz save/load roundtrip preserving full schema + trim_first/last flags
     - quality-fail hand kept in npz with quality_passed=False (raw post-trim)

Run:
    cd code/opc_data_pipeline
    python -m pytest tests/test_process.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.process.core import (
    ProcessConfig, _max_consecutive_zeros, process_hand,
    quality_check, rotation_jump_diagnostic, trim_leading_trailing_invalid,
)
from scripts._schema import HAND_FIELDS, make_empty_hand_arrays


_spec = importlib.util.spec_from_file_location(
    "iphone_02_process_mod",
    str(_PROJECT_ROOT / "scripts" / "02_process.py"),
)
_proc = importlib.util.module_from_spec(_spec)
sys.modules["iphone_02_process_mod"] = _proc
_spec.loader.exec_module(_proc)


# ============================================================
# Helpers
# ============================================================

def _make_hand(n: int, valid_mask: np.ndarray | None = None,
               base_pos=(0.1, 0.2, 0.5)):
    """Build a per-hand HAND_FIELDS dict (post axis-angle→quat conversion)."""
    h = make_empty_hand_arrays(n)
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not valid_mask[i]:
            continue
        h["wrist_cam"][i] = [base_pos[0] + i * 0.01,
                              base_pos[1] + i * 0.01,
                              base_pos[2] + i * 0.01]
        h["wrist_quat_cam"][i] = [0.0, 0.0, 0.0, 1.0]  # identity xyzw
        for j in range(21):
            h["joints_cam"][i, j] = [j * 0.01, 0.0, 0.5 + i * 0.01]
        h["bbox"][i] = [0.0, 0.0, 100.0, 100.0]
        h["confidence"][i] = 0.9
    return h


# ============================================================
# 1. utils/process/core.py — trim_leading_trailing_invalid
# ============================================================

class TestTrim:

    def test_no_boundary_invalid_keeps_all(self):
        h = _make_hand(5)
        sl, trimmed, stats = trim_leading_trailing_invalid(h)
        assert (sl.start, sl.stop) == (0, 5)
        assert len(trimmed["confidence"]) == 5
        assert stats["n_trimmed_leading"] == 0
        assert stats["n_trimmed_trailing"] == 0

    def test_leading_trailing_invalid_trimmed(self):
        valid = np.array([False, False, True, True, True, True, False])
        h = _make_hand(7, valid_mask=valid)
        sl, trimmed, stats = trim_leading_trailing_invalid(h)
        assert (sl.start, sl.stop) == (2, 6)
        assert len(trimmed["confidence"]) == 4
        assert stats["n_trimmed_leading"] == 2
        assert stats["n_trimmed_trailing"] == 1

    def test_mid_invalid_kept(self):
        """NaN frames inside trim are preserved (raw-first principle)."""
        valid = np.array([True, False, False, True, True])
        h = _make_hand(5, valid_mask=valid)
        sl, trimmed, _ = trim_leading_trailing_invalid(h)
        assert (sl.start, sl.stop) == (0, 5)
        # frames 1-2 are still NaN inside the trim
        assert np.isnan(trimmed["wrist_cam"][1, 0])
        assert np.isnan(trimmed["wrist_cam"][2, 0])

    def test_all_invalid_returns_empty(self):
        h = _make_hand(5, valid_mask=np.zeros(5, dtype=bool))
        sl, trimmed, _ = trim_leading_trailing_invalid(h)
        assert (sl.start, sl.stop) == (0, 0)
        assert len(trimmed["confidence"]) == 0

    def test_iou_trim_rejects_misaligned_bbox(self):
        """When MP bbox + projected joint cloud disagree, trim that frame.

        Joints in _make_hand are at z=0.5+t*0.01, x=[0..0.20], y=0. With
        K=[[500,0,320],[0,500,240],[0,0,1]] (px units), projected u spans
        roughly 320..520 and v=240. Joint cloud bbox is [312,232,528,248]
        (with 8-px padding). Set MP bbox to overlap that for valid frames;
        frame 0's MP bbox is far away to trigger IoU rejection there only.
        """
        n = 4
        h = _make_hand(n)
        K = np.array([[500.0, 0, 320.0],
                      [0, 500.0, 240.0],
                      [0, 0, 1.0]])
        good_bbox = np.array([320.0, 235.0, 525.0, 245.0])
        for i in range(n):
            h["bbox"][i] = good_bbox
        # Frame 0 misaligned far from the joint projection bbox.
        h["bbox"][0] = np.array([10000.0, 10000.0, 10100.0, 10100.0])
        sl, trimmed, stats = trim_leading_trailing_invalid(
            h, K=K, bbox_iou_threshold=0.3,
        )
        assert sl.start == 1, f"Expected leading trim, got {sl}"
        assert stats["n_iou_rejected_total"] == 1


# ============================================================
# 2. utils/process/core.py — quality_check
# ============================================================

class TestQualityCheck:

    def test_clean_passes(self):
        h = _make_hand(60)  # 60 frames @ 30fps = 2s
        cfg = ProcessConfig(min_detection_rate=0.5, max_gap_frames=30,
                            min_duration_s=1.0)
        ok, reason = quality_check(h, fps=30.0, cfg=cfg)
        assert ok, reason

    def test_low_detect_rate_fails(self):
        valid = np.zeros(60, dtype=bool)
        valid[:10] = True
        h = _make_hand(60, valid_mask=valid)
        cfg = ProcessConfig(min_detection_rate=0.5, max_gap_frames=200,
                            min_duration_s=0.1)
        ok, reason = quality_check(h, fps=30.0, cfg=cfg)
        assert not ok
        assert "low_detection" in reason

    def test_long_gap_fails(self):
        valid = np.ones(120, dtype=bool)
        valid[30:90] = False  # 60-frame gap
        h = _make_hand(120, valid_mask=valid)
        cfg = ProcessConfig(min_detection_rate=0.2, max_gap_frames=30,
                            min_duration_s=0.1)
        ok, reason = quality_check(h, fps=30.0, cfg=cfg)
        assert not ok
        assert "long_gap" in reason

    def test_too_short_fails(self):
        h = _make_hand(20)  # 20 frames @ 30fps = 0.67s
        cfg = ProcessConfig(min_detection_rate=0.5, max_gap_frames=100,
                            min_duration_s=1.0)
        ok, reason = quality_check(h, fps=30.0, cfg=cfg)
        assert not ok
        assert "too_short" in reason

    def test_max_consecutive_zeros(self):
        m = np.array([True, False, False, True, False, False, False, True])
        assert _max_consecutive_zeros(m) == 3
        assert _max_consecutive_zeros(np.ones(5, dtype=bool)) == 0


# ============================================================
# 3. utils/process/core.py — rotation_jump_diagnostic
# ============================================================

class TestRotationJump:

    def test_zero_jumps_for_constant_quat(self):
        h = _make_hand(10)  # all identity quat
        out = rotation_jump_diagnostic(h, threshold_rad=0.5)
        assert out["n_jumps_above_threshold"] == 0
        assert out["max_delta_rad"] < 1e-6

    def test_detects_large_jump(self):
        h = _make_hand(3)
        # Frame 1 → flip 90° around X
        h["wrist_quat_cam"][1] = Rotation.from_rotvec(
            [np.pi / 2, 0, 0]
        ).as_quat()
        out = rotation_jump_diagnostic(h, threshold_rad=0.5)
        # Two consecutive deltas: identity→90°X (large), 90°X→identity (large)
        assert out["n_jumps_above_threshold"] >= 1
        assert out["max_delta_rad"] > 1.0

    def test_hemisphere_aware(self):
        """q and -q are the same rotation; dot's abs() handles flip."""
        h = _make_hand(2)
        h["wrist_quat_cam"][1] = -h["wrist_quat_cam"][0]  # sign flip
        out = rotation_jump_diagnostic(h, threshold_rad=0.5)
        assert out["max_delta_rad"] < 1e-6


# ============================================================
# 4. utils/process/core.py — process_hand top-level
# ============================================================

class TestProcessHand:

    def test_passes_clean_hand(self):
        h = _make_hand(60)
        cfg = ProcessConfig(min_detection_rate=0.5, max_gap_frames=30,
                            min_duration_s=1.0,
                            bbox_iou_trim_threshold=0.0)
        processed, stats = process_hand(h, fps=30.0, cfg=cfg)
        assert stats["quality_passed"]
        assert processed is not None
        assert len(processed["confidence"]) == 60
        # raw-first: signal values should be untouched (no smoothing)
        np.testing.assert_array_equal(
            processed["wrist_cam"], h["wrist_cam"][:],
        )

    def test_fails_returns_none_with_reason(self):
        h = _make_hand(10)  # too short
        cfg = ProcessConfig(min_duration_s=10.0)
        processed, stats = process_hand(h, fps=30.0, cfg=cfg)
        assert not stats["quality_passed"]
        assert processed is None
        assert "too_short" in stats["skip_reason"]

    def test_preserves_mid_nan(self):
        """Inside trim, NaN frames remain NaN (raw-first principle)."""
        valid = np.ones(60, dtype=bool)
        valid[20:25] = False
        h = _make_hand(60, valid_mask=valid)
        cfg = ProcessConfig(min_duration_s=1.0,
                            bbox_iou_trim_threshold=0.0)
        processed, stats = process_hand(h, fps=30.0, cfg=cfg)
        assert stats["quality_passed"]
        # rows 20..24 still NaN
        assert np.all(np.isnan(processed["wrist_cam"][20:25]))


# ============================================================
# 5. scripts/02_process.py — axis-angle → quat hemisphere
# ============================================================

class TestAxisAngleToQuat:

    def test_identity_passes_through(self):
        n = 5
        rotvec = np.zeros((n, 3))
        quat = _proc._axis_angle_to_quat_hemisphere(rotvec)
        # scipy identity quat is (0, 0, 0, 1)
        for i in range(n):
            np.testing.assert_allclose(quat[i], [0, 0, 0, 1], atol=1e-10)

    def test_nan_preserved(self):
        rotvec = np.full((3, 3), np.nan)
        rotvec[1] = [0.0, 0.0, np.pi / 4]
        quat = _proc._axis_angle_to_quat_hemisphere(rotvec)
        assert np.all(np.isnan(quat[0]))
        assert np.all(np.isfinite(quat[1]))
        assert np.all(np.isnan(quat[2]))

    def test_hemisphere_continuity_flips_sign(self):
        """Two consecutive quats that differ by sign are flipped to align."""
        rotvec = np.zeros((2, 3))
        # frame 0: rotation by +π around z → quat (0,0,1,0) approx
        rotvec[0] = [0, 0, np.pi - 0.01]
        # frame 1: rotation by -π around z → quat (0,0,-1,0) approx (same rot)
        rotvec[1] = [0, 0, -(np.pi - 0.01)]
        quat = _proc._axis_angle_to_quat_hemisphere(rotvec)
        # After hemisphere fix, dot product should be >= 0
        assert float(np.dot(quat[0], quat[1])) >= 0.0


# ============================================================
# 6. scripts/02_process.py — npz save/load roundtrip
# ============================================================

class TestNpzRoundtrip:

    def test_full_schema_present(self, tmp_path):
        """Every HAND_FIELDS key + trim_first/last + quality_passed must be in npz."""
        n = 60
        ts_us = (np.arange(n, dtype=np.float64) / 30.0 * 1e6).astype(np.int64)
        K = np.eye(3, dtype=np.float64)
        T_wc = np.tile(np.eye(4), (n, 1, 1))

        left_raw = _make_hand(n)
        right_raw = _make_hand(n, valid_mask=np.zeros(n, dtype=bool))

        cfg = ProcessConfig(min_duration_s=1.0,
                             bbox_iou_trim_threshold=0.0)
        l_proc, l_stats = process_hand(left_raw, fps=30.0, cfg=cfg)
        r_proc, r_stats = process_hand(right_raw, fps=30.0, cfg=cfg)

        out_path = tmp_path / "ep0.processed.npz"
        _proc._save_processed_npz(
            out_path,
            timestamps_us=ts_us, K=K, T_world_cam=T_wc,
            source="hamer", episode_name="ep0",
            raw_hands={"left": left_raw, "right": right_raw},
            processed_hands={"left": l_proc, "right": r_proc},
            stats_per_hand={"left": l_stats, "right": r_stats},
        )

        d = np.load(out_path, allow_pickle=True)
        # All hand fields present for BOTH hands
        for hand_name in ("left", "right"):
            for f in HAND_FIELDS:
                assert f"{hand_name}_{f}" in d.files
            assert f"{hand_name}_trim_first" in d.files
            assert f"{hand_name}_trim_last" in d.files
            assert f"{hand_name}_quality_passed" in d.files
        # T_world_cam + K + ts preserved
        assert "T_world_cam" in d.files
        assert d["T_world_cam"].shape == (n, 4, 4)

    def test_failed_hand_still_in_npz(self, tmp_path):
        """quality_passed=False hand keeps raw post-trim values, not None."""
        n = 30
        ts_us = (np.arange(n, dtype=np.float64) / 30.0 * 1e6).astype(np.int64)
        K = np.eye(3, dtype=np.float64)
        T_wc = np.tile(np.eye(4), (n, 1, 1))

        # Right hand passes, left hand fails (too short).
        left_raw = _make_hand(n, valid_mask=np.zeros(n, dtype=bool))
        right_raw = _make_hand(n)

        cfg_strict = ProcessConfig(min_duration_s=10.0,
                                    bbox_iou_trim_threshold=0.0)
        cfg_lenient = ProcessConfig(min_duration_s=0.1,
                                     bbox_iou_trim_threshold=0.0)
        l_proc, l_stats = process_hand(left_raw, fps=30.0, cfg=cfg_strict)
        r_proc, r_stats = process_hand(right_raw, fps=30.0, cfg=cfg_lenient)

        out_path = tmp_path / "ep0.processed.npz"
        _proc._save_processed_npz(
            out_path,
            timestamps_us=ts_us, K=K, T_world_cam=T_wc,
            source="hamer", episode_name="ep0",
            raw_hands={"left": left_raw, "right": right_raw},
            processed_hands={"left": l_proc, "right": r_proc},
            stats_per_hand={"left": l_stats, "right": r_stats},
        )

        d = np.load(out_path, allow_pickle=True)
        assert bool(d["left_quality_passed"]) is False
        assert bool(d["right_quality_passed"]) is True
        # left_wrist_cam still present (NaN-filled placeholders)
        assert d["left_wrist_cam"].shape == (n, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
