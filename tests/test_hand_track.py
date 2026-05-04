"""
Tests for scripts/01_hand_track.py (perception layer, cam-frame only).

Covers:
  1. process_episode end-to-end → flat-key npz on disk
  2. Single-hand episode (other hand stays NaN)
  3. No detections → all NaN
  4. bbox + confidence passthrough
  5. T_world_cam + K passthrough
  6. timestamps_us conversion to int64 microseconds
  7. _make_empty_hand_arrays helper

01 is cam-frame only — no world transform / no trim / no quality / no
gripper_width (derivable downstream). World transform / smoothing / IK
all live in 04_build_so101.py per the iPhone-line raw-first refactor.

Run:
    cd code/opc_data_pipeline
    python -m pytest tests/test_hand_track.py -v
"""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# 01_hand_track.py starts with a digit — import via spec.
# Register in sys.modules BEFORE exec_module so @dataclass works (it does
# `sys.modules[cls.__module__]` lookups during class creation).
_spec = importlib.util.spec_from_file_location(
    "iphone_01_hand_track",
    str(_PROJECT_ROOT / "scripts" / "01_hand_track.py"),
)
_hand_track = importlib.util.module_from_spec(_spec)
sys.modules["iphone_01_hand_track"] = _hand_track
_spec.loader.exec_module(_hand_track)

HandTrackConfig = _hand_track.HandTrackConfig
process_episode = _hand_track.process_episode
_make_empty_hand_arrays = _hand_track._make_empty_hand_arrays

from utils.hand_tracker.base import HandDetection, HandTracker


# ============================================================
# Helpers
# ============================================================

class MockTracker(HandTracker):
    """Returns configurable detections for testing process_episode."""

    def __init__(self, detections_per_frame=None):
        self._detections = detections_per_frame or []

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        if self._detections:
            return self._detections.pop(0)
        return []

    def get_backend_name(self) -> str:
        return "mock"


def make_detection(handedness="right", z=0.5, confidence=0.9,
                   wrist_xy=(0.1, -0.05), bbox=(10.0, 20.0, 30.0, 40.0)):
    """Create a synthetic HandDetection (camera frame)."""
    joints = np.zeros((21, 3))
    for i in range(21):
        joints[i] = [i * 0.01, 0.0, z]
    joints[4] = [0.0, 0.0, z]       # thumb tip
    joints[8] = [0.05, 0.0, z]      # index tip (gripper width = 0.05 m)
    return HandDetection(
        handedness=handedness,
        wrist_pos=np.array([wrist_xy[0], wrist_xy[1], z]),
        wrist_rot=np.array([0.0, 0.0, 0.0]),
        joints_3d=joints,
        confidence=confidence,
        bbox=np.array(bbox, dtype=np.float64),
    )


def _identity_poses(n: int) -> list[list[float]]:
    """Per-frame identity T_world_cam as [qx, qy, qz, qw, tx, ty, tz]."""
    return [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for _ in range(n)]


def create_synthetic_r3d(
    tmp_path: Path,
    n_frames: int = 5,
    poses: list[list[float]] | None = None,
    w: int = 32,
    h: int = 48,  # portrait by default (w <= h → no rotation)
    name: str = "test_episode",
) -> Path:
    """Create a minimal .r3d file with synthetic frames + per-frame poses."""
    r3d_path = tmp_path / f"{name}.r3d"

    img = np.zeros((h, w, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode(".jpg", img)

    if poses is None:
        poses = _identity_poses(n_frames)
    assert len(poses) == n_frames, "poses must be one per frame"

    metadata = {
        "w": w,
        "h": h,
        "fps": 30,
        "frameTimestamps": [i / 30.0 for i in range(n_frames)],
        "K": [500.0, 0, 0, 0, 500.0, 0, w / 2.0, h / 2.0, 1],
        "poses": poses,
    }

    with zipfile.ZipFile(r3d_path, "w") as zf:
        zf.writestr("metadata", json.dumps(metadata))
        for i in range(n_frames):
            zf.writestr(f"rgbd/{i}.jpg", img_bytes.tobytes())
    return r3d_path


def _run_and_load(tmp_path, n_frames, detections, *, poses=None):
    """Run process_episode → load resulting .tracking.npz from disk."""
    r3d = create_synthetic_r3d(tmp_path, n_frames=n_frames, poses=poses)
    tracker = MockTracker(detections)
    config = HandTrackConfig(read_depth=False)
    output_root = tmp_path / "01_tracking"
    process_episode(r3d, output_root=output_root, config=config, tracker=tracker)
    npz_path = output_root / "test_episode" / "test_episode.tracking.npz"
    assert npz_path.exists(), f"Expected output at {npz_path}"
    return np.load(npz_path, allow_pickle=True)


# ============================================================
# 1. process_episode end-to-end
# ============================================================

class TestProcessEpisode:

    def test_dual_hand_npz_schema(self, tmp_path):
        """Both hands detected → flat-key npz with full schema."""
        n = 3
        detections = [
            [make_detection("left"), make_detection("right")],
            [make_detection("left"), make_detection("right")],
            [make_detection("left"), make_detection("right")],
        ]
        d = _run_and_load(tmp_path, n, detections)

        # Top-level keys
        assert "timestamps_us" in d.files
        assert d["timestamps_us"].dtype == np.int64
        assert d["timestamps_us"].shape == (n,)
        assert "K" in d.files
        assert d["K"].shape == (3, 3)
        assert "T_world_cam" in d.files
        assert d["T_world_cam"].shape == (n, 4, 4)
        assert "source" in d.files
        assert str(d["source"]) == "mock"
        assert "episode_name" in d.files
        assert str(d["episode_name"]) == "test_episode"
        assert str(d["coord_frame"]) == "camera"

        # Per-hand flat keys (no nested dicts in npz)
        for hand in ("left", "right"):
            assert f"{hand}_wrist_cam" in d.files
            assert d[f"{hand}_wrist_cam"].shape == (n, 3)
            assert f"{hand}_wrist_rot_cam" in d.files
            assert d[f"{hand}_wrist_rot_cam"].shape == (n, 3)
            assert f"{hand}_joints_cam" in d.files
            assert d[f"{hand}_joints_cam"].shape == (n, 21, 3)
            assert f"{hand}_bbox" in d.files
            assert d[f"{hand}_bbox"].shape == (n, 4)
            assert f"{hand}_confidence" in d.files
            assert d[f"{hand}_confidence"].shape == (n,)
            # All detected → no NaN
            assert not np.any(np.isnan(d[f"{hand}_wrist_cam"]))

    def test_meta_json_sidecar(self, tmp_path):
        """Sidecar .tracking.meta.json contains backend + detection stats."""
        r3d = create_synthetic_r3d(tmp_path, n_frames=2)
        tracker = MockTracker([[make_detection("right")] for _ in range(2)])
        config = HandTrackConfig(read_depth=False)
        output_root = tmp_path / "01_tracking"
        process_episode(r3d, output_root=output_root, config=config,
                        tracker=tracker)
        meta_path = output_root / "test_episode" / "test_episode.tracking.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["session_id"] == "test_episode"
        assert meta["n_frames"] == 2
        assert meta["detection"]["right_frames"] == 2
        assert meta["detection"]["left_frames"] == 0


# ============================================================
# 2. Single hand only
# ============================================================

class TestSingleHand:

    def test_right_only(self, tmp_path):
        n = 3
        detections = [[make_detection("right")] for _ in range(n)]
        d = _run_and_load(tmp_path, n, detections)
        assert not np.any(np.isnan(d["right_wrist_cam"]))
        assert np.all(np.isnan(d["left_wrist_cam"]))

    def test_left_only(self, tmp_path):
        n = 3
        detections = [[make_detection("left")] for _ in range(n)]
        d = _run_and_load(tmp_path, n, detections)
        assert not np.any(np.isnan(d["left_wrist_cam"]))
        assert np.all(np.isnan(d["right_wrist_cam"]))


# ============================================================
# 3. No detections
# ============================================================

class TestNoDetection:

    def test_empty_frames(self, tmp_path):
        n = 3
        d = _run_and_load(tmp_path, n, [[] for _ in range(n)])
        assert np.all(np.isnan(d["left_wrist_cam"]))
        assert np.all(np.isnan(d["right_wrist_cam"]))
        assert d["timestamps_us"].shape == (n,)


# ============================================================
# 4. bbox / confidence passthrough
# ============================================================

class TestPassthrough:

    def test_bbox_values(self, tmp_path):
        det = make_detection("right", bbox=(11.0, 22.0, 33.0, 44.0))
        d = _run_and_load(tmp_path, 2, [[det], [det]])
        np.testing.assert_allclose(
            d["right_bbox"][0], [11.0, 22.0, 33.0, 44.0], atol=1e-10,
        )

    def test_confidence_values(self, tmp_path):
        det = make_detection("right", confidence=0.77)
        d = _run_and_load(tmp_path, 2, [[det], [det]])
        np.testing.assert_allclose(d["right_confidence"], [0.77, 0.77])


# ============================================================
# 5. T_world_cam + K passthrough (no world transform in 01)
# ============================================================

class TestCamFrameContract:

    def test_wrist_cam_equals_detector_output(self, tmp_path):
        """Non-trivial T_world_cam does NOT affect wrist_cam — stays in cam."""
        poses = [[0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0]] * 2
        det = make_detection("right", z=0.5, wrist_xy=(0.1, -0.05))
        d = _run_and_load(tmp_path, 2, [[det], [det]], poses=poses)
        np.testing.assert_allclose(
            d["right_wrist_cam"][0], [0.1, -0.05, 0.5], atol=1e-10,
        )

    def test_T_world_cam_passthrough(self, tmp_path):
        """Non-identity T_world_cam preserved per-frame for 02 to consume."""
        poses = [[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]] * 2
        d = _run_and_load(tmp_path, 2, [[] for _ in range(2)], poses=poses)
        np.testing.assert_allclose(
            d["T_world_cam"][0, :3, 3], [1.0, 2.0, 3.0], atol=1e-10,
        )


# ============================================================
# 6. timestamps_us conversion
# ============================================================

class TestTimestamps:

    def test_microsecond_int64(self, tmp_path):
        """Record3D float seconds → int64 microseconds (cross-line consistent)."""
        n = 3
        d = _run_and_load(tmp_path, n, [[] for _ in range(n)])
        # frameTimestamps in fixture is [0, 1/30, 2/30] seconds.
        # → microseconds: [0, 33333, 66666] (rounding to nearest int).
        expected_us = np.array([0, 33333, 66666], dtype=np.int64)
        # Allow ±1 us due to float→int rounding.
        np.testing.assert_allclose(d["timestamps_us"], expected_us, atol=1)
        assert d["timestamps_us"].dtype == np.int64


# ============================================================
# 7. _make_empty_hand_arrays helper
# ============================================================

class TestHelpers:

    def test_make_empty_hand_arrays(self):
        hand = _make_empty_hand_arrays(10)
        assert hand["wrist_cam"].shape == (10, 3)
        assert hand["wrist_rot_cam"].shape == (10, 3)
        assert hand["joints_cam"].shape == (10, 21, 3)
        assert hand["confidence"].shape == (10,)
        assert hand["bbox"].shape == (10, 4)
        assert "gripper_width" not in hand, (
            "gripper_width must NOT be stored (matches HAND_FIELDS / 335)"
        )
        assert np.all(np.isnan(hand["wrist_cam"]))
        assert np.all(hand["confidence"] == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
