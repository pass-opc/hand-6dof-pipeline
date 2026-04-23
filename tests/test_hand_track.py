"""
Tests for scripts/01_hand_track.py (perception layer, cam-frame only).

Covers:
  1. process_episode — dual-hand output, cam-frame contract
  2. Single hand — other hand stays NaN
  3. Empty detections
  4. Gripper width from joints
  5. bbox passthrough
  6. Output contract — coord_frame='camera', T_world_cam + K passthrough,
     joints_cam shape
  7. Helpers

World-frame transform / filter / centering now live in 02_process.py and
are tested by tests/test_process.py.

Run:
    cd hand-6dof-pipeline
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 01_hand_track.py starts with a digit — import via spec
_spec = importlib.util.spec_from_file_location(
    "hand_track",
    str(Path(__file__).resolve().parent.parent / "scripts" / "01_hand_track.py"),
)
_hand_track = importlib.util.module_from_spec(_spec)
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
    joints[8] = [0.05, 0.0, z]      # index tip (gripper width = 0.05)
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
) -> Path:
    """Create a minimal .r3d file with synthetic frames + per-frame poses."""
    r3d_path = tmp_path / "test_episode.r3d"

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


# ============================================================
# 1. Dual-hand output + cam-frame contract
# ============================================================

class TestProcessEpisode:

    def test_both_hands_detected(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=3)
        detections = [
            [make_detection("left"), make_detection("right")],
            [make_detection("left"), make_detection("right")],
            [make_detection("left"), make_detection("right")],
        ]
        tracker = MockTracker(detections)
        config = HandTrackConfig(read_depth=False)

        result = process_episode(r3d, config, tracker)

        assert "timestamps" in result
        assert "left_hand" in result
        assert "right_hand" in result
        assert result["source"] == "mock"
        assert len(result["timestamps"]) == 3

        for hand in ("left_hand", "right_hand"):
            h = result[hand]
            assert not np.any(np.isnan(h["wrist_cam"]))
            assert h["wrist_cam"].shape == (3, 3)
            assert h["wrist_rot_cam"].shape == (3, 3)
            assert h["gripper_width"].shape == (3,)
            assert h["joints_cam"].shape == (3, 21, 3)
            assert h["confidence"].shape == (3,)
            assert h["bbox"].shape == (3, 4)

    def test_output_contract(self, tmp_path):
        """Cam-frame contract: coord_frame='camera', T_world_cam + K passthrough."""
        r3d = create_synthetic_r3d(tmp_path, n_frames=2)
        tracker = MockTracker([[make_detection("right")], [make_detection("right")]])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        assert result["coord_frame"] == "camera"
        assert "T_world_cam" in result
        assert result["T_world_cam"].shape == (2, 4, 4)
        assert "K" in result
        assert result["K"].shape == (3, 3)
        assert result["episode_name"] == "test_episode"

        for hand_key in ("left_hand", "right_hand"):
            hand = result[hand_key]
            for k in ("wrist_cam", "wrist_rot_cam", "joints_cam",
                      "bbox", "gripper_width", "confidence"):
                assert k in hand, f"missing {k} in {hand_key}"


# ============================================================
# 2. Single hand
# ============================================================

class TestSingleHand:

    def test_right_only(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=3)
        detections = [[make_detection("right")] for _ in range(3)]
        tracker = MockTracker(detections)
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        assert not np.any(np.isnan(result["right_hand"]["wrist_cam"]))
        assert np.all(np.isnan(result["left_hand"]["wrist_cam"]))

    def test_left_only(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=3)
        detections = [[make_detection("left")] for _ in range(3)]
        tracker = MockTracker(detections)
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        assert not np.any(np.isnan(result["left_hand"]["wrist_cam"]))
        assert np.all(np.isnan(result["right_hand"]["wrist_cam"]))


# ============================================================
# 3. No detections
# ============================================================

class TestNoDetection:

    def test_empty_frames(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=3)
        tracker = MockTracker([[] for _ in range(3)])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        assert np.all(np.isnan(result["left_hand"]["wrist_cam"]))
        assert np.all(np.isnan(result["right_hand"]["wrist_cam"]))
        assert len(result["timestamps"]) == 3


# ============================================================
# 4. Gripper width
# ============================================================

class TestGripperWidth:

    def test_gripper_width_stored(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=2)
        detections = [[make_detection("right")] for _ in range(2)]
        tracker = MockTracker(detections)
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        np.testing.assert_allclose(
            result["right_hand"]["gripper_width"],
            [0.05, 0.05], atol=1e-10,
        )


# ============================================================
# 5. bbox passthrough
# ============================================================

class TestBbox:

    def test_bbox_values(self, tmp_path):
        """bbox provided on detection must land in output verbatim."""
        r3d = create_synthetic_r3d(tmp_path, n_frames=2)
        det = make_detection("right", bbox=(11.0, 22.0, 33.0, 44.0))
        tracker = MockTracker([[det], [det]])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        np.testing.assert_allclose(
            result["right_hand"]["bbox"][0], [11.0, 22.0, 33.0, 44.0], atol=1e-10,
        )


# ============================================================
# 6. Cam-frame passthrough (no world transform in 01)
# ============================================================

class TestCamFramePassthrough:
    """01 is cam-frame only. No matter what T_world_cam is, wrist_cam and
    joints_cam stay in camera frame and equal the detector output."""

    def test_wrist_cam_equals_detector_output(self, tmp_path):
        """Non-trivial T_world_cam does NOT affect wrist_cam (stays in cam)."""
        # Translation pose: t_wc = (5, 6, 7). Would shift wrist if applied.
        poses = [[0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0]] * 2
        r3d = create_synthetic_r3d(tmp_path, n_frames=2, poses=poses)
        det = make_detection("right", z=0.5, wrist_xy=(0.1, -0.05))
        tracker = MockTracker([[det], [det]])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        # wrist_cam must equal the detector output, not pose-translated value
        np.testing.assert_allclose(
            result["right_hand"]["wrist_cam"][0],
            [0.1, -0.05, 0.5], atol=1e-10,
        )

    def test_joints_cam_equals_detector_output(self, tmp_path):
        poses = [[0.0, 0.0, 0.0, 1.0, 5.0, 6.0, 7.0]] * 2
        r3d = create_synthetic_r3d(tmp_path, n_frames=2, poses=poses)
        det = make_detection("right", z=0.5)
        tracker = MockTracker([[det], [det]])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        # joints[0] from make_detection: [0, 0, z]
        np.testing.assert_allclose(
            result["right_hand"]["joints_cam"][0, 0],
            [0.0, 0.0, 0.5], atol=1e-10,
        )

    def test_T_world_cam_passthrough(self, tmp_path):
        """Non-identity T_world_cam must be preserved for 02 to consume."""
        poses = [[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]] * 2
        r3d = create_synthetic_r3d(tmp_path, n_frames=2, poses=poses)
        tracker = MockTracker([[make_detection("right")] for _ in range(2)])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)

        # Identity R (portrait), translation (1, 2, 3) from pose
        np.testing.assert_allclose(
            result["T_world_cam"][0, :3, 3], [1.0, 2.0, 3.0], atol=1e-10,
        )


# ============================================================
# 7. Helpers
# ============================================================

class TestHelpers:

    def test_make_empty_hand_arrays(self):
        hand = _make_empty_hand_arrays(10)
        assert hand["wrist_cam"].shape == (10, 3)
        assert hand["wrist_rot_cam"].shape == (10, 3)
        assert hand["gripper_width"].shape == (10,)
        assert hand["joints_cam"].shape == (10, 21, 3)
        assert hand["confidence"].shape == (10,)
        assert hand["bbox"].shape == (10, 4)
        assert np.all(np.isnan(hand["wrist_cam"]))
        assert np.all(hand["confidence"] == 0.0)


# ============================================================
# 8. Metadata
# ============================================================

class TestMetadata:

    def test_episode_name(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=2)
        tracker = MockTracker([[] for _ in range(2)])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)
        assert result["episode_name"] == "test_episode"

    def test_source_from_backend(self, tmp_path):
        r3d = create_synthetic_r3d(tmp_path, n_frames=2)
        tracker = MockTracker([[] for _ in range(2)])
        config = HandTrackConfig(read_depth=False)
        result = process_episode(r3d, config, tracker)
        assert result["source"] == "mock"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
