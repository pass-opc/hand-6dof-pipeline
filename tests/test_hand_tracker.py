"""
Tests for utils/hand_tracker/ (base, factory).

Covers:
  1. HandDetection — dataclass validation, gripper_width property
  2. HandTracker ABC — MockTracker implementation
  3. Factory — create_tracker with unknown backend, wilor not implemented

All tests are synthetic (no HaMeR model or hardware needed).

Run:
    cd hand-6dof-pipeline
    python -m pytest tests/test_hand_tracker.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hand_tracker.base import HandBox, HandDetectorBase, HandDetection, HandTracker
from utils.hand_tracker.factory import create_detector, create_tracker


# ============================================================
# Helpers
# ============================================================

def make_joints_3d(thumb_tip=None, index_tip=None):
    """Create synthetic MANO joints (21, 3) with controllable thumb/index tips."""
    joints = np.zeros((21, 3))
    # Spread joints along x for variety
    for i in range(21):
        joints[i] = [i * 0.01, 0.0, 0.5]
    if thumb_tip is not None:
        joints[4] = thumb_tip
    if index_tip is not None:
        joints[8] = index_tip
    return joints


def make_hand_detection(handedness="right", thumb_tip=None, index_tip=None):
    """Create a valid HandDetection with optional thumb/index override."""
    joints = make_joints_3d(thumb_tip, index_tip)
    return HandDetection(
        handedness=handedness,
        wrist_pos=np.array([0.1, -0.05, 0.5]),
        wrist_rot=np.array([0.0, 0.0, 0.0]),
        joints_3d=joints,
        confidence=0.95,
    )


class MockTracker(HandTracker):
    """Mock tracker that returns configurable detections for testing."""

    def __init__(self, detections=None):
        self._detections = detections or []

    def detect(self, rgb: np.ndarray) -> list[HandDetection]:
        return self._detections

    def get_backend_name(self) -> str:
        return "mock"


# ============================================================
# 1. HandDetection
# ============================================================

class TestHandDetection:

    def test_valid_construction(self):
        """HandDetection with valid inputs should construct without error."""
        det = make_hand_detection()
        assert det.handedness == "right"
        assert det.confidence == 0.95
        np.testing.assert_array_equal(det.wrist_pos, [0.1, -0.05, 0.5])

    def test_gripper_width_open(self):
        """Thumb and index far apart → large gripper_width."""
        det = make_hand_detection(
            thumb_tip=np.array([0.0, 0.0, 0.5]),
            index_tip=np.array([0.1, 0.0, 0.5]),
        )
        assert abs(det.gripper_width - 0.1) < 1e-10

    def test_gripper_width_closed(self):
        """Thumb and index at same position → gripper_width ≈ 0."""
        pos = np.array([0.05, 0.0, 0.5])
        det = make_hand_detection(thumb_tip=pos, index_tip=pos)
        assert det.gripper_width < 1e-10

    def test_gripper_width_3d(self):
        """Gripper width uses 3D Euclidean distance, not just one axis."""
        det = make_hand_detection(
            thumb_tip=np.array([0.0, 0.0, 0.0]),
            index_tip=np.array([0.03, 0.04, 0.0]),
        )
        # 3-4-5 triangle → distance = 0.05
        assert abs(det.gripper_width - 0.05) < 1e-10

    def test_invalid_handedness(self):
        """Invalid handedness should raise AssertionError."""
        with pytest.raises(AssertionError):
            HandDetection(
                handedness="center",
                wrist_pos=np.zeros(3),
                wrist_rot=np.zeros(3),
                joints_3d=np.zeros((21, 3)),
                confidence=0.5,
            )

    def test_invalid_wrist_pos_shape(self):
        """Wrong wrist_pos shape should raise AssertionError."""
        with pytest.raises(AssertionError):
            HandDetection(
                handedness="left",
                wrist_pos=np.zeros(4),  # wrong
                wrist_rot=np.zeros(3),
                joints_3d=np.zeros((21, 3)),
                confidence=0.5,
            )

    def test_invalid_joints_shape(self):
        """Wrong joints_3d shape should raise AssertionError."""
        with pytest.raises(AssertionError):
            HandDetection(
                handedness="left",
                wrist_pos=np.zeros(3),
                wrist_rot=np.zeros(3),
                joints_3d=np.zeros((20, 3)),  # wrong: 20 not 21
                confidence=0.5,
            )

    def test_left_hand(self):
        """Left hand detection should work identically."""
        det = make_hand_detection(handedness="left")
        assert det.handedness == "left"


# ============================================================
# 2. HandTracker ABC + MockTracker
# ============================================================

class TestMockTracker:

    def test_empty_detection(self):
        """MockTracker with no detections returns empty list."""
        tracker = MockTracker()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.detect(rgb)
        assert result == []

    def test_single_hand(self):
        """MockTracker returns configured single hand."""
        det = make_hand_detection(handedness="right")
        tracker = MockTracker(detections=[det])
        result = tracker.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result) == 1
        assert result[0].handedness == "right"

    def test_both_hands(self):
        """MockTracker returns both hands."""
        left = make_hand_detection(handedness="left")
        right = make_hand_detection(handedness="right")
        tracker = MockTracker(detections=[left, right])
        result = tracker.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result) == 2
        hands = {d.handedness for d in result}
        assert hands == {"left", "right"}

    def test_backend_name(self):
        """MockTracker should report 'mock' as backend name."""
        tracker = MockTracker()
        assert tracker.get_backend_name() == "mock"


# ============================================================
# 3. Factory
# ============================================================

class TestHandBox:

    def test_valid_construction(self):
        """HandBox with valid inputs should construct without error."""
        hb = HandBox(bbox=np.array([10, 20, 100, 200], dtype=np.float32),
                     is_right=True, confidence=0.9)
        assert hb.is_right is True
        assert hb.confidence == 0.9

    def test_invalid_bbox_shape(self):
        """Wrong bbox shape should raise AssertionError."""
        with pytest.raises(AssertionError):
            HandBox(bbox=np.zeros(5), is_right=True, confidence=0.5)


class TestFactory:

    def test_unknown_backend_raises(self):
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown hand tracker backend"):
            create_tracker("nonexistent")

    def test_wilor_not_implemented(self):
        """WiLoR backend should raise ImportError (not yet implemented)."""
        with pytest.raises(ImportError, match="WiLoR backend not yet implemented"):
            create_tracker("wilor")

    def test_unknown_detector_raises(self):
        """Unknown detector should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown hand detector"):
            create_detector("nonexistent")

    def test_vitpose_not_implemented(self):
        """ViTPose detector should raise ImportError (not yet implemented)."""
        with pytest.raises(ImportError, match="ViTPose detector not yet implemented"):
            create_detector("vitpose")

    def test_hamer_creates_tracker(self):
        """HaMeR backend should construct without loading model."""
        tracker = create_tracker("hamer")
        assert tracker.get_backend_name() == "hamer"
        # Model not loaded until detect() is called (lazy loading)
        assert tracker._model is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
