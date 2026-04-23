"""Tests for spatial hand identity tracker (utils/spatial_tracker.py)."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.hand_tracker.base import HandDetection
from utils.spatial_tracker import SpatialHandTracker, _bbox_center


def _make_det(
    handedness: str = "right",
    bbox_center: tuple[float, float] = (320.0, 240.0),
    confidence: float = 0.9,
    bbox_size: float = 80.0,
) -> HandDetection:
    """Helper: create a HandDetection with a bbox centered at given position."""
    cx, cy = bbox_center
    half = bbox_size / 2.0
    bbox = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)
    return HandDetection(
        handedness=handedness,
        wrist_pos=np.zeros(3),
        wrist_rot=np.zeros(3),
        joints_3d=np.zeros((21, 3)),
        confidence=confidence,
        bbox=bbox,
    )


class TestBboxCenter:
    def test_basic(self):
        bbox = np.array([100.0, 200.0, 300.0, 400.0])
        c = _bbox_center(bbox)
        np.testing.assert_allclose(c, [200.0, 300.0])


class TestSingleHandStable:
    """One hand detected consistently -> same handedness maintained."""

    def test_stable_right_hand(self):
        tracker = SpatialHandTracker()
        positions = [(300, 240), (305, 242), (310, 238), (308, 241)]

        for pos in positions:
            dets = [_make_det("right", pos)]
            result = tracker.update(dets)
            assert len(result) == 1
            assert result[0].handedness == "right"

    def test_stable_left_hand(self):
        tracker = SpatialHandTracker()
        for _ in range(5):
            result = tracker.update([_make_det("left", (100, 200))])
            assert len(result) == 1
            assert result[0].handedness == "left"


class TestHandednessFlipResilience:
    """Right hand detected, then MediaPipe flips to 'left' at same position.
    Spatial tracker should keep it as 'right'."""

    def test_flip_rejected_by_spatial_continuity(self):
        tracker = SpatialHandTracker()

        # Establish track: 5 frames of "right" at position (300, 240)
        for _ in range(5):
            tracker.update([_make_det("right", (300, 240), confidence=0.9)])

        # Now MediaPipe says "left" at same position — spatial tracker should
        # keep calling it "right" because cumulative vote is strongly right
        result = tracker.update([_make_det("left", (302, 241), confidence=0.7)])
        assert len(result) == 1
        assert result[0].handedness == "right"

    def test_sustained_flip_eventually_changes(self):
        """If MediaPipe consistently says 'left' for many frames, the
        cumulative vote should eventually flip (genuine re-identification)."""
        tracker = SpatialHandTracker()

        # Establish as right: 3 frames, confidence 0.5
        for _ in range(3):
            tracker.update([_make_det("right", (300, 240), confidence=0.5)])

        # Now consistently report left with high confidence — should eventually flip
        for _ in range(20):
            result = tracker.update([_make_det("left", (300, 240), confidence=0.9)])

        assert result[0].handedness == "left"


class TestDualHand:
    """Two hands tracked simultaneously, correct identity maintained."""

    def test_two_hands_tracked(self):
        tracker = SpatialHandTracker()

        # Frame 1: establish both hands
        dets = [
            _make_det("left", (100, 240)),
            _make_det("right", (500, 240)),
        ]
        result = tracker.update(dets)
        assert len(result) == 2
        labels = {r.handedness for r in result}
        assert labels == {"left", "right"}

    def test_two_hands_labels_swapped_by_mediapipe(self):
        """Both hands present, MediaPipe swaps labels — spatial tracker corrects."""
        tracker = SpatialHandTracker()

        # Establish tracks
        for _ in range(5):
            tracker.update([
                _make_det("left", (100, 240), confidence=0.9),
                _make_det("right", (500, 240), confidence=0.9),
            ])

        # MediaPipe swaps labels (left hand reported as right, vice versa)
        result = tracker.update([
            _make_det("right", (102, 241), confidence=0.7),
            _make_det("left", (498, 239), confidence=0.7),
        ])

        # Spatial matching should assign by position, not label
        for det in result:
            center = _bbox_center(det.bbox)
            if center[0] < 300:
                assert det.handedness == "left", "Left-side detection should stay 'left'"
            else:
                assert det.handedness == "right", "Right-side detection should stay 'right'"


class TestTrackInitialization:
    """First frame establishes tracks from MediaPipe labels."""

    def test_first_frame_uses_mediapipe_labels(self):
        tracker = SpatialHandTracker()
        assert len(tracker.tracks) == 0

        result = tracker.update([
            _make_det("left", (100, 200)),
            _make_det("right", (400, 200)),
        ])

        assert len(tracker.tracks) == 2
        assert "left" in tracker.tracks
        assert "right" in tracker.tracks
        assert len(result) == 2

    def test_empty_frame(self):
        tracker = SpatialHandTracker()
        result = tracker.update([])
        assert result == []
        assert len(tracker.tracks) == 0


class TestTrackExpiry:
    """Hand disappears for many frames -> track expires -> re-detection starts fresh."""

    def test_track_expires_after_missing_frames(self):
        tracker = SpatialHandTracker(max_frames_missing=5)

        # Establish a track
        tracker.update([_make_det("right", (300, 240))])
        assert "right" in tracker.tracks

        # Hand disappears for 6 frames (> max_frames_missing)
        for _ in range(6):
            tracker.update([])

        # Track should be expired
        assert len(tracker.tracks) == 0

    def test_redetection_after_expiry_starts_fresh(self):
        tracker = SpatialHandTracker(max_frames_missing=3)

        # Establish right hand, build up strong right vote
        for _ in range(10):
            tracker.update([_make_det("right", (300, 240), confidence=0.9)])

        # Disappear long enough to expire
        for _ in range(5):
            tracker.update([])
        assert len(tracker.tracks) == 0

        # Re-detect at same position but as "left" — should be left now
        # (fresh track, no history)
        result = tracker.update([_make_det("left", (300, 240))])
        assert result[0].handedness == "left"


class TestLargeJumpRejection:
    """Detection far from any track -> new track, don't corrupt existing."""

    def test_far_detection_creates_new_track(self):
        tracker = SpatialHandTracker(max_distance_px=150)

        # Establish right hand at (300, 240)
        tracker.update([_make_det("right", (300, 240))])

        # New detection at (600, 240) — 300px away, beyond max_distance_px
        # Should not match existing track
        result = tracker.update([
            _make_det("right", (300, 240)),  # continues existing track
            _make_det("left", (600, 240)),   # new track
        ])
        assert len(result) == 2
        assert "left" in tracker.tracks
        assert "right" in tracker.tracks

    def test_jump_does_not_corrupt_existing_track(self):
        tracker = SpatialHandTracker(max_distance_px=100)

        # Establish right hand
        for _ in range(5):
            tracker.update([_make_det("right", (300, 240))])

        original_center = tracker.tracks["right"].last_bbox_center.copy()

        # Detection jumps far away — should NOT update existing track position
        result = tracker.update([_make_det("right", (600, 240))])

        # The far detection can't match existing track, so it tries to create
        # a new one. Since "right" is taken, it gets assigned "left".
        # Original track's position should be unchanged (it was not matched).
        # Note: track was aged but not updated with new position.
        np.testing.assert_allclose(
            tracker.tracks["right"].last_bbox_center,
            original_center,
        )


class TestRekey:
    """Track re-keying when voted handedness diverges from dict key."""

    def test_rekey_after_vote_flip(self):
        """Track starts as 'left' (MediaPipe label), votes flip to 'right'.
        After re-keying, new 'left' detection should create a fresh track."""
        tracker = SpatialHandTracker(max_frames_missing=5)

        # MediaPipe says "left" but it's really a right hand — first frame
        tracker.update([_make_det("left", (300, 240), confidence=0.5)])
        assert "left" in tracker.tracks

        # Many frames MediaPipe says "right" at same position → vote flips
        for _ in range(10):
            tracker.update([_make_det("right", (302, 242), confidence=0.9)])

        # Track should have been re-keyed to "right"
        assert "right" in tracker.tracks, "Track should be re-keyed to 'right'"
        assert "left" not in tracker.tracks, "Old 'left' key should be gone"

    def test_rekey_frees_slot_for_real_hand(self):
        """After re-keying frees a slot, a genuinely different hand can take it."""
        tracker = SpatialHandTracker(max_frames_missing=5)

        # Start with "left" label at position (300, 240), votes to "right"
        tracker.update([_make_det("left", (300, 240), confidence=0.3)])
        for _ in range(5):
            tracker.update([_make_det("right", (302, 242), confidence=0.9)])

        # "left" slot is now free. A real left hand appears far away.
        result = tracker.update([
            _make_det("right", (304, 244), confidence=0.9),  # existing track
            _make_det("left", (100, 240), confidence=0.9),   # new hand
        ])
        assert len(result) == 2
        assert "left" in tracker.tracks
        assert "right" in tracker.tracks


class TestNoBbox:
    """Detections without bbox are passed through unchanged."""

    def test_no_bbox_passthrough(self):
        tracker = SpatialHandTracker()
        det = HandDetection(
            handedness="right",
            wrist_pos=np.zeros(3),
            wrist_rot=np.zeros(3),
            joints_3d=np.zeros((21, 3)),
            confidence=0.9,
            bbox=None,
        )
        result = tracker.update([det])
        assert len(result) == 1
        assert result[0].handedness == "right"
