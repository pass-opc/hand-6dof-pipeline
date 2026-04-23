"""
Spatial hand identity tracker using position continuity.

Pipeline position: sits between HandTracker.detect() and per-hand data storage
                   in scripts/01_hand_track.py.

Input:  list[HandDetection] per frame (with bbox from detector)
Output: list[HandDetection] with corrected handedness labels

Problem solved: MediaPipe's handedness classifier flips L<->R during object
grasping (finger occlusion). This module overrides the label using spatial
continuity — a hand that was "right" last frame at position (x,y) is still
"right" this frame if it's nearby, regardless of what MediaPipe says.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.hand_tracker.base import HandDetection


@dataclass
class TrackState:
    """Persistent state for one tracked hand."""
    last_bbox_center: np.ndarray  # (2,) pixel coords (cx, cy)
    handedness: str               # current voted identity: "left" or "right"
    handedness_score: float       # cumulative confidence-weighted vote; >0 = right, <0 = left
    frames_since_seen: int = 0


def _bbox_center(bbox: np.ndarray) -> np.ndarray:
    """Compute center of [x1, y1, x2, y2] bounding box."""
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


class SpatialHandTracker:
    """Track hand identity by spatial position continuity, not handedness label.

    Maintains up to two tracks (one per hand). Each frame, incoming detections
    are matched to existing tracks via Hungarian algorithm on bbox center
    distance. Handedness is decided by cumulative confidence-weighted voting,
    not the per-frame MediaPipe label.

    Args:
        max_distance_px: Maximum bbox center distance (pixels) to accept a
                         match. Beyond this, the detection starts a new track.
        max_frames_missing: Expire a track after this many consecutive frames
                            without a matching detection.
    """

    def __init__(
        self,
        max_distance_px: float = 150.0,
        max_frames_missing: int = 30,
    ):
        self.tracks: dict[str, TrackState] = {}  # keyed by handedness label
        self.max_distance_px = max_distance_px
        self.max_frames_missing = max_frames_missing

    def update(self, detections: list[HandDetection]) -> list[HandDetection]:
        """Match detections to tracks, return detections with corrected handedness.

        Algorithm:
          1. Compute bbox center for each detection
          2. If tracks exist: Hungarian matching by Euclidean distance
             between detection centers and last known track positions.
             Only accept matches within max_distance_px.
          3. Unmatched detections: initialize new track using MediaPipe's
             handedness as initial label
          4. Each track updates cumulative confidence-weighted handedness vote
          5. Return detections with handedness set to track's voted identity
        """
        if not detections:
            self._age_all_tracks()
            self._expire_old_tracks()
            return []

        # Filter out detections without bbox (shouldn't happen in normal pipeline)
        valid_dets = [d for d in detections if d.bbox is not None]
        if not valid_dets:
            self._age_all_tracks()
            self._expire_old_tracks()
            return detections

        det_centers = np.array([_bbox_center(d.bbox) for d in valid_dets])

        if self.tracks:
            result = self._match_to_tracks(valid_dets, det_centers)
        else:
            # First frame: initialize tracks from MediaPipe labels
            result = self._init_tracks(valid_dets, det_centers)

        self._rekey_tracks()
        self._age_all_tracks()
        self._expire_old_tracks()

        return result

    def _match_to_tracks(
        self,
        detections: list[HandDetection],
        det_centers: np.ndarray,
    ) -> list[HandDetection]:
        """Match detections to existing tracks using Hungarian algorithm."""
        track_keys = list(self.tracks.keys())
        track_centers = np.array([self.tracks[k].last_bbox_center for k in track_keys])

        # Cost matrix: Euclidean distance between each detection and each track
        # shape: (n_detections, n_tracks)
        n_det = len(detections)
        n_trk = len(track_keys)
        cost = np.zeros((n_det, n_trk))
        for i in range(n_det):
            for j in range(n_trk):
                cost[i, j] = np.linalg.norm(det_centers[i] - track_centers[j])

        # Hungarian matching (minimizes total distance)
        det_idx, trk_idx = linear_sum_assignment(cost)

        matched_det = set()
        matched_trk = set()
        results = []

        for di, ti in zip(det_idx, trk_idx):
            if cost[di, ti] <= self.max_distance_px:
                matched_det.add(di)
                matched_trk.add(ti)

                track_key = track_keys[ti]
                track = self.tracks[track_key]

                # Update track position and handedness vote
                track.last_bbox_center = det_centers[di]
                track.frames_since_seen = 0

                det = detections[di]
                # Vote: positive for right, negative for left
                vote = det.confidence * (1.0 if det.handedness == "right" else -1.0)
                track.handedness_score += vote
                track.handedness = "right" if track.handedness_score > 0 else "left"

                results.append(self._corrected_detection(det, track.handedness))

        # Unmatched detections: start new tracks (if slot available)
        for i in range(n_det):
            if i not in matched_det:
                new_det = self._try_init_track(detections[i], det_centers[i])
                results.append(new_det)

        return results

    def _init_tracks(
        self,
        detections: list[HandDetection],
        det_centers: np.ndarray,
    ) -> list[HandDetection]:
        """Initialize tracks from first-frame detections using MediaPipe labels."""
        results = []
        for det, center in zip(detections, det_centers):
            new_det = self._try_init_track(det, center)
            results.append(new_det)
        return results

    def _try_init_track(
        self,
        det: HandDetection,
        center: np.ndarray,
    ) -> HandDetection:
        """Create a new track for this detection if a slot is available.

        Uses MediaPipe's label. If that label is already taken by another
        track, assigns the opposite label.
        """
        label = det.handedness
        initial_score = det.confidence * (1.0 if label == "right" else -1.0)

        # If this label is already taken, try the other
        if label in self.tracks:
            opposite = "left" if label == "right" else "right"
            if opposite not in self.tracks:
                label = opposite
                initial_score = -initial_score  # flip score to match new label
            else:
                # Both slots taken — can't create new track, return as-is
                return det

        self.tracks[label] = TrackState(
            last_bbox_center=center.copy(),
            handedness=label,
            handedness_score=initial_score,
        )
        return self._corrected_detection(det, label)

    def _rekey_tracks(self) -> None:
        """Re-key tracks when voted handedness diverges from dict key.

        Without re-keying, a track initialized as "left" (MediaPipe label) but
        voted to "right" stays under the "left" key. When it expires and a new
        detection arrives with label "left", the new track takes the "left" slot
        — but it's the same physical hand re-detected, so it should be "right".
        Re-keying prevents this by keeping keys aligned with voted identity.
        """
        rekey = [(k, t.handedness) for k, t in self.tracks.items() if k != t.handedness]
        for old_key, new_key in rekey:
            if new_key not in self.tracks:
                self.tracks[new_key] = self.tracks.pop(old_key)
            else:
                # Conflict: two tracks claim same handedness. Keep the one
                # with stronger conviction (higher |score|).
                existing = self.tracks[new_key]
                challenger = self.tracks[old_key]
                if abs(challenger.handedness_score) > abs(existing.handedness_score):
                    self.tracks[new_key] = challenger
                del self.tracks[old_key]

    def _age_all_tracks(self) -> None:
        """Increment frames_since_seen for all tracks.

        Called once per frame. Tracks that were matched this frame already had
        their counter reset to 0 before this call.
        """
        for track in self.tracks.values():
            track.frames_since_seen += 1

    def _expire_old_tracks(self) -> None:
        """Remove tracks not seen for too long."""
        expired = [
            k for k, t in self.tracks.items()
            if t.frames_since_seen > self.max_frames_missing
        ]
        for k in expired:
            del self.tracks[k]

    @staticmethod
    def _corrected_detection(det: HandDetection, handedness: str) -> HandDetection:
        """Return a copy of the detection with corrected handedness."""
        return HandDetection(
            handedness=handedness,
            wrist_pos=det.wrist_pos,
            wrist_rot=det.wrist_rot,
            joints_3d=det.joints_3d,
            confidence=det.confidence,
            bbox=det.bbox,
        )
