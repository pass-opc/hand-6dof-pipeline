"""
Per-hand data contract for the iPhone-line pipeline.

Pipeline position: business-level schema definition for per-frame hand data
in the .tracking.npz / .processed.npz stages. Mirrors scripts335/_schema.py
so both lines produce LeRobot v3 source datasets with identical field
schemas (the 'parallel' part of 'parallel development').

The fields here are the contract between 02_process (producer) and
03_build_source / 04_build_so101 (consumers). Adding/removing a field is a
pipeline business decision and should be done here, not buried in IO code.

Fields:
  wrist_cam       (T, 3)    float64  metric position in cam frame, NaN if absent
  wrist_quat_cam  (T, 4)    float64  scipy xyzw, hemisphere-continuous
  joints_cam      (T,21,3)  float64  21 MANO keypoints (cam frame, LiDAR-corrected)
  bbox            (T, 4)    float64  [x1,y1,x2,y2] pixels (MediaPipe detector)
  confidence      (T,)      float64  detector confidence (0 = no detection)

01_hand_track.py persists wrist orientation as axis-angle (`wrist_rot_cam`,
HaMeR's native output); 02_process.py converts to xyzw quaternion with
hemisphere continuity to match this schema before quality processing.

Notes:
  gripper_width is NOT stored — derivable as ||joints_cam[4] - joints_cam[8]||
  downstream (thumb tip ↔ index tip distance, meters). 04_build_so101
  re-derives it for the SO-101 7D action.
"""

from __future__ import annotations

import numpy as np


HAND_FIELDS: tuple[str, ...] = (
    "wrist_cam",
    "wrist_quat_cam",
    "joints_cam",
    "bbox",
    "confidence",
)


def make_empty_hand_arrays(n_frames: int) -> dict[str, np.ndarray]:
    """NaN-filled per-hand arrays. Layout matches HAND_FIELDS.

    NaN means 'no detection that frame'. confidence defaults to 0 so a
    sum-over-frames > 0 distinguishes 'ever seen' from 'never seen'.
    """
    return {
        "wrist_cam":      np.full((n_frames, 3), np.nan, dtype=np.float64),
        "wrist_quat_cam": np.full((n_frames, 4), np.nan, dtype=np.float64),
        "joints_cam":     np.full((n_frames, 21, 3), np.nan, dtype=np.float64),
        "bbox":           np.full((n_frames, 4), np.nan, dtype=np.float64),
        "confidence":     np.zeros(n_frames, dtype=np.float64),
    }
