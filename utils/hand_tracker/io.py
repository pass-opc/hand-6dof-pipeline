"""
Per-bag tracking IO — npz dataset + JSON sidecar.

Pipeline position: shared utility for scripts335/01_track and 02_process.

Schema-agnostic: save_track_npz iterates whatever per-hand keys the caller
provides (the schema itself lives in scripts335/_schema.HAND_FIELDS, where
it can be evolved without touching this IO layer).

Why npz over pickle:
  - inspectable: np.load(path); list(npz.keys()) shows every field
  - cross-language: numpy .npy is portable
  - lossless + compressed via savez_compressed
  - one file per session; multi-episode batching is 03_build's job
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def save_track_npz(
    path: Path,
    *,
    timestamps_us: np.ndarray,
    K: np.ndarray,
    hands: dict[str, dict[str, np.ndarray] | None],
    extras: dict[str, np.ndarray] | None = None,
) -> None:
    """Write per-frame arrays for each hand to a compressed npz.

    hands maps hand_name → per-hand dict (any keys/shapes) OR None to skip.
    Skipped (None) hands are simply omitted from the npz; downstream
    consumers can introspect via npz.files to see which hands are present.

    Each per-hand key f produces an npz field named {hand_name}_{f}.

    extras (optional) is a dict of top-level array fields to add to the
    npz alongside timestamps_us / K — e.g. T_world_cam / T_world_cam_valid
    when the producer pipeline includes a camera-pose source. Keys must
    not collide with existing top-level names.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamps_us": timestamps_us.astype(np.int64),
        "K": K.astype(np.float64),
    }
    if extras:
        for key, arr in extras.items():
            if key in payload:
                raise ValueError(
                    f"extras key {key!r} collides with reserved top-level "
                    f"field; pick a different name"
                )
            payload[key] = arr
    for hand_name, arrs in hands.items():
        if arrs is None:
            continue
        for f, arr in arrs.items():
            payload[f"{hand_name}_{f}"] = arr
    np.savez_compressed(path, **payload)


def write_track_meta(
    path: Path,
    *,
    session_id: str,
    bag_path: Path,
    backend: str,
    backend_version: str | None,
    duration_s: float,
    n_frames: int,
    n_left_detected: int,
    n_right_detected: int,
    depth_correction_stats: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Sidecar JSON next to the .npz with backend stamps + detection summary."""
    meta = {
        "session_id": session_id,
        "timestamp_iso": datetime.now().isoformat(),
        "source_bag": str(bag_path),
        "backend": backend,
        "backend_version": backend_version,
        "duration_s": round(duration_s, 3),
        "n_frames": n_frames,
        "detection": {
            "left_frames": int(n_left_detected),
            "right_frames": int(n_right_detected),
            "left_rate": round(n_left_detected / max(1, n_frames), 4),
            "right_rate": round(n_right_detected / max(1, n_frames), 4),
        },
        "depth_correction": depth_correction_stats,
        # `world_frame` describes the source of T_world_cam in the npz.
        # Default when no cam_pose source ran. Override via `extra` to e.g.
        # {"backend": "aruco", "real_detection_rate": 0.97, ...}.
        "world_frame": "not_provided",
        "wrist_rotation_format": "quaternion_xyzw_scipy_hemisphere_continuous",
        "wrist_rotation_note": ("HaMeR temporal noise can produce real "
                                 "matrix-rotation jumps of 1-3 rad/frame "
                                 "(distinct from axis-angle wraps which the "
                                 "quaternion storage already eliminates). "
                                 "Apply downstream smoothing if needed."),
        "schema_version": 2,
    }
    if extra:
        meta.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8")
