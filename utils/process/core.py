"""
Per-hand tracking-data cleanup — raw-first trim + quality.

Pipeline position: shared utility for both lines:
  - scripts/02_process       (iPhone .r3d line)
  - scripts335/02_process    (Orbbec Gemini 335 .bag line)
The cleanup logic is hardware-agnostic; the caller adapts source-specific
I/O (.r3d frame indexing vs HW timestamp matching) before invoking
process_hand.

Design principle (raw-first):
  02_process does *only* trim + quality decisions. It does NOT modify
  signal values (no fill, no smooth). Customers / OPC's downstream
  scripts (03_build_source / 04_build_<robot> / replay) decide what to
  do with NaN gaps, jitter, and other raw-data quirks. This matches the
  publishing convention of mainstream human-hand datasets (DexYCB /
  HumanPlus / DROID).

Per-hand pipeline:
  [1] trim_leading_trailing_invalid → records trim_slice
        rejects frames with confidence==0 or low IoU(MP_bbox, joint_cloud)
  [2] quality_check                 → detection rate / max gap / duration
                                       gates; sets `quality_passed` flag
  [3] rotation_jump_diagnostic      → log only; quat dot product is
                                       hemisphere-aware and cheap

Quaternion convention: scipy xyzw, hemisphere-continuous (sign-flipped to
match previous frame). 01_track persists this layout; iPhone caller
converts axis-angle → xyzw before calling process_hand here.

What downstream sees:
  Inside trim, NaN frames remain NaN (HaMeR didn't detect that frame):
    - wrist_cam[t] / wrist_quat_cam[t] / joints_cam[t]: all NaN
    - confidence[t]: 0
  03_build_source converts NaN → placeholder + valid=0 + raw confidence
  passthrough; clients mask training frames via valid / confidence.

  Smoothing now lives in `optimize/` — run `python -m optimize ...` over
  `.processed.npz` to produce `.optimized.npz` with One-Euro / interp etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# =============================================================================
# Config
# =============================================================================

@dataclass
class ProcessConfig:
    """Per-hand processing knobs.

    Defaults chosen for 30 Hz HaMeR pipelines (Gemini 335 / Record3D); tune
    per dataset. No signal-modification knobs — those moved out (raw-first
    principle).
    """
    # Quality gates
    min_detection_rate: float = 0.5     # at least 50% frames detected
    max_gap_frames: int = 30            # no >1s consecutive miss at 30fps
    min_duration_s: float = 1.0         # very short hand visibility = drop
    # Trim by bbox consistency: when MediaPipe detector bbox and the
    # projection of HaMeR's 21 joints disagree (low IoU), HaMeR has
    # mis-localized the hand in 3D — typically at frame edges where the
    # hand is leaving the view. Trim those leading/trailing frames so the
    # mp4 / dataset does not end on a "flying skeleton" frame. 0=disabled.
    bbox_iou_trim_threshold: float = 0.3
    # Rotation diagnostic threshold (logged only)
    rot_jump_warn_rad: float = 0.5


# =============================================================================
# Geometry helpers
# =============================================================================

def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two axis-aligned [x1,y1,x2,y2] boxes. NaN-safe (returns 0)."""
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _joint_cloud_bbox(joints_cam: np.ndarray, K: np.ndarray,
                       padding_px: int = 8) -> np.ndarray:
    """Project 21 joints to image space, return enclosing [x1,y1,x2,y2]+pad."""
    if not np.all(np.isfinite(joints_cam)):
        return np.full(4, np.nan)
    z = joints_cam[:, 2]
    if (z <= 0).any():
        return np.full(4, np.nan)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    px = fx * joints_cam[:, 0] / z + cx
    py = fy * joints_cam[:, 1] / z + cy
    return np.array([
        px.min() - padding_px, py.min() - padding_px,
        px.max() + padding_px, py.max() + padding_px,
    ], dtype=np.float64)


# =============================================================================
# Step 1: Trim by validity
# =============================================================================

def trim_leading_trailing_invalid(
    hand: dict, K: np.ndarray | None = None,
    bbox_iou_threshold: float = 0.0,
) -> tuple[slice, dict, dict]:
    """Crop leading/trailing invalid frames.

    Validity per frame:
      (1) confidence > 0  (detector found a hand)
      (2) If K + threshold given: IoU(MP_bbox, projected joint-cloud bbox)
          >= threshold. Catches "flying skeleton" frames where HaMeR
          mis-localized the hand far from the detected bbox.

    Mid-recording invalid frames are NOT trimmed (downstream filters those).

    Returns (trim_slice, trimmed_hand, stats).
    """
    conf = hand["confidence"]
    n = len(conf)
    valid = conf > 0

    n_iou_rejected = 0
    if K is not None and bbox_iou_threshold > 0:
        for i in range(n):
            if not valid[i]:
                continue
            mp = hand["bbox"][i]
            jc = _joint_cloud_bbox(hand["joints_cam"][i], K)
            if _bbox_iou(mp, jc) < bbox_iou_threshold:
                valid[i] = False
                n_iou_rejected += 1

    if not valid.any():
        return slice(0, 0), {k: v[:0] for k, v in hand.items()}, {
            "n_iou_rejected_total": n_iou_rejected,
            "n_trimmed_leading": int(n),
            "n_trimmed_trailing": 0,
        }
    first = int(np.argmax(valid))
    last = int(len(valid) - np.argmax(valid[::-1]))  # exclusive
    sl = slice(first, last)
    stats = {
        "n_iou_rejected_total": n_iou_rejected,
        "n_trimmed_leading": first,
        "n_trimmed_trailing": int(n - last),
    }
    return sl, {k: v[sl] for k, v in hand.items()}, stats


# =============================================================================
# Step 2: Quality check
# =============================================================================

def quality_check(hand: dict, fps: float, cfg: ProcessConfig
                  ) -> tuple[bool, str]:
    """Return (passed, reason). Inputs are post-trim."""
    conf = hand["confidence"]
    n = len(conf)
    if n == 0:
        return False, "no_detections"
    duration_s = n / fps
    if duration_s < cfg.min_duration_s:
        return False, f"too_short ({duration_s:.2f}s < {cfg.min_duration_s}s)"
    detect_rate = (conf > 0).mean()
    if detect_rate < cfg.min_detection_rate:
        return False, (f"low_detection ({detect_rate:.1%} < "
                       f"{cfg.min_detection_rate:.0%})")
    max_gap = _max_consecutive_zeros(conf > 0)
    if max_gap > cfg.max_gap_frames:
        return False, (f"long_gap ({max_gap} frames > "
                       f"{cfg.max_gap_frames})")
    return True, "ok"


def _max_consecutive_zeros(mask: np.ndarray) -> int:
    """Longest run of False in a bool array."""
    if mask.all():
        return 0
    longest = run = 0
    for v in mask:
        if not v:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


# =============================================================================
# Step 3: Rotation jump diagnostic — quat dot product based
# =============================================================================

def rotation_jump_diagnostic(hand: dict, threshold_rad: float) -> dict:
    """Per-frame angle delta between consecutive wrist quaternions.

    Uses quat dot product: angle = 2·acos(|q1·q2|). Hemisphere-aware via
    abs(); cheap (4 mults + acos vs matrix conversion + Frobenius).
    Logging only — no data mutation. quality_check / trim run before this.
    """
    quat = hand["wrist_quat_cam"]
    valid = ~np.isnan(quat[:, 0])
    if valid.sum() < 2:
        return {"n_jumps_above_threshold": 0, "max_delta_rad": 0.0,
                "median_delta_rad": 0.0}
    q_valid = quat[valid]
    deltas = np.empty(len(q_valid) - 1)
    for i in range(len(q_valid) - 1):
        d = abs(float(np.dot(q_valid[i], q_valid[i + 1])))
        d = max(0.0, min(1.0, d))     # clip for acos
        deltas[i] = 2.0 * math.acos(d)
    return {
        "n_jumps_above_threshold": int((deltas > threshold_rad).sum()),
        "max_delta_rad": float(deltas.max()),
        "median_delta_rad": float(np.median(deltas)),
    }


# =============================================================================
# Top-level orchestration
# =============================================================================

def process_hand(hand: dict, *, fps: float, K: np.ndarray | None = None,
                 cfg: ProcessConfig
                 ) -> tuple[dict | None, dict]:
    """Run the per-hand pipeline.

    Returns (processed_hand_or_None, stats):
      - If quality FAILS: returns (None, stats with skip_reason). Caller
        decides whether to drop the hand or write it as quality_passed=False;
        contract here is "this hand is not safe to use as training data".
      - If quality PASSES: returns (trimmed raw hand, stats). NaN frames
        within trim are preserved as-is — no fill, no smooth (raw-first
        principle; downstream decides).
    """
    sl, trimmed, trim_stats = trim_leading_trailing_invalid(
        hand, K=K, bbox_iou_threshold=cfg.bbox_iou_trim_threshold,
    )
    n_after_trim = len(trimmed["confidence"])
    passed, reason = quality_check(trimmed, fps, cfg)
    stats = {
        "trim_slice": (sl.start, sl.stop),
        "n_after_trim": n_after_trim,
        "trim_iou_rejected": trim_stats["n_iou_rejected_total"],
        "n_trimmed_leading": trim_stats["n_trimmed_leading"],
        "n_trimmed_trailing": trim_stats["n_trimmed_trailing"],
        "quality_passed": passed,
        "skip_reason": reason,
    }
    if not passed:
        stats["rotation_diagnostic"] = None
        return None, stats
    stats["rotation_diagnostic"] = rotation_jump_diagnostic(
        trimmed, cfg.rot_jump_warn_rad
    )
    return trimmed, stats
