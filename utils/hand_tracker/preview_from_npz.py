"""
Schema-agnostic preview-mp4 renderer for any stage that holds MANO-21
keypoints in cam frame.

Pipeline position: shared helper consumed by `scripts/02_process.py`
(post-process preview, raw HaMeR keypoints) and `optimize/preview.py`
(post-optimize preview, smoothed keypoints). Mirrors the per-frame
overlay style of `scripts/01_hand_track.py` so the three stage previews
can be inspected side by side.

Why a single shared helper:
  Both 02 and optimize want the same "render skeleton overlay on source
  RGB" QA artefact. Duplicating the streaming render loop in two scripts
  invites schema drift. This module owns the IO (.r3d → frames, npz →
  HandDetections, mp4 writer) so callers just supply paths.

Schema accepted (hand fields under `<hand>_*`, hand ∈ {left, right}):
  REQUIRED  : K (3,3); per hand: wrist_cam (T,3), wrist_quat_cam (T,4)
              xyzw, joints_cam (T,21,3).
  OPTIONAL  : per hand: confidence (T,), valid (T,) bool, trim_first /
              trim_last (int). When `valid` absent, derived from
              isfinite(joints) & isfinite(wrist) & isfinite(quat).
  PASSTHROUGH: everything else ignored.

This matches both `.processed.npz` (no `valid`, has trim_first/trim_last)
and `.optimized.npz` (has `valid`, no per-hand trim_first/last at the
top level).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from utils.hand_tracker.base import HandDetection
from utils.hand_tracker.overlay import PreviewVideoWriter, draw_overlay
from utils.iphone.r3d_reader import iter_r3d_frames, read_r3d_metadata


_HANDS = ("left", "right")


def _per_frame_valid(npz: dict[str, np.ndarray], hand: str) -> np.ndarray | None:
    """Return a (T,) bool mask of "draw this hand at frame t".

    Order of preference:
      1. `<hand>_valid` if present (optimize stage uses this — already
         AND-combines in_trim with finite checks).
      2. Else derive: isfinite(wrist) & isfinite(quat) & isfinite(joints).
         Trim is implicit because 02 stores NaN outside trim.
    Returns None if the hand has no `joints_cam` array at all.
    """
    joints_key = f"{hand}_joints_cam"
    if joints_key not in npz:
        return None
    if f"{hand}_valid" in npz:
        return npz[f"{hand}_valid"].astype(bool)
    j = npz[joints_key]
    w = npz[f"{hand}_wrist_cam"]
    q = npz[f"{hand}_wrist_quat_cam"]
    return (
        np.all(np.isfinite(j), axis=(1, 2))
        & np.all(np.isfinite(w), axis=1)
        & np.all(np.isfinite(q), axis=1)
    )


def _build_detections(
    npz: dict[str, np.ndarray], frame_idx: int,
) -> list[HandDetection]:
    """Construct HandDetection list for one frame.

    Skips a hand whose row is non-finite or marked invalid so draw_overlay
    doesn't pinhole-project NaN.
    """
    out: list[HandDetection] = []
    for hand in _HANDS:
        valid = _per_frame_valid(npz, hand)
        if valid is None or not bool(valid[frame_idx]):
            continue
        wrist = npz[f"{hand}_wrist_cam"][frame_idx]
        joints = npz[f"{hand}_joints_cam"][frame_idx]
        quat = npz[f"{hand}_wrist_quat_cam"][frame_idx]
        if not (np.all(np.isfinite(wrist)) and np.all(np.isfinite(joints))
                and np.all(np.isfinite(quat))):
            continue

        # quat (xyzw) → rotvec; HandDetection schema expects axis-angle to
        # match what 01's HaMeR backend originally produced.
        try:
            wrist_rot = Rotation.from_quat(quat).as_rotvec()
        except ValueError:
            continue

        conf_arr = npz.get(f"{hand}_confidence")
        conf = float(conf_arr[frame_idx]) if conf_arr is not None else 1.0
        out.append(HandDetection(
            handedness=hand,
            wrist_pos=wrist.astype(np.float64),
            wrist_rot=wrist_rot.astype(np.float64),
            joints_3d=joints.astype(np.float64),
            confidence=conf,
            bbox=None,                 # joint-cloud bbox suffices for QA
        ))
    return out


def render_preview_from_npz(
    *,
    sid: str,
    stage_label: str,
    npz_path: Path,
    source_r3d: Path,
    out_mp4: Path,
    orientation: str = "auto",
    fps: int | None = None,
) -> dict:
    """Stream r3d frames + draw npz keypoints → mp4.

    Args:
        sid          : episode id, used in HUD overlay text only.
        stage_label  : short string ("02_processed" / "03_optimized" / ...)
                       prepended to HUD so the produced mp4 self-identifies.
        npz_path     : npz with the schema described in module docstring.
        source_r3d   : original .r3d for RGB source frames.
        out_mp4      : output mp4 path. Parent created if missing.
        orientation  : MUST match the orientation 01 was run with for this
                       batch — K stored in npz was baked under that rotation,
                       so any mismatch puts skeleton overlay in wrong frame.
        fps          : encode fps. None → derive from r3d metadata.
    """
    if not source_r3d.exists():
        raise FileNotFoundError(
            f"source r3d not found for preview: {source_r3d}"
        )
    arr = np.load(npz_path, allow_pickle=True)
    npz: dict[str, np.ndarray] = {
        k: arr[k] for k in arr.files if arr[k].dtype != object
    }
    if "K" not in npz:
        raise KeyError(f"{npz_path.name} missing K — cannot project")
    K = npz["K"].astype(np.float64)

    metadata, _jpgs = read_r3d_metadata(source_r3d)
    if fps is None:
        fps = int(round(metadata.get("fps", 60)))

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = PreviewVideoWriter(out_mp4, fps=fps)

    n_total = len(npz["timestamps_us"])
    n_drawn = 0
    n_with_hands = 0
    for frame_idx, rgb, _ts, _depth in iter_r3d_frames(
        source_r3d, read_depth=False, orientation=orientation,
    ):
        if frame_idx >= n_total:
            # r3d may carry trailing frames the npz didn't keep.
            break
        detections = _build_detections(npz, frame_idx)
        hud = (
            f"{stage_label}  {sid}  f={frame_idx}/{n_total}  "
            f"{len(detections)} hands"
        )
        bgr = draw_overlay(rgb, detections, K, hud_text=hud)
        writer.write(bgr)
        n_drawn += 1
        if detections:
            n_with_hands += 1

    writer.close()
    return {
        "out_mp4": str(out_mp4),
        "n_frames_written": n_drawn,
        "n_frames_with_hands": n_with_hands,
        "fps": fps,
    }
