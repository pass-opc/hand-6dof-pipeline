"""
Per-r3d bare-hand detection from Record3D captures → cam-frame .tracking.npz
(+ optional preview mp4).

Pipeline position: Step 1/6 (iPhone-line) — perception layer.
Input:  output/iphone/00_record/<sid>.r3d (or external --r3d-dir)
Output: output/iphone/01_tracking/<sid>/
        ├── <sid>.tracking.npz        per-frame hand keypoints + axis-angle (cam frame)
        ├── <sid>.tracking.meta.json  backend versions + detection stats
        └── <sid>_preview.mp4         annotated overlay for human QA (default on)

Per-frame pipeline:
  1. Read RGB + LiDAR depth + T_world_cam[t] (Record3D / ARKit passthrough)
  2. HaMeR detect → wrist_cam + joints_cam + bbox (cam frame, virtual scale)
  3. Spatial tracker corrects MediaPipe handedness flips
  4. LiDAR depth correction: scale all 21 joints along their rays from camera
     origin so wrist lands at sensor depth (multiplicative; preserves 2D
     projection AND gives metric-correct 3D — matches 335 01_track + DexCap).

Schema (npz fields, T = video frame count, NaN at no-detection frames):
  timestamps_us         (T,)     int64    Record3D timestamps × 1e6
  K                     (3, 3)   float64  iPhone intrinsics (portrait)
  T_world_cam           (T, 4, 4) float64 ARKit poses (per frame)
  source                str               backend name
  episode_name          str               r3d file stem
  coord_frame           str               "camera"
  <hand>_wrist_cam      (T, 3)   float64  cam frame, NaN if no detection
  <hand>_wrist_rot_cam  (T, 3)   float64  axis-angle, cam frame
  <hand>_joints_cam     (T,21,3) float64  cam frame, LiDAR-corrected
  <hand>_bbox           (T, 4)   float64  [x1,y1,x2,y2] pixels
  <hand>_confidence     (T,)     float64  detector confidence (0=no detection)

Why .npz over old .pkl: inspectable (np.load(p).files); cross-language;
lossless compressed; one file per session (mirrors 335 layout). Trim,
quality, axis-angle→quat conversion all happen in 02_process.

Usage:
    conda activate lerobot
    cd code/opc_data_pipeline

    # Single .r3d
    python scripts/01_hand_track.py --r3d path/to/<sid>.r3d

    # All .r3d under a directory
    python scripts/01_hand_track.py --r3d-dir path/to/raw/

    # Force every recording to landscape (sensor-native) — needed when iOS
    # screen lock baked landscape captures into portrait canvases
    python scripts/01_hand_track.py --r3d-dir <dir> --orientation landscape

    # Pick output root (default output/iphone/<batch>/01_tracking/)
    python scripts/01_hand_track.py --r3d-dir <dir> --output-root other/

    # Skip LiDAR depth correction (HaMeR-only metric, less accurate)
    python scripts/01_hand_track.py --r3d-dir <dir> --no-depth

    # Skip preview mp4 rendering (faster, no QA video)
    python scripts/01_hand_track.py --r3d-dir <dir> --no-preview-video
"""

from __future__ import annotations

import os

# Windows OpenMP duplicate-DLL fix: cv2 (libomp.dll) and torch
# (libiomp5md.dll) each ship their own OpenMP runtime; mediapipe's
# detector + HaMeR's torch backbone running in the same process aborts
# at first cross-call without this. Same pattern as scripts335/01_track
# and retarget/dex_hands. MUST be set before any torch / cv2 import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.hand_tracker.depth_correction import (
    back_project_depth, print_depth_correction_summary,
)
from utils.hand_tracker import HandTracker, create_tracker
from utils.hand_tracker.overlay import PreviewVideoWriter, draw_overlay
from utils.iphone.r3d_reader import (
    ORIENTATION_MODES, iter_r3d_frames, read_iphone_intrinsics, read_poses,
    read_r3d_metadata,
)
from utils.hand_tracker.spatial_tracker import SpatialHandTracker


_LINE_ROOT = _PROJECT_ROOT / "output" / "iphone"
_STAGE = "01_tracking"


def _resolve_batch(args) -> str:
    """Auto-derive batch name when --batch not given.

    Source dir basename (`--r3d-dir`) or single-file parent name fits how
    OPC names capture campaigns (e.g. `test_HaMeR_4_15`). Pass --batch
    explicitly to override.
    """
    if args.batch:
        return args.batch
    if args.r3d_dir is not None:
        return args.r3d_dir.resolve().name
    if args.r3d is not None:
        return args.r3d.resolve().parent.name
    raise ValueError(
        "Cannot derive --batch: pass --batch <name> or --r3d-dir <dir>."
    )


@dataclass
class HandTrackConfig:
    """Configuration for hand perception pipeline."""
    backend: str = "hamer"
    read_depth: bool = True
    orientation: str = "auto"
    save_preview_video: bool = True
    preview_fps: int | None = None  # None = use Record3D metadata fps


def _make_empty_hand_arrays(n_frames: int) -> dict[str, np.ndarray]:
    """NaN-filled per-hand arrays. confidence defaults to 0 so a sum>0
    distinguishes 'ever seen' from 'never seen'.

    No gripper_width: it's derivable as ||joints_cam[4] - joints_cam[8]||
    downstream (matches 335 / scripts/_schema HAND_FIELDS convention).
    """
    return {
        "wrist_cam":     np.full((n_frames, 3), np.nan, dtype=np.float64),
        "wrist_rot_cam": np.full((n_frames, 3), np.nan, dtype=np.float64),
        "joints_cam":    np.full((n_frames, 21, 3), np.nan, dtype=np.float64),
        "bbox":          np.full((n_frames, 4), np.nan, dtype=np.float64),
        "confidence":    np.zeros(n_frames, dtype=np.float64),
    }


def _save_tracking_npz(
    path: Path,
    *,
    timestamps_us: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    source: str,
    episode_name: str,
) -> None:
    """Write per-frame arrays to a compressed npz. Per-hand keys are
    flattened (left_wrist_cam, right_joints_cam, ...) so np.load returns a
    flat key namespace mirroring 335's track.npz layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "timestamps_us": timestamps_us.astype(np.int64),
        "K": K.astype(np.float64),
        "T_world_cam": T_world_cam.astype(np.float64),
        # Strings stored as numpy 0-D unicode arrays (npz friendly).
        "source": np.array(source, dtype=object),
        "episode_name": np.array(episode_name, dtype=object),
        "coord_frame": np.array("camera", dtype=object),
    }
    for hand_name, arrs in (("left", left), ("right", right)):
        for f, arr in arrs.items():
            payload[f"{hand_name}_{f}"] = arr
    # allow_pickle=True because object arrays carry the str fields. The npz
    # is otherwise pure numeric; the strings are a few-byte ledger.
    np.savez_compressed(path, **payload)


def _write_tracking_meta(
    path: Path,
    *,
    session_id: str,
    r3d_path: Path,
    backend: str,
    duration_s: float,
    n_frames: int,
    n_left_detected: int,
    n_right_detected: int,
    depth_correction_stats: dict | None,
) -> None:
    """Sidecar JSON for human inspection + 02_process to discover .r3d."""
    meta = {
        "session_id": session_id,
        "timestamp_iso": datetime.now().isoformat(),
        "source_r3d": str(r3d_path),
        "backend": backend,
        "duration_s": round(duration_s, 3),
        "n_frames": n_frames,
        "detection": {
            "left_frames": int(n_left_detected),
            "right_frames": int(n_right_detected),
            "left_rate": round(n_left_detected / max(1, n_frames), 4),
            "right_rate": round(n_right_detected / max(1, n_frames), 4),
        },
        "depth_correction": depth_correction_stats,
        "world_frame": "ARKit T_world_cam stored per-frame in npz",
        "wrist_rotation_format": "axis_angle (cam frame; 02_process converts to xyzw)",
        "schema_version": 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def process_episode(
    r3d_path: Path,
    *,
    output_root: Path,
    config: HandTrackConfig,
    tracker: HandTracker,
) -> dict:
    """Process one .r3d episode → <sid>/<sid>.tracking.npz under output_root.

    Returns a stats dict for batch aggregation.
    """
    sid = r3d_path.stem
    out_dir = output_root / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sid}.tracking.npz"
    meta_path = out_dir / f"{sid}.tracking.meta.json"
    video_path = out_dir / f"{sid}_preview.mp4"

    print("=" * 70)
    print(f"Track: {r3d_path.name}")
    print(f"  output: {out_dir}")

    metadata, jpg_names = read_r3d_metadata(r3d_path)
    n_frames = len(jpg_names)

    T_world_cam = read_poses(r3d_path, orientation=config.orientation)
    if len(T_world_cam) != n_frames:
        raise ValueError(
            f"{r3d_path.name}: poses count {len(T_world_cam)} != "
            f"jpg count {n_frames}"
        )

    K_real = read_iphone_intrinsics(metadata, orientation=config.orientation)
    tracker.set_focal_length_px(float(K_real[0, 0]))

    # Preview video — draw_overlay + PyAV-backed mp4 writer (same as 335 line).
    # fps default tracks the source recording so playback matches reality;
    # caller can override (e.g. drop to 30 for storage).
    preview_fps = config.preview_fps or int(metadata.get("fps", 30))
    preview = (PreviewVideoWriter(video_path, fps=preview_fps)
               if config.save_preview_video else None)

    left = _make_empty_hand_arrays(n_frames)
    right = _make_empty_hand_arrays(n_frames)
    timestamps_s = np.zeros(n_frames, dtype=np.float64)

    spatial_tracker = SpatialHandTracker()
    depth_stats_left: list[dict] = []
    depth_stats_right: list[dict] = []
    n_left_det = 0
    n_right_det = 0

    for i, rgb, ts, depth in tqdm(
        iter_r3d_frames(
            r3d_path,
            read_depth=config.read_depth,
            orientation=config.orientation,
        ),
        total=n_frames,
        desc=f"  {sid}",
        unit="frame",
    ):
        timestamps_s[i] = ts

        detections = tracker.detect(rgb)
        detections = spatial_tracker.update(detections)

        for det in detections:
            is_left = det.handedness == "left"
            hand_data = left if is_left else right
            stats_list = depth_stats_left if is_left else depth_stats_right
            if is_left:
                n_left_det += 1
            else:
                n_right_det += 1

            pos_cam = det.wrist_pos.copy()
            rot_cam = det.wrist_rot.copy()
            joints_cam = det.joints_3d.copy()

            # LiDAR residual correction: probe sensor depth at HaMeR's
            # projected wrist pixel, then SCALE all 21 joints along their
            # rays from camera origin so wrist lands at sensor depth.
            # Multiplicative scaling (joints *= z_real/z_pred) preserves
            # 2D pixel projection (because (sx)/(sz) == x/z) AND gives
            # metric-correct 3D — matches 335 01_track / DexCap convention.
            if config.read_depth and depth is not None and det.bbox is not None:
                rgb_h, rgb_w = rgb.shape[:2]
                dep_h, dep_w = depth.shape[:2]
                bbox_center = np.array([
                    (det.bbox[0] + det.bbox[2]) / 2.0,
                    (det.bbox[1] + det.bbox[3]) / 2.0,
                ])
                bp_pos, stats = back_project_depth(
                    bbox_center, depth, K_real,
                    K_real_src_wh=(rgb_w, rgb_h),
                    depth_wh=(dep_w, dep_h),
                )
                stats["z_hamer"] = float(pos_cam[2])
                stats_list.append(stats)
                if bp_pos is not None and pos_cam[2] > 0:
                    scale = bp_pos[2] / pos_cam[2]
                    joints_cam = joints_cam * scale
                    pos_cam = pos_cam * scale  # == bp_pos

            hand_data["wrist_cam"][i] = pos_cam
            hand_data["wrist_rot_cam"][i] = rot_cam
            hand_data["joints_cam"][i] = joints_cam
            if det.bbox is not None:
                hand_data["bbox"][i] = det.bbox
            hand_data["confidence"][i] = det.confidence
            # gripper_width intentionally NOT stored — derivable as
            # ||joints[4] - joints[8]|| downstream. Matches 335 schema.

            # Mirror corrected pose into det so the preview overlay draws
            # what we actually save (335 01_track had a bug where preview
            # rendered raw HaMeR joints, hiding depth-correction errors).
            det.wrist_pos = pos_cam
            det.joints_3d = joints_cam

        if preview is not None:
            hud = (f"f={i:4d}  L={n_left_det:4d}  R={n_right_det:4d}  "
                   f"orient={config.orientation}")
            bgr = draw_overlay(rgb, detections, K_real, hud_text=hud)
            preview.write(bgr)

    if preview is not None:
        preview.close()

    # Detection summary
    for hand_name, n_det in (("left", n_left_det), ("right", n_right_det)):
        rate = n_det / n_frames if n_frames > 0 else 0
        print(f"  {hand_name}: {n_det}/{n_frames} frames ({rate:.1%})")

    # Depth-correction summary
    dc_summary: dict | None = None
    if config.read_depth:
        all_stats = depth_stats_left + depth_stats_right
        if all_stats:
            valid = [
                s for s in all_stats
                if s.get("valid")
                and np.isfinite(s.get("z_hamer", float("nan")))
                and s["z_hamer"] > 0
            ]
            if valid:
                z_lidar = np.array([s["z_lidar"] for s in valid])
                z_hamer = np.array([s["z_hamer"] for s in valid])
                scales = z_lidar / z_hamer
                dc_summary = {
                    "n_attempted": len(all_stats),
                    "n_valid": len(valid),
                    "valid_rate": round(len(valid) / len(all_stats), 4),
                    "z_lidar_mean_m": round(float(z_lidar.mean()), 4),
                    "z_hamer_mean_m": round(float(z_hamer.mean()), 4),
                    "scale_mean": round(float(scales.mean()), 4),
                    "scale_std": round(float(scales.std()), 4),
                }
            else:
                dc_summary = {"n_attempted": len(all_stats), "n_valid": 0}
        for hand_name, stats_list in (("left", depth_stats_left),
                                       ("right", depth_stats_right)):
            if stats_list:
                print(f"\n  [{hand_name} depth correction]")
                print_depth_correction_summary(stats_list)

    # Convert seconds → microseconds for cross-line consistency with 335.
    timestamps_us = (timestamps_s * 1e6).astype(np.int64)

    duration_s = float(timestamps_s[-1] - timestamps_s[0]) if n_frames > 0 else 0.0

    _save_tracking_npz(
        npz_path,
        timestamps_us=timestamps_us,
        K=K_real,
        T_world_cam=T_world_cam,
        left=left,
        right=right,
        source=tracker.get_backend_name(),
        episode_name=sid,
    )
    _write_tracking_meta(
        meta_path,
        session_id=sid,
        r3d_path=r3d_path,
        backend=config.backend,
        duration_s=duration_s,
        n_frames=n_frames,
        n_left_detected=n_left_det,
        n_right_detected=n_right_det,
        depth_correction_stats=dc_summary,
    )
    print(f"  npz:  {npz_path.name}")
    print(f"  meta: {meta_path.name}")

    return {"sid": sid, "n_frames": n_frames,
            "n_left": n_left_det, "n_right": n_right_det}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-r3d hand 6DoF tracking → cam-frame .tracking.npz. "
            "ARKit pose passed through; LiDAR-corrected metric scale."
        ),
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--r3d", type=Path, default=None,
                   help="Single .r3d file to process")
    g.add_argument("--r3d-dir", type=Path, default=None,
                   help="Directory tree to scan for .r3d files")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch name (output goes to "
                             f"{_LINE_ROOT}/<batch>/{_STAGE}/<sid>/). "
                             "Default: --r3d-dir basename.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help=f"Override per-batch root. Default: "
                             f"{_LINE_ROOT}/<batch>/{_STAGE}/")
    parser.add_argument("--backend", type=str, default="hamer",
                        choices=["hamer", "wilor"])
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable LiDAR metric-scale correction")
    parser.add_argument("--orientation", type=str, default="auto",
                        choices=list(ORIENTATION_MODES),
                        help=("Output canvas orientation. 'auto' = legacy "
                              "(W>H rotated CCW90 to portrait). 'landscape' "
                              "= restore sensor-native landscape on every "
                              "recording (use when iOS screen lock baked "
                              "landscape captures into portrait canvases). "
                              "'portrait' = mirror of landscape."))
    parser.add_argument("--no-preview-video", action="store_true",
                        help="Skip rendering the per-episode QA preview mp4")
    parser.add_argument("--preview-fps", type=int, default=None,
                        help="Override preview mp4 fps (default: source fps)")
    args = parser.parse_args()

    if args.r3d is not None:
        r3d_files = [args.r3d]
    else:
        r3d_files = sorted(args.r3d_dir.rglob("*.r3d"))
        if not r3d_files:
            raise FileNotFoundError(f"No .r3d files under {args.r3d_dir}")
    print(f"Found {len(r3d_files)} .r3d file(s) to process")

    batch = _resolve_batch(args)
    output_root = args.output_root or (_LINE_ROOT / batch / _STAGE)
    print(f"Batch:  {batch}")
    print(f"Output: {output_root}")

    config = HandTrackConfig(
        backend=args.backend,
        read_depth=not args.no_depth,
        orientation=args.orientation,
        save_preview_video=not args.no_preview_video,
        preview_fps=args.preview_fps,
    )
    print(f"Backend: {config.backend}")
    print(f"Orientation: {config.orientation}")
    print(f"Preview video: {'on' if config.save_preview_video else 'off'}")
    tracker = create_tracker(config.backend)
    print(f"Tracker: {tracker.get_backend_name()}")

    results = []
    for r3d_path in r3d_files:
        results.append(process_episode(
            r3d_path,
            output_root=output_root,
            config=config,
            tracker=tracker,
        ))

    print("=" * 70)
    print(f"Done: {len(results)} episode(s) processed under {output_root}")


if __name__ == "__main__":
    main()
