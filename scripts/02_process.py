"""
Per-r3d tracking-data cleanup → cam-frame .processed.npz (raw-first).

Pipeline position: Step 2/6 (iPhone-line) — processing layer.
Input:  output/iphone/01_tracking/<sid>/<sid>.tracking.npz   (cam-frame raw HaMeR)
Output: output/iphone/02_processed/<sid>/
        ├── <sid>.processed.npz             cleaned per-hand arrays
        └── <sid>.processed.meta.json       per-hand stats + filter params

Per-hand pipeline (utils/process/iphone.py — mirror of utils/process/orbbec.py):
  trim → quality_check → rotation_jump_diagnostic
  Raw-first: no signal modification (no fill, no smooth, no world transform).
  NaN frames within trim are preserved. Downstream (03_build_source /
  04_build_so101) decides how to handle them.

Conversion: axis-angle wrist_rot_cam (HaMeR native, from 01) → xyzw quat
with hemisphere continuity (q[t]·q[t-1] >= 0). Matches 335 schema. Wrist
smoothing lives in `optimize/` — run `python -m optimize ...` to apply
One-Euro / interp etc. on top of the raw `.processed.npz` output.

Schema v2 (npz fields, T = same length as input tracking):
  timestamps_us               (T,)     int64
  K                           (3, 3)   float64
  T_world_cam                 (T, 4, 4) float64   ARKit per-frame poses (passthrough)
  source                      str
  episode_name                str

  Per-hand cam-frame (v1, kept for backward compat with retarget/loader.py):
    <hand>_wrist_cam          (T, 3)   float64
    <hand>_wrist_quat_cam     (T, 4)   float64   scipy xyzw, hemisphere-continuous
    <hand>_joints_cam         (T,21,3) float64   raw HaMeR joints (NaN at gaps)
    <hand>_bbox               (T, 4)   float64   MediaPipe detector bbox
    <hand>_confidence         (T,)     float64

  Per-hand world-frame (v2, derived via cam→world using T_world_cam):
    <hand>_wrist_world        (T, 3)   float64   wrist position in ARKit world
    <hand>_wrist_quat_world   (T, 4)   float64   wrist orientation in ARKit world
    <hand>_joints_world       (T,21,3) float64   MANO 21 joints in ARKit world
    <hand>_gripper            (T,)     float64   [0=closed,1=open] from thumb↔index dist

  Per-hand flags:
    <hand>_trim_first         int32    inclusive start into original T
    <hand>_trim_last          int32    exclusive stop into original T
    <hand>_quality_passed     bool     True iff trim+quality both passed

Usage:
    conda activate lerobot
    cd code/opc_data_pipeline

    # Single tracking npz
    python scripts/02_process.py --track output/iphone/01_tracking/<sid>/<sid>.tracking.npz

    # All tracking npz under a directory
    python scripts/02_process.py --track-dir output/iphone/01_tracking/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.process.core import ProcessConfig, process_hand           # noqa: E402
from utils.process.world_frame import (                              # noqa: E402
    transform_points_cam_to_world,
    transform_quats_cam_to_world,
)
from utils.hand_tracker.preview_from_npz import (                    # noqa: E402
    render_preview_from_npz,
)

from scripts._schema import HAND_FIELDS                              # noqa: E402


# v2 derived fields appended to per-hand output (in addition to HAND_FIELDS):
#   wrist_world / wrist_quat_world / joints_world: cam-frame fields lifted
#   into ARKit world frame using per-frame T_world_cam (gravity-aligned).
#   gripper: scalar [0, 1] open value derived from MANO thumb-tip ↔ index-tip
#   distance (rigid-invariant, computed once here so downstream doesn't repeat).
HAND_FIELDS_V2 = (
    "wrist_world",
    "wrist_quat_world",
    "joints_world",
    "gripper",
)

# MANO joint indices for gripper derivation (matches retarget/so101.py).
_MANO_THUMB_TIP_IDX = 4
_MANO_INDEX_TIP_IDX = 8
# Linear range mapping thumb-index distance → gripper [0, 1].
_GRIPPER_OPEN_M = 0.10    # ≥ 10 cm → fully open (1.0)
_GRIPPER_CLOSE_M = 0.02   # ≤  2 cm → fully closed (0.0)


def _compute_world_fields(
    cam_hand: dict,
    T_world_cam_slice: np.ndarray,
) -> dict:
    """Lift one trimmed cam-frame hand dict into world frame + add gripper.

    Args:
        cam_hand: dict with cam-frame trimmed arrays (wrist_cam, wrist_quat_cam,
            joints_cam, bbox, confidence). Output of process_hand.
        T_world_cam_slice: (T_trimmed, 4, 4) per-frame extrinsics matching
            the same trim window.

    Returns:
        dict with HAND_FIELDS_V2 keys (wrist_world, wrist_quat_world,
        joints_world, gripper). All NaN-propagating.
    """
    wrist_world = transform_points_cam_to_world(
        cam_hand["wrist_cam"], T_world_cam_slice,
    )
    wrist_quat_world = transform_quats_cam_to_world(
        cam_hand["wrist_quat_cam"], T_world_cam_slice,
    )
    joints_world = transform_points_cam_to_world(
        cam_hand["joints_cam"], T_world_cam_slice,
    )
    # Gripper: rigid-invariant under cam↔world, compute from cam joints to
    # avoid extra precision loss from world transform on a difference.
    joints_cam = cam_hand["joints_cam"]
    thumb = joints_cam[:, _MANO_THUMB_TIP_IDX, :]
    index_tip = joints_cam[:, _MANO_INDEX_TIP_IDX, :]
    dist_m = np.linalg.norm(thumb - index_tip, axis=1)
    span = max(_GRIPPER_OPEN_M - _GRIPPER_CLOSE_M, 1e-6)
    # NaN propagates through the norm; clip ignores NaN (returns NaN).
    gripper = np.clip((dist_m - _GRIPPER_CLOSE_M) / span, 0.0, 1.0)
    return {
        "wrist_world": wrist_world.astype(np.float64),
        "wrist_quat_world": wrist_quat_world.astype(np.float64),
        "joints_world": joints_world.astype(np.float64),
        "gripper": gripper.astype(np.float64),
    }


_LINE_ROOT = _PROJECT_ROOT / "output" / "iphone"
_STAGE = "02_processed"


def _resolve_batch(args) -> str:
    """Auto-derive batch from input path. Convention: input .tracking.npz
    lives at <_LINE_ROOT>/<batch>/01_tracking/<sid>/<sid>.tracking.npz, so
    the batch is `--track-dir.parent.name` or `--track.parent.parent.parent.name`.
    """
    if args.batch:
        return args.batch
    if args.track_dir is not None:
        # --track-dir = <batch>/01_tracking/  → batch = parent.name
        return args.track_dir.resolve().parent.name
    if args.track is not None:
        # --track = <batch>/01_tracking/<sid>/<sid>.tracking.npz
        # Walk up: file → <sid>/ → 01_tracking/ → <batch>/
        return args.track.resolve().parents[2].name
    raise ValueError(
        "Cannot derive --batch: pass --batch <name> or --track-dir <dir>."
    )


def _axis_angle_to_quat_hemisphere(rotvec: np.ndarray) -> np.ndarray:
    """Convert (T, 3) axis-angle → (T, 4) xyzw quat with hemisphere continuity.

    NaN axis-angle rows produce NaN quat rows. Hemisphere fix: for each
    valid quat, sign-flip if dot product with the previous valid quat is
    negative — keeps q and -q (same rotation) on the same side so consumers
    that interpolate / derivative the quat don't see spurious sign jumps.
    """
    n = len(rotvec)
    out = np.full((n, 4), np.nan, dtype=np.float64)
    valid = ~np.isnan(rotvec[:, 0])
    if not valid.any():
        return out

    # Bulk convert valid rows.
    out[valid] = Rotation.from_rotvec(rotvec[valid]).as_quat()  # xyzw

    # Hemisphere continuity (one pass over the timeline).
    prev = None
    for i in range(n):
        if not valid[i]:
            continue
        if prev is not None and float(np.dot(out[i], prev)) < 0.0:
            out[i] = -out[i]
        prev = out[i]
    return out


def _load_tracking_npz(npz_path: Path) -> dict:
    """Return dict with timestamps_us, K, T_world_cam, source, episode_name,
    and per-hand HAND_FIELDS arrays (axis-angle converted to xyzw quat)."""
    d = np.load(npz_path, allow_pickle=True)
    out = {
        "timestamps_us": d["timestamps_us"],
        "K": d["K"],
        "T_world_cam": d["T_world_cam"],
        "source": str(d["source"]) if "source" in d.files else "unknown",
        "episode_name": str(d["episode_name"]) if "episode_name" in d.files else npz_path.stem,
    }
    for hand_name in ("left", "right"):
        rot_aa = d[f"{hand_name}_wrist_rot_cam"]
        quat = _axis_angle_to_quat_hemisphere(rot_aa)
        out[hand_name] = {
            "wrist_cam":      d[f"{hand_name}_wrist_cam"].copy(),
            "wrist_quat_cam": quat,
            "joints_cam":     d[f"{hand_name}_joints_cam"].copy(),
            "bbox":           d[f"{hand_name}_bbox"].copy(),
            "confidence":     d[f"{hand_name}_confidence"].copy(),
        }
    return out


def _save_processed_npz(
    path: Path,
    *,
    timestamps_us: np.ndarray,
    K: np.ndarray,
    T_world_cam: np.ndarray,
    source: str,
    episode_name: str,
    raw_hands: dict,
    processed_hands: dict,
    stats_per_hand: dict,
) -> None:
    """Write npz with all per-hand fields + trim/quality flags.

    Schema-stability rationale (matches 335 / LeRobot / NEP-12):
      - All hand fields ALWAYS present so downstream code never KeyErrors
      - <hand>_quality_passed flags whether the data was vetted (True) or
        left raw (False)
      - Quality-failed hands keep post-trim raw values for inspection
    Episode-level drop is 03_build_source's concern, not 02's.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "timestamps_us": timestamps_us.astype(np.int64),
        "K": K.astype(np.float64),
        "T_world_cam": T_world_cam.astype(np.float64),
        "source": np.array(source, dtype=object),
        "episode_name": np.array(episode_name, dtype=object),
    }
    n_total = len(timestamps_us)
    for hand_name in ("left", "right"):
        s = stats_per_hand[hand_name]
        passed = s["quality_passed"]
        h = processed_hands[hand_name] if passed else None
        if h is None:
            sl_first, sl_last = s["trim_slice"]
            h = {f: raw_hands[hand_name][f][sl_first:sl_last]
                 for f in HAND_FIELDS}
        first, last = s["trim_slice"]
        for f in HAND_FIELDS:
            full_shape = (n_total,) + h[f].shape[1:]
            fill = 0.0 if f == "confidence" else np.nan
            full = np.full(full_shape, fill, dtype=h[f].dtype)
            full[first:last] = h[f]
            payload[f"{hand_name}_{f}"] = full

        # v2 derived: cam→world + gripper. Computed on the trimmed slice and
        # padded to full length with NaN (matches NaN-padding for cam fields).
        # Quality-failed hands still get world fields (computed from raw trimmed
        # data) so downstream sees a uniform schema; mask via quality_passed.
        T_wc_slice = T_world_cam[first:last]
        derived = _compute_world_fields(h, T_wc_slice)
        for f in HAND_FIELDS_V2:
            full_shape = (n_total,) + derived[f].shape[1:]
            full = np.full(full_shape, np.nan, dtype=np.float64)
            full[first:last] = derived[f]
            payload[f"{hand_name}_{f}"] = full

        payload[f"{hand_name}_trim_first"] = np.int32(first)
        payload[f"{hand_name}_trim_last"] = np.int32(last)
        payload[f"{hand_name}_quality_passed"] = bool(passed)
    np.savez_compressed(path, **payload)


def _write_processed_meta(
    path: Path,
    *,
    session_id: str,
    source_track: Path,
    source_r3d: Path | None,
    cfg: ProcessConfig,
    n_frames: int,
    fps: float,
    stats_per_hand: dict,
    accepted_hands: list[str],
    rejected_hands: list[tuple[str, str]],
) -> None:
    """Sidecar JSON summarising what was kept and what was rejected."""
    meta = {
        "session_id": session_id,
        "source_track": str(source_track),
        "source_r3d": str(source_r3d) if source_r3d else None,
        "n_frames_total": n_frames,
        "fps": fps,
        "accepted_hands": accepted_hands,
        "rejected_hands": [
            {"hand": h, "reason": r} for h, r in rejected_hands
        ],
        "filter_config": {
            "min_detection_rate": cfg.min_detection_rate,
            "max_gap_frames": cfg.max_gap_frames,
            "min_duration_s": cfg.min_duration_s,
            "bbox_iou_trim_threshold": cfg.bbox_iou_trim_threshold,
            "rot_jump_warn_rad": cfg.rot_jump_warn_rad,
        },
        "wrist_smoothing": "raw_pass_through",
        "joint_smoothing": "raw_pass_through",
        "world_frame": (
            "ARKit T_world_cam stored per-frame; v2 npz fields "
            "(wrist_world, wrist_quat_world, joints_world) are derived by "
            "cam→world transform; cam-frame fields preserved for backward "
            "compat with v1 readers (retarget/loader.py)."
        ),
        "schema_version": 2,
        "schema_changes_v2": [
            "Added per-hand wrist_world (T,3), wrist_quat_world (T,4), "
            "joints_world (T,21,3) — cam→world via T_world_cam.",
            "Added per-hand gripper (T,) [0,1] from MANO thumb-tip↔index-tip "
            f"distance ({_GRIPPER_CLOSE_M}m closed → {_GRIPPER_OPEN_M}m open).",
            "cam-frame fields (wrist_cam, wrist_quat_cam, joints_cam) "
            "preserved unchanged for backward compatibility.",
        ],
        "per_hand_stats": stats_per_hand,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def _discover_r3d(track_npz: Path) -> Path | None:
    """Read source_r3d from tracking.meta.json sidecar (if present)."""
    meta_path = track_npz.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    sr = meta.get("source_r3d")
    if not sr:
        return None
    p = Path(sr)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p if p.exists() else None


def process_track(track_npz: Path, *, output_root: Path,
                   cfg: ProcessConfig,
                   save_preview: bool = True,
                   orientation: str = "auto") -> dict:
    """Process one .tracking.npz → <sid>/<sid>.processed.npz under output_root.

    Returns a stats dict for batch aggregation.
    """
    sid = track_npz.stem.replace(".tracking", "")
    out_dir = output_root / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_out = out_dir / f"{sid}.processed.npz"
    meta_out = out_dir / f"{sid}.processed.meta.json"

    print("=" * 70)
    print(f"Process: {track_npz.name}")
    r3d_path = _discover_r3d(track_npz)
    print(f"  r3d:    {r3d_path}")
    print(f"  output: {out_dir}")

    track = _load_tracking_npz(track_npz)
    ts_us = track["timestamps_us"]
    K = track["K"]
    T_world_cam = track["T_world_cam"]
    n_frames = len(ts_us)

    if n_frames >= 2:
        # int64 microseconds → fps via median dt. ts == 0 (no-detect frames
        # don't fill ts here; iPhone fills every frame) but be robust anyway.
        valid_ts = ts_us[ts_us > 0]
        if len(valid_ts) >= 2:
            dt_us = np.median(np.diff(valid_ts))
            fps_est = float(1e6 / dt_us) if dt_us > 0 else 30.0
        else:
            fps_est = 30.0
    else:
        fps_est = 30.0
    print(f"  frames: {n_frames}, fps: {fps_est:.1f}")

    raw_hands = {h: track[h] for h in ("left", "right")}

    t0 = time.monotonic()
    stats_per_hand: dict[str, dict] = {}
    processed_hands: dict[str, dict | None] = {}
    accepted_hands: list[str] = []
    rejected_hands: list[tuple[str, str]] = []

    for hand_name in ("left", "right"):
        print(f"\n  [{hand_name}]")
        processed, stats = process_hand(
            raw_hands[hand_name], fps=fps_est, K=K, cfg=cfg,
        )
        stats_per_hand[hand_name] = stats
        processed_hands[hand_name] = processed
        first, last = stats["trim_slice"]
        print(f"    trim:   [{first}..{last}) → {stats['n_after_trim']} frames "
              f"(leading -{stats['n_trimmed_leading']}, "
              f"trailing -{stats['n_trimmed_trailing']}, "
              f"iou-rejected {stats['trim_iou_rejected']})")
        if processed is None:
            print(f"    quality: FAIL ({stats['skip_reason']}) — kept in npz "
                  f"with {hand_name}_quality_passed=False (raw, untrimmed-mid)")
            rejected_hands.append((hand_name, stats["skip_reason"]))
            continue
        print(f"    quality: PASS")
        accepted_hands.append(hand_name)
        rd = stats["rotation_diagnostic"]
        if rd:
            print(f"    rotation delta: median={rd['median_delta_rad']:.3f}rad, "
                  f"max={rd['max_delta_rad']:.3f}rad, "
                  f">{cfg.rot_jump_warn_rad}rad jumps={rd['n_jumps_above_threshold']}")

    if not accepted_hands:
        print(f"\n  ALL HANDS REJECTED — npz still has full schema with "
              f"quality_passed=False on all hands")

    _save_processed_npz(
        npz_out,
        timestamps_us=ts_us, K=K, T_world_cam=T_world_cam,
        source=track["source"], episode_name=track["episode_name"],
        raw_hands=raw_hands, processed_hands=processed_hands,
        stats_per_hand=stats_per_hand,
    )
    _write_processed_meta(
        meta_out,
        session_id=sid, source_track=track_npz, source_r3d=r3d_path,
        cfg=cfg, n_frames=n_frames, fps=fps_est,
        stats_per_hand=stats_per_hand,
        accepted_hands=accepted_hands, rejected_hands=rejected_hands,
    )

    print(f"\n  npz:  {npz_out.name}")
    print(f"        accepted: {accepted_hands or 'NONE'}")
    if rejected_hands:
        for h, r in rejected_hands:
            print(f"        rejected: {h} ({r}) — flagged in npz, raw kept")
    print(f"  meta: {meta_out.name}")

    # Preview mp4: post-trim + post-quality keypoints overlaid on source RGB.
    # Same overlay style as 01 and 03 so users can flip across stages and
    # see what each step changed. Raw HaMeR signal — no smoothing here, by
    # design (02 is "raw-first"). Skip silently when r3d unavailable so a
    # missing source doesn't fail the whole batch.
    if save_preview:
        if r3d_path is None:
            print(f"  preview skipped: no source r3d for {sid}")
        else:
            mp4_out = out_dir / f"{sid}_preview.mp4"
            try:
                stats = render_preview_from_npz(
                    sid=sid,
                    stage_label="02_processed",
                    npz_path=npz_out,
                    source_r3d=r3d_path,
                    out_mp4=mp4_out,
                    orientation=orientation,
                )
                print(
                    f"  preview: {stats['n_frames_with_hands']}/"
                    f"{stats['n_frames_written']} frames with hands "
                    f"@ {stats['fps']} fps  → {mp4_out.name}"
                )
            except Exception as exc:                 # noqa: BLE001
                print(f"  preview FAILED for {sid}: {exc}")

    print(f"  total: {time.monotonic() - t0:.1f}s")

    return {"sid": sid, "n_frames": n_frames,
            "accepted": accepted_hands, "rejected": rejected_hands}


def main():
    parser = argparse.ArgumentParser(
        description="Per-r3d tracking → trim+quality .processed.npz (raw-first)",
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--track", type=Path, default=None,
                   help="Single .tracking.npz from 01_hand_track")
    g.add_argument("--track-dir", type=Path, default=None,
                   help="Directory tree to scan for .tracking.npz files")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch name (output goes to "
                             f"{_LINE_ROOT}/<batch>/{_STAGE}/<sid>/). "
                             "Default: derived from --track-dir parent.")
    parser.add_argument("--output-root", type=Path, default=None,
                        help=f"Override per-batch root. Default: "
                             f"{_LINE_ROOT}/<batch>/{_STAGE}/")
    parser.add_argument("--bbox-iou-trim-threshold", type=float, default=0.3,
                        help="Trim leading/trailing frames where MediaPipe "
                             "bbox and projected joint-cloud bbox have IoU "
                             "below this. Catches HaMeR 'flying skeleton' "
                             "frames at edges. 0=disabled.")
    parser.add_argument("--min-detection-rate", type=float, default=0.5)
    parser.add_argument("--max-gap-frames", type=int, default=30)
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    parser.add_argument("--no-preview-video", action="store_true",
                        help="Skip rendering the per-episode QA preview mp4. "
                             "Default: write <sid>_preview.mp4 next to the "
                             "processed npz, drawn with the trimmed/quality-"
                             "filtered (but still raw, unsmoothed) keypoints.")
    parser.add_argument("--orientation",
                        choices=("auto", "landscape", "portrait"),
                        default="auto",
                        help="r3d frame rotation policy for the preview "
                             "render. MUST match the orientation 01_hand_track "
                             "was run with for this batch — K stored in the "
                             "npz was baked under that rotation. Default "
                             "'auto' matches 01's default.")
    args = parser.parse_args()

    if args.track is not None:
        tracks = [args.track]
    else:
        tracks = sorted(args.track_dir.rglob("*.tracking.npz"))
        if not tracks:
            raise FileNotFoundError(
                f"No .tracking.npz under {args.track_dir}")
    print(f"Found {len(tracks)} tracking file(s) to process")

    batch = _resolve_batch(args)
    output_root = args.output_root or (_LINE_ROOT / batch / _STAGE)
    print(f"Batch:  {batch}")
    print(f"Output: {output_root}")

    cfg = ProcessConfig(
        min_detection_rate=args.min_detection_rate,
        max_gap_frames=args.max_gap_frames,
        min_duration_s=args.min_duration_s,
        bbox_iou_trim_threshold=args.bbox_iou_trim_threshold,
    )

    save_preview = not args.no_preview_video
    print(f"Preview video: {'on' if save_preview else 'off'}"
          f"  orientation={args.orientation}")
    results = [
        process_track(
            t, output_root=output_root, cfg=cfg,
            save_preview=save_preview, orientation=args.orientation,
        )
        for t in tracks
    ]
    print("=" * 70)
    print(f"Done: {len(results)} track(s) processed under {output_root}")


if __name__ == "__main__":
    main()
