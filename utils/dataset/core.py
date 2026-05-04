"""
Shared LeRobot v3 dataset writer infrastructure for both lines (iPhone .r3d
and Orbbec Gemini 335 .bag).

Pipeline position:
  - utils/dataset/iphone_writer.py    iPhone-specific upstream iterator
  - utils/dataset/orbbec_writer.py    Orbbec-specific upstream iterator
  - This file (core.py)               source-agnostic data contracts +
                                      packing helpers + LeRobot v3 driver

Why split:
  pyorbbecsdk2 (bag reader) hard-pins numpy<2.0 ABI; dex-retargeting (LEAP)
  needs numpy>=2.0. Both build_source scripts only call source-specific
  iterators in the lerobot env. Derived builders (0X_build_<robot>) live
  downstream of the source dataset and never touch upstream readers.

What this module owns:
  * EpisodeBundle / FrameContext data contracts (source-agnostic;
    T_world_cam is optional — iPhone fills it, Orbbec leaves None)
  * RGB / depth packing helpers (uint16 mm lossless + uint8 cm lossy)
  * load_processed_episode + per-hand validity helpers
  * build_lerobot_dataset main driver
  * meta/episodes/*.parquet custom-column merge + sidecar JSON

What this module does NOT own:
  * Upstream readers (lives in utils/iphone, utils/orbbec)
  * Upstream-specific episode iterators (iter_episodes_from_r3d /
    iter_episodes_from_bags) — see iphone_writer.py / orbbec_writer.py
  * Feature schema (build script declares features dict)
  * Per-frame state/action contents (build script supplies frame_builder)
  * Retargeting (lives in retarget/)
  * Per-embodiment dataset semantics
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


# =============================================================================
# Data contracts
# =============================================================================

@dataclass
class FrameContext:
    """Per-frame data passed to a build script's frame_builder callback.

    Source-agnostic: identical shape whether produced from a .r3d or .bag,
    or from a previously built source dataset.

    sid, t, rgb, K: as named.
    depth: (H, W) uint16 raw depth in millimeters, or None when depth was
        disabled. Lossless storage path (uint16 mm) preserves the full
        0..65535 mm range and per-mm precision; pack_depth_uint8_cm offers
        a lossy storage-sensitive fallback.
    T_world_cam: (4, 4) float64 ARKit per-frame extrinsics — iPhone-only,
        None for sources without a static world frame (Orbbec head-mounted).

    left_valid / right_valid:
      True iff this frame has real (non-placeholder) hand data. Combined
      check: in trim range AND episode quality_passed AND no NaN in joints.
      False = use placeholder (zeros + identity quat) so LeRobot doesn't
      see NaN (trainers crash on NaN).

    left_confidence / right_confidence: float in [0, 1]
      Per-frame HaMeR detector confidence, **passed through verbatim from
      01_track output** regardless of trim. 0 means "HaMeR did not detect
      this frame". Customers use this to mask / weight training loss
      independently of OPC's trim quality gate.

    left_wrist_pose / right_wrist_pose: (7,) float32 or None when invalid
        [x, y, z, qx, qy, qz, qw] cam-frame
    left_mano_joints / right_mano_joints: (21, 3) float32 or None
        cam-frame MANO joints
    """
    sid: str
    t: int
    rgb: np.ndarray
    depth: np.ndarray | None
    K: np.ndarray
    left_valid: bool
    right_valid: bool
    left_confidence: float
    right_confidence: float
    left_wrist_pose: np.ndarray | None
    right_wrist_pose: np.ndarray | None
    left_mano_joints: np.ndarray | None
    right_mano_joints: np.ndarray | None
    T_world_cam: np.ndarray | None = None
    # ----- v2 schema (world-frame, optional; populated only when 02 npz
    #       has v2 fields — i.e. iPhone-line schema_version >= 2) -----
    # Per-frame state in world frame (gravity-aligned via T_world_cam).
    # Position+quat 7-D (xyz + xyzw). None when out-of-trim / quality-failed
    # / NaN — frame_builder fills placeholder so LeRobot doesn't see NaN.
    left_wrist_pose_world: np.ndarray | None = None
    right_wrist_pose_world: np.ndarray | None = None
    # MANO 21 keypoints in world frame (3-D positions per joint). None on invalid.
    left_hand_keypoints_world: np.ndarray | None = None
    right_hand_keypoints_world: np.ndarray | None = None
    # Gripper open value [0=closed, 1=open] from MANO thumb-index distance.
    # None on invalid; 0.0 placeholder for invalid frames in frame_builder.
    left_gripper: float | None = None
    right_gripper: float | None = None
    # Action = next-frame state target (absolute, world-frame). For the last
    # frame of an episode, this equals the current frame ("hold-last"). None
    # follows the same invalid-frame convention as state.
    left_action_wrist_pose_world: np.ndarray | None = None
    right_action_wrist_pose_world: np.ndarray | None = None
    left_action_gripper: float | None = None
    right_action_gripper: float | None = None


@dataclass
class EpisodeBundle:
    """One episode's frames (streaming) + metadata for parquet merge.

    sid:    session id (matches source filename stem and folder)
    frames: lazy iterable of FrameContext — consumed once by the writer;
            generators are fine and preferred for memory bound
    extras: dict merged into the dataset's meta/episodes/*.parquet for this
            episode's row (K_flat, source paths, trim ranges, etc.)
    """
    sid: str
    frames: Iterable[FrameContext]
    extras: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# RGB / depth packing helpers (used by frame_builders)
# =============================================================================

def pack_rgb(ctx: FrameContext, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize ctx.rgb to (H, W). INTER_AREA for downscaling photographic
    content. Passthrough zero-copy when sizes already match."""
    h, w = target_hw
    if ctx.rgb.shape[0] == h and ctx.rgb.shape[1] == w:
        return ctx.rgb
    return cv2.resize(ctx.rgb, (w, h), interpolation=cv2.INTER_AREA)


def pack_depth_uint16_mm(
    ctx: FrameContext, target_hw: tuple[int, int],
) -> np.ndarray | None:
    """Resize ctx.depth (uint16 mm, shape (H, W)) to target_hw.

    Lossless storage path: depth is the camera's raw uint16 mm reading.
    INTER_NEAREST avoids invented values; bilinear would corrupt 0-pixels
    (which mean "no return") into mid-range fake distances.
    """
    if ctx.depth is None:
        return None
    h, w = target_hw
    if ctx.depth.shape[0] == h and ctx.depth.shape[1] == w:
        return ctx.depth
    return cv2.resize(ctx.depth, (w, h), interpolation=cv2.INTER_NEAREST)


def pack_depth_uint8_cm(
    ctx: FrameContext, target_hw: tuple[int, int],
) -> np.ndarray | None:
    """Encode depth (uint16 mm) as uint8 cm 3-channel image.

    Lossy fallback for storage-sensitive captures: depth_cm = depth_mm // 10
    (clipped to 0..255). Stored as 3-channel uint8 image so LeRobot v3 can
    apply video-style storage. Range cap: 0..2.55 m. Range above is clipped.

    Recover meters at load time:
        depth_m = (image[..., 0].astype(float32)) / 100
    """
    if ctx.depth is None:
        return None
    h, w = target_hw
    depth_mm = ctx.depth
    if depth_mm.shape[0] != h or depth_mm.shape[1] != w:
        depth_mm = cv2.resize(
            depth_mm, (w, h), interpolation=cv2.INTER_NEAREST,
        )
    depth_cm = np.clip(depth_mm // 10, 0, 255).astype(np.uint8)
    return np.stack([depth_cm, depth_cm, depth_cm], axis=-1)


# =============================================================================
# Episode loading + helpers
# =============================================================================

def load_processed_episode(
    npz_path: Path, meta_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load 02_process output: <sid>.processed.npz + sidecar meta.json.

    allow_pickle=True is required by the iPhone line (.processed.npz packs
    string fields into object arrays per scripts/01_hand_track.py). Files
    are produced by our own pipeline so the security risk is bounded.
    Orbbec-line npz contains only numerical arrays and works either way.
    """
    if meta_path is None:
        meta_path = npz_path.with_suffix(".meta.json")
    arr = np.load(npz_path, allow_pickle=True)
    npz = {k: arr[k] for k in arr.files}
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return npz, meta


def compute_union_range(npz: dict[str, np.ndarray]) -> tuple[int, int]:
    """Return (first, last) covering whichever hand(s) had any in-trim data.

    Hands with empty trim (start==stop) contribute nothing to the lower
    bound. Both empty → (0, 0).
    """
    lf = int(npz["left_trim_first"])
    ll = int(npz["left_trim_last"])
    rf = int(npz["right_trim_first"])
    rl = int(npz["right_trim_last"])
    starts = [s for s, e in ((lf, ll), (rf, rl)) if e > s]
    if not starts:
        return 0, 0
    return min(starts), max(ll, rl)


def hand_valid_at(
    npz: dict[str, np.ndarray], hand: str, t_in_track: int,
) -> bool:
    """True iff t is in <hand>_trim AND <hand>_quality_passed."""
    if not bool(npz[f"{hand}_quality_passed"]):
        return False
    first = int(npz[f"{hand}_trim_first"])
    last = int(npz[f"{hand}_trim_last"])
    return first <= t_in_track < last


def _extract_wrist_pose(
    npz: dict[str, np.ndarray], hand: str, t: int,
) -> np.ndarray:
    """7D float32: pos(3) + quat_xyzw(4) in cam frame (v1)."""
    pos = npz[f"{hand}_wrist_cam"][t].astype(np.float32)
    quat = npz[f"{hand}_wrist_quat_cam"][t].astype(np.float32)
    return np.concatenate([pos, quat], axis=0)


def _extract_wrist_pose_world(
    npz: dict[str, np.ndarray], hand: str, t: int,
) -> np.ndarray | None:
    """7D float32: pos(3) + quat_xyzw(4) in world frame (v2). None if
    the npz has no v2 fields (schema_version < 2)."""
    pos_key = f"{hand}_wrist_world"
    quat_key = f"{hand}_wrist_quat_world"
    if pos_key not in npz or quat_key not in npz:
        return None
    pos = npz[pos_key][t].astype(np.float32)
    quat = npz[quat_key][t].astype(np.float32)
    return np.concatenate([pos, quat], axis=0)


def has_v2_world_fields(npz: dict[str, np.ndarray]) -> bool:
    """True iff this npz carries v2 world-frame fields (set by 02 v2)."""
    return "right_wrist_world" in npz or "left_wrist_world" in npz


def compute_action_arrays(npz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pre-compute per-episode v2 action arrays (next-frame state, hold-last).

    Action[t] = state[t+1] for t in [0, T-1); action[T-1] = state[T-1].
    Hold-last avoids NaN at the episode boundary so LeRobot trainers don't
    crash; matches the convention of most LeRobot v3 datasets that emit
    absolute action targets.

    Returns a dict with `<hand>_action_wrist_world`, `<hand>_action_wrist_quat_world`,
    `<hand>_action_gripper`. Empty dict if v2 fields are absent.
    """
    if not has_v2_world_fields(npz):
        return {}
    out: dict[str, np.ndarray] = {}
    for hand in ("left", "right"):
        for src, key in (
            (f"{hand}_wrist_world", f"{hand}_action_wrist_world"),
            (f"{hand}_wrist_quat_world", f"{hand}_action_wrist_quat_world"),
            (f"{hand}_gripper", f"{hand}_action_gripper"),
        ):
            if src not in npz:
                continue
            state = npz[src]
            shifted = np.empty_like(state)
            shifted[:-1] = state[1:]
            shifted[-1] = state[-1]    # hold-last
            out[key] = shifted
    return out


def build_per_hand_fields(
    npz: dict[str, np.ndarray], t_in_track: int,
    *, action_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build the per-hand fields of FrameContext for one frame.

    Source-agnostic: both upstream iterators (iter_episodes_from_r3d,
    iter_episodes_from_bags) call this for every frame in trim and splat
    the result into FrameContext(**hand_fields, ...).

    Three-tier validity check (raw-first — 02_process no longer fills /
    smooths so wrist may also be NaN at no-detection frames):
      1. trim+quality (hand_valid_at): in per-hand trim AND quality_passed
      2. wrist finite check: NaN-within-trim is possible
      3. joints finite check: same — never filled by 02_process

    `{side}_valid` is the AND of all three. Invalid frames return None for
    pose/joints — frame_builder fills placeholders so LeRobot doesn't see
    NaN (trainers crash on NaN). `{side}_confidence` is HaMeR's raw value
    verbatim regardless of trim, so customers can mask / weight loss
    independently of OPC's trim quality gate.

    v2 schema: when npz has world-frame fields (set by 02_process v2 ≥ 2),
    also populates `{side}_wrist_pose_world`, `{side}_hand_keypoints_world`,
    `{side}_gripper`. When `action_arrays` is provided (pre-computed via
    `compute_action_arrays`), also populates `{side}_action_wrist_pose_world`
    and `{side}_action_gripper` (next-frame state, hold-last at boundary).
    All v2 fields are None when 02 npz is v1 only.
    """
    out: dict[str, Any] = {}
    has_v2 = has_v2_world_fields(npz)

    for side in ("left", "right"):
        in_trim = hand_valid_at(npz, side, t_in_track)
        out[f"{side}_confidence"] = float(npz[f"{side}_confidence"][t_in_track])

        # v1 cam-frame
        pose_raw = _extract_wrist_pose(npz, side, t_in_track) if in_trim else None
        pose_finite = pose_raw is not None and np.isfinite(pose_raw).all()
        out[f"{side}_wrist_pose"] = pose_raw if pose_finite else None

        joints_raw = (
            npz[f"{side}_joints_cam"][t_in_track].astype(np.float32)
            if in_trim else None
        )
        joints_finite = (
            joints_raw is not None and np.isfinite(joints_raw).all()
        )
        out[f"{side}_mano_joints"] = joints_raw if joints_finite else None

        out[f"{side}_valid"] = in_trim and pose_finite and joints_finite

        # v2 world-frame (only if 02 produced v2 fields)
        if has_v2:
            pose_w = _extract_wrist_pose_world(npz, side, t_in_track) if in_trim else None
            pose_w_finite = pose_w is not None and np.isfinite(pose_w).all()
            out[f"{side}_wrist_pose_world"] = pose_w if pose_w_finite else None

            kp_w = (
                npz[f"{side}_joints_world"][t_in_track].astype(np.float32)
                if in_trim and f"{side}_joints_world" in npz else None
            )
            kp_w_finite = kp_w is not None and np.isfinite(kp_w).all()
            out[f"{side}_hand_keypoints_world"] = kp_w if kp_w_finite else None

            gripper_v = (
                float(npz[f"{side}_gripper"][t_in_track])
                if in_trim and f"{side}_gripper" in npz else None
            )
            out[f"{side}_gripper"] = (
                gripper_v if (gripper_v is not None and np.isfinite(gripper_v)) else None
            )

            # v2 action (next-frame target, pre-computed)
            if action_arrays:
                a_pos_key = f"{side}_action_wrist_world"
                a_quat_key = f"{side}_action_wrist_quat_world"
                a_grip_key = f"{side}_action_gripper"
                if (in_trim and a_pos_key in action_arrays
                        and a_quat_key in action_arrays):
                    a_pos = action_arrays[a_pos_key][t_in_track].astype(np.float32)
                    a_quat = action_arrays[a_quat_key][t_in_track].astype(np.float32)
                    a_pose = np.concatenate([a_pos, a_quat], axis=0)
                    out[f"{side}_action_wrist_pose_world"] = (
                        a_pose if np.isfinite(a_pose).all() else None
                    )
                else:
                    out[f"{side}_action_wrist_pose_world"] = None

                if in_trim and a_grip_key in action_arrays:
                    g = float(action_arrays[a_grip_key][t_in_track])
                    out[f"{side}_action_gripper"] = g if np.isfinite(g) else None
                else:
                    out[f"{side}_action_gripper"] = None
    return out


# =============================================================================
# Main driver
# =============================================================================

def build_lerobot_dataset(
    *,
    episodes: Iterable[EpisodeBundle],
    output_dir: Path,
    repo_id: str,
    fps: int,
    img_size: tuple[int, int],
    task_description: str,
    features: dict[str, Any],
    frame_builder: Callable[[FrameContext], dict[str, Any]],
    robot_type: str = "human_hand",
    image_writer_threads: int = 2,
) -> dict[str, Any]:
    """Run the writer over a sequence of EpisodeBundles.

    The episodes argument is a lazy iterable — each EpisodeBundle in turn
    yields its frames (also lazy) so memory stays O(1 frame).

    Parameters
    ----------
    episodes :          source of EpisodeBundles. Caller picks the iterator
                        (iter_episodes_from_r3d / iter_episodes_from_bags).
                        Episode-level filtering is the iterator's job.
    output_dir :        destination LeRobot v3 dataset root.
    fps :               must match capture rate (30 for Gemini 335 / iPhone).
    img_size :          (H, W) — frame_builder is responsible for resizing
                        rgb/depth to this via pack_rgb / pack_depth_*.
    features :          LeRobot v3 features dict; must declare every key
                        that frame_builder will produce.
    frame_builder :     (FrameContext) → dict with the keys in `features`.
                        `task` is auto-added by the driver.
    robot_type :        embodiment label persisted in meta/info.json.
                        Caller passes the source-specific value (e.g.
                        "human_hand_iphone_source", "human_hand_335_source").
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # LeRobotDataset.create insists on a fresh directory; do not pre-create.
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=robot_type,
        use_videos=True,
        image_writer_threads=image_writer_threads,
    )

    ep_extras: dict[int, dict[str, Any]] = {}
    n_accepted = 0
    n_frames_total = 0

    for bundle in episodes:
        n_written = 0
        print(f"  [{bundle.sid}]")
        for ctx in bundle.frames:
            frame_dict = frame_builder(ctx)
            frame_dict.setdefault("task", task_description)
            dataset.add_frame(frame_dict)
            n_written += 1

        dataset.save_episode()
        ep_extras[n_accepted] = bundle.extras
        n_accepted += 1
        n_frames_total += n_written
        print(f"    wrote {n_written} frames")

    # Officially-recommended finalization (lerobot_dataset.py:657 docstring):
    # "Close the parquet writers. This function needs to be called after
    #  data collection/conversion, else footer metadata won't be written
    #  to the parquet files."
    # Without this, custom-column merge below races AsyncImageWriter and
    # parquet writer threads.
    dataset.finalize()

    # Persist OPC extras two ways:
    #   1. JSON sidecar at meta/opc_episode_extras.json — discoverable,
    #      easy to read from any consumer (no parquet dep)
    #   2. As columns appended to meta/episodes/*.parquet — survive on
    #      disk but LeRobot loader's select_columns drops them; included
    #      for advanced users who read the parquet directly
    _write_opc_extras_sidecar(output_dir, ep_extras)
    _append_episode_meta_columns(output_dir, ep_extras)

    return {
        "n_episodes_accepted": n_accepted,
        "n_frames_total": n_frames_total,
    }


# =============================================================================
# Internals
# =============================================================================

def _write_opc_extras_sidecar(
    output_dir: Path, ep_extras: dict[int, dict[str, Any]],
) -> None:
    """Write meta/opc_episode_extras.json keyed by episode_index.

    LeRobot v3 loaders silently drop user-added columns from
    meta/episodes/*.parquet (`select_columns` whitelist). To keep our
    extras discoverable for any consumer (including non-pandas users),
    mirror them into a sidecar JSON file under meta/.
    """
    if not ep_extras:
        return
    sidecar = output_dir / "meta" / "opc_episode_extras.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "description": (
            "OPC-specific per-episode metadata. Keyed by episode_index "
            "(string). Mirrored as columns in meta/episodes/*.parquet "
            "but those are dropped by LeRobot v3 loader at load time."
        ),
        "episodes": {str(k): v for k, v in ep_extras.items()},
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _append_episode_meta_columns(
    output_dir: Path, ep_extras: dict[int, dict[str, Any]],
) -> None:
    """Merge custom columns into LeRobot v3 meta/episodes/*.parquet.

    v3 chunks episodes into multiple shards; we touch every shard so the
    schema stays uniform. Missing keys for an episode_index become None.
    """
    if not ep_extras:
        return
    episodes_dir = output_dir / "meta" / "episodes"
    if not episodes_dir.exists():
        print(f"  WARN: {episodes_dir} not found; skipping episode meta merge")
        return
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"  WARN: no parquet shards under {episodes_dir}")
        return

    all_keys: set[str] = set()
    for extras in ep_extras.values():
        all_keys.update(extras.keys())

    for pq in parquet_files:
        df = pd.read_parquet(pq)
        for key in all_keys:
            df[key] = [
                ep_extras.get(int(idx), {}).get(key)
                for idx in df["episode_index"]
            ]
        df.to_parquet(pq, index=False)
