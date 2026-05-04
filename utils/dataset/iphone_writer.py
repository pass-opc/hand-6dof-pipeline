"""
iPhone-specific upstream iterator for the LeRobot v3 dataset writer.

Produces EpisodeBundles by streaming Record3D .r3d archives matched to
.processed.npz tracking output. iPhone-line frame correspondence is 1:1
between .r3d and .processed.npz (01_hand_track wrote one npz row per
.r3d frame), so no timestamp lookup is needed.

Shared infrastructure (FrameContext, EpisodeBundle, packing helpers,
build_lerobot_dataset driver) lives in utils/dataset/core.py — import
from there. This module only defines what is iPhone-specific:
  * iter_episodes_from_r3d        — top-level entrypoint
  * _iter_r3d_frames_as_context   — generator (Record3D float32 m → uint16 mm,
                                    fills T_world_cam from ARKit per-frame)
  * _r3d_extras                   — per-episode parquet row extras
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from utils.dataset.core import (
    EpisodeBundle,
    FrameContext,
    build_per_hand_fields,
    compute_action_arrays,
    compute_union_range,
    load_processed_episode,
)
from utils.iphone.r3d_reader import iter_r3d_frames


def iter_episodes_from_r3d(
    processed_npz_paths: list[Path],
    r3d_path_for: Callable[[str], Path],
    *,
    include_depth: bool = True,
    orientation: str = "auto",
) -> Iterable[EpisodeBundle]:
    """Yield EpisodeBundles by streaming .r3d archives matched to track frames.

    iPhone-line frame correspondence is 1:1 between .r3d and the .processed.npz
    (01_hand_track wrote one npz row per .r3d frame). No timestamp lookup
    needed — frame_idx i in .r3d is npz row i.

    Filters out episodes where (a) both hands failed quality, (b) .r3d file
    is missing on disk.

    `orientation` MUST match the value passed to 01_hand_track for this
    batch. K and T_world_cam read from the .processed.npz are already in
    the orientation-resolved canvas frame (01 baked them in); RGB / depth
    must be decoded under the same mode so projections line up.
    """
    for npz_path in processed_npz_paths:
        sid = npz_path.stem.replace(".processed", "")
        npz, _meta = load_processed_episode(npz_path)
        l_pass = bool(npz.get("left_quality_passed", False))
        r_pass = bool(npz.get("right_quality_passed", False))
        if not (l_pass or r_pass):
            print(f"  SKIP {sid}: both_hands_failed_quality")
            continue
        r3d_path = r3d_path_for(sid)
        if not r3d_path.exists():
            print(f"  SKIP {sid}: r3d_not_found at {r3d_path}")
            continue

        union = compute_union_range(npz)
        extras = _r3d_extras(sid, r3d_path, npz, union)
        yield EpisodeBundle(
            sid=sid,
            frames=_iter_r3d_frames_as_context(
                sid, r3d_path, npz, union, include_depth,
                orientation=orientation,
            ),
            extras=extras,
        )


def _iter_r3d_frames_as_context(
    sid: str, r3d_path: Path, npz: dict[str, np.ndarray],
    union_range: tuple[int, int], include_depth: bool,
    *,
    orientation: str = "auto",
) -> Iterable[FrameContext]:
    """Generator: r3d frames within union → FrameContext, hand data looked up
    from npz at the matching track-frame index. Fills T_world_cam from
    ARKit per-frame extrinsics (iPhone-only)."""
    first, last = union_range
    if last <= first:
        return
    K = npz["K"].astype(np.float64)
    T_world_cam_all = npz["T_world_cam"].astype(np.float64)

    # Pre-compute v2 action arrays per episode (next-frame state, hold-last).
    # Empty dict if 02 npz lacks v2 fields, in which case build_per_hand_fields
    # falls back to v1-only output (action_* fields stay None).
    action_arrays = compute_action_arrays(npz)

    # Restrict frame_indices to the union range so iter_r3d_frames doesn't
    # decode out-of-trim frames.
    frame_indices = set(range(first, last))

    out_t = 0
    for i, rgb, _ts, depth_m in iter_r3d_frames(
        r3d_path, read_depth=include_depth, frame_indices=frame_indices,
        orientation=orientation,
    ):
        t_in_track = i

        # Convert Record3D float32 m → uint16 mm (lossless within sensor noise
        # floor of ~1cm; full 0..65 m range vs uint8 cm capped at 2.55 m).
        if include_depth and depth_m is not None:
            depth_mm = (np.clip(depth_m, 0, 65.535) * 1000.0).astype(np.uint16)
        else:
            depth_mm = None

        yield FrameContext(
            sid=sid, t=out_t,
            rgb=rgb, depth=depth_mm, K=K,
            T_world_cam=T_world_cam_all[t_in_track],
            **build_per_hand_fields(npz, t_in_track, action_arrays=action_arrays),
        )
        out_t += 1


def _r3d_extras(
    sid: str, r3d_path: Path, npz: dict[str, np.ndarray],
    union: tuple[int, int],
) -> dict[str, Any]:
    K = npz["K"].astype(np.float64)
    return {
        "session_id": sid,
        "source_r3d": str(r3d_path),
        "K_flat": K.flatten().tolist(),
        "left_trim_first": int(npz["left_trim_first"]),
        "left_trim_last": int(npz["left_trim_last"]),
        "right_trim_first": int(npz["right_trim_first"]),
        "right_trim_last": int(npz["right_trim_last"]),
        "left_quality_passed": bool(npz["left_quality_passed"]),
        "right_quality_passed": bool(npz["right_quality_passed"]),
        "union_first": int(union[0]),
        "union_last": int(union[1]),
    }
