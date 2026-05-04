"""
CLI: retarget any compatible `.npz` → `.qpos.npz` for a given robot.

Pipeline position: stage 5 of either recording line. Discovers source
files under a flat `<sid>/<sid>.<stage>.npz` layout; the chosen file is
either `.processed.npz` (raw, stage-2 output) or `.optimized.npz`
(post-`python -m optimize`). When both exist for the same sid, optimized
is preferred — running optimize is a deliberate "use this for retarget"
signal.

Layout convention (line-agnostic):
    INPUT  : <_LINE_ROOT>/<batch>/<stage>/<sid>/<sid>.{optimized,processed}.npz
    OUTPUT : <_LINE_ROOT>/<batch>/05_qpos_<robot>/<sid>/<sid>.qpos.npz
                                                    + <sid>.qpos.meta.json

The CLI handles I/O + schema validation + validity padding; the backend
stays pure (retarget-only). Schema validation enforces required keys
AND NaN density — sources that are too sparse for direct retarget are
rejected with a message pointing at `python -m optimize`.

Usage:
    conda activate opc-dex
    cd code/opc_data_pipeline

    # raw input (335-line, Shadow Hand right)
    python -m retarget --robot shadow --hand right \\
        --source-root output/gemini335/<batch>/02_processed

    # post-optimize input
    python -m retarget --robot shadow --hand right \\
        --source-root output/gemini335/<batch>/03_optimized

    # subset of episodes
    python -m retarget --robot shadow --hand right \\
        --source-root output/gemini335/<batch>/03_optimized \\
        --episodes sid_a,sid_b
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from retarget import (  # noqa: E402
    RetargetResult, get_backend, supported_envs, supported_robots,
)
from retarget.loader import (  # noqa: E402
    ProcessedSource,
    discover_episodes,
    load_npz_source,
)


_LINE_ROOT_BY_LINE = {
    "iphone":    _PROJECT_ROOT / "output" / "iphone",
    "gemini335": _PROJECT_ROOT / "output" / "gemini335",
}
_HAND_CHOICES = ("left", "right")


# =============================================================================
# Per-episode driver
# =============================================================================

def _resolve_batch(source_root: Path, batch: str | None) -> str:
    """Derive batch name from source_root convention `<line>/<batch>/<stage>/`."""
    if batch:
        return batch
    return source_root.resolve().parent.name


def _resolve_line_root(source_root: Path) -> Path:
    """Derive `<_LINE_ROOT>` from source_root convention."""
    return source_root.resolve().parent.parent


def _validate_source(
    backend_cls, source: ProcessedSource, hand: str,
) -> None:
    """Two-tier schema check before backends touch the data.

    1. Backend-declared required keys must exist (otherwise dex_retargeting
       chokes mid-batch with a KeyError that doesn't say which sid).
    2. NaN density on the chosen hand must be below the loader threshold,
       otherwise raise pointing the user at `python -m optimize`.
    """
    required = backend_cls.required_keys(hand)
    missing = [k for k in required if not source.has(k)]
    if missing:
        raise KeyError(
            f"{source.path.name} is missing required keys for "
            f"{backend_cls.name} backend (hand={hand}): {sorted(missing)}.\n"
            f"  Available: {sorted(source.raw)}"
        )
    source.validate_for_retarget(hand)


def _pad_to_full_length(
    result: RetargetResult, n_total: int, first: int, last: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert the trimmed (qpos, qpos_valid) back into a full-length
    array so downstream replay can index by the original timestamps."""
    n_joints = result.qpos.shape[1]
    qpos_full = np.full((n_total, n_joints), np.nan, dtype=np.float32)
    qpos_valid_full = np.zeros(n_total, dtype=bool)
    qpos_full[first:last] = result.qpos
    qpos_valid_full[first:last] = result.qpos_valid
    return qpos_full, qpos_valid_full


def _save_episode(
    out_dir: Path, sid: str, hand: str,
    timestamps_us: np.ndarray,
    qpos_full: np.ndarray, qpos_valid_full: np.ndarray,
    meta: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{sid}.qpos.npz"
    meta_path = out_dir / f"{sid}.qpos.meta.json"

    np.savez_compressed(
        npz_path,
        timestamps_us=timestamps_us.astype(np.int64),
        **{
            f"{hand}_qpos": qpos_full,
            f"{hand}_qpos_valid": qpos_valid_full,
        },
    )
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return npz_path, meta_path


def process_one_episode(
    npz_path: Path, backend, hand: str, output_root: Path,
    *, min_confidence: float = 0.0, env: str = "mujoco",
) -> dict:
    """Load → retarget → pad → save. Returns stats dict.

    Schema validation (required keys + NaN density) ran once in main()
    before any backend was constructed — see _validate_source.
    """
    source = load_npz_source(npz_path)
    sid = source.sid

    if not source.quality_passed(hand):
        first, last = source.trim_range(hand)
        print(f"  SKIP {sid} [{hand}]: quality_failed  trim=[{first},{last})")
        return {"sid": sid, "accepted": False, "reason": "quality_failed"}

    first, last = source.trim_range(hand)
    if last <= first:
        print(f"  SKIP {sid} [{hand}]: empty trim [{first},{last})")
        return {"sid": sid, "accepted": False, "reason": "empty_trim"}

    T_trim = last - first
    print(f"  [{sid}] hand={hand}  trim=[{first}..{last})  T={T_trim}")

    result = backend.retarget_episode(
        source, hand, min_confidence=min_confidence,
    )
    qpos_full, qpos_valid_full = _pad_to_full_length(
        result, source.n_frames_total, first, last,
    )

    K_flat = source.K.flatten().tolist()
    meta = {
        "schema_version": 3,           # bumped: backend-agnostic CLI
        "session_id": sid,
        "robot": backend.robot,
        "env": env,
        "hand": hand,
        "backend": backend.name,
        "retargeting_type": "position",
        "source_npz": str(npz_path),
        "joint_names": result.joint_names,
        "n_joints": len(result.joint_names),
        "trim": [first, last],
        "n_frames_total": source.n_frames_total,
        "n_frames_in_trim": int(T_trim),
        "n_frames_retarget_succeeded": int(result.qpos_valid.sum()),
        # K is the recording-time intrinsic. Replay uses it to derive
        # `fovy = 2*atan(cy/fy)` so the simulated camera matches the
        # real recording's image scale.
        "K_flat": K_flat,
        "extras": result.extras,
        "run_timestamp_iso": datetime.now().isoformat(),
    }

    out_dir = output_root / sid
    npz_out, _ = _save_episode(
        out_dir, sid, hand, source.timestamps_us,
        qpos_full, qpos_valid_full, meta,
    )
    print(f"    retarget OK: {int(result.qpos_valid.sum())}/{T_trim}")
    try:
        rel = npz_out.resolve().relative_to(_PROJECT_ROOT)
    except ValueError:
        rel = npz_out
    print(f"    → {rel}")

    return {
        "sid": sid, "accepted": True,
        "n_frames_in_trim": int(T_trim),
        "n_frames_retarget_succeeded": int(result.qpos_valid.sum()),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m retarget",
        description="Retarget .npz (raw .processed or post-optimize "
                    ".optimized) → .qpos.npz, per robot per hand. Pure "
                    "retarget — no smoothing, no interpolation. Frames "
                    "the optimizer can't solve come back as qpos_valid=False.",
    )
    p.add_argument("--robot", required=True,
                   choices=sorted(supported_robots()),
                   help="Robot hand / arm model")
    p.add_argument("--hand", required=True, choices=_HAND_CHOICES,
                   help="Which human hand to retarget")
    p.add_argument("--env", default="mujoco", choices=sorted(supported_envs()),
                   help="Target env recorded into qpos meta. SO-101 retargets "
                        "the same way for either; replay can override at run time.")
    p.add_argument("--source-root", type=Path, default=None,
                   help="Path to <line>/<batch>/<stage>/ — discovers "
                        "<sid>/<sid>.{optimized,processed}.npz, "
                        "preferring optimized when both exist.")
    p.add_argument("--processed-root", type=Path, default=None,
                   help="DEPRECATED alias for --source-root.")
    p.add_argument("--batch", type=str, default=None,
                   help="Batch name. Default: derived from --source-root parent.")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Override output dir. Default: "
                        "<line>/<batch>/05_qpos_<robot>/")
    p.add_argument("--urdf-dir", type=Path, default=None,
                   help="Root of dex-urdf/robots/hands/ (dex backends)")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated sid list; default all")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Drop frames below this confidence (default 0)")
    # SO-101 specific tuning. Ignored by other backends. Defaults match
    # retarget/so101.py So101Backend constructor; pass to override.
    p.add_argument("--so101-orientation-cost", type=float, default=None,
                   help="(so101) Orientation soft-task weight relative to "
                        "position_cost=1.0. Default 0.05 (pos-dominant). "
                        "Higher = orient tracks tighter / pos worse. "
                        "lerobot-seeed (placo) defaults to 0.01 — for "
                        "offline retargeting we lift it so wrist tilt "
                        "actually tracks.")
    p.add_argument("--so101-workspace-scale", type=float, default=None,
                   help="(so101) Force a scale factor for cam→arm "
                        "pinch motion. Default None = auto-fit so the "
                        "trajectory bbox half-diagonal lands at "
                        "workspace_arm_reach * workspace_fit_factor.")
    p.add_argument("--so101-orientation-tracking", type=str, default=None,
                   choices=["full", "yaw_only"],
                   help="(so101) full = match all 3 axes of wrist "
                        "orientation; yaw_only = only match the wrist "
                        "rotation around the gripper approach axis "
                        "(roll for SO-101). 'yaw_only' produces less "
                        "jitter when the source wrist orientation is "
                        "physically unreachable for a 5-DoF arm.")
    args = p.parse_args()

    if args.source_root is None and args.processed_root is None:
        p.error("--source-root is required (or its deprecated alias --processed-root)")
    source_root = args.source_root or args.processed_root
    if args.source_root is None:
        print("WARN: --processed-root is deprecated; use --source-root")

    batch = _resolve_batch(source_root, args.batch)
    line_root = _resolve_line_root(source_root)
    output_root = (
        args.output_root
        or (line_root / batch / f"05_qpos_{args.robot}")
    )

    sid_filter = (
        [e.strip() for e in args.episodes.split(",") if e.strip()]
        if args.episodes else None
    )
    npz_paths = discover_episodes(source_root, sid_filter)
    if not npz_paths:
        print("No source episodes found; nothing to do.")
        return 1

    print(f"retarget  ({args.robot}/{args.env}, hand={args.hand})")
    print(f"  source_root: {source_root}")
    print(f"  output_root: {output_root}")
    print(f"  episodes:    {len(npz_paths)}")
    print(f"  min_confidence: {args.min_confidence}")
    print()

    backend_cls = get_backend(args.robot, args.env)
    backend_kwargs = {}
    if args.urdf_dir is not None:
        backend_kwargs["urdf_dir"] = args.urdf_dir
    # SO-101 specific kwargs — only forward when given so other backends
    # don't choke on unknown args.
    if args.robot == "so101":
        if args.so101_orientation_cost is not None:
            backend_kwargs["orientation_cost"] = args.so101_orientation_cost
        if args.so101_workspace_scale is not None:
            backend_kwargs["workspace_scale"] = args.so101_workspace_scale
        if args.so101_orientation_tracking is not None:
            backend_kwargs["orientation_tracking"] = args.so101_orientation_tracking
    backend = backend_cls(robot=args.robot, hand=args.hand, **backend_kwargs)
    print(f"  joints ({backend.n_joints}): {backend.joint_names}")
    print()

    output_root.mkdir(parents=True, exist_ok=True)

    # Pre-flight key + NaN-density check across all episodes — fail fast
    # on schema drift or "you forgot to run optimize" instead of half-way
    # through a long batch.
    for npz_path in npz_paths:
        src = load_npz_source(npz_path)
        _validate_source(backend_cls, src, args.hand)

    stats = []
    for npz_path in npz_paths:
        s = process_one_episode(
            npz_path, backend, args.hand, output_root,
            min_confidence=args.min_confidence, env=args.env,
        )
        stats.append(s)

    print()
    print("=" * 60)
    accepted = [s for s in stats if s["accepted"]]
    skipped = [s for s in stats if not s["accepted"]]
    print(f"Done. accepted={len(accepted)}  skipped={len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"  SKIP {s['sid']}: {s['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
