"""
Build the OPC source LeRobot v3 dataset (v2 schema, world-frame) from
02_process v2 output + .r3d.

Pipeline position: Step 3/6 (iPhone-line) — embodiment-agnostic packaging.
This is the only Step-3-family script that touches .r3d archives.
Per-embodiment derived datasets / customer training pipelines consume this
source dataset — they don't re-decode .r3d.

Input:  output/iphone/02_processed/<sid>/<sid>.processed.npz   (v2: world-frame fields present)
        output/iphone/00_record/<sid>.r3d (or external --r3d-root)
Output: output/iphone/03_source/                                (LeRobot v3 dataset)

Schema v2 (world frame, bimanual stacking, mirrors AgiBot World style):
  observation.images.rgb            video (H, W, 3) uint8 — H.264/AV1 mp4
  observation.depth                 uint16 (H, W) raw mm — lossless
                                    OR uint8 (H, W, 3) cm — lossy CLI fallback

  # State (world frame, gravity-aligned via T_world_cam from 02 v2)
  observation.state.wrist_pose      float32 (2, 7)     L/R × (xyz + scipy xyzw quat)
  observation.state.wrist_valid     float32 (2,)       L/R × {0=placeholder, 1=real}
  observation.state.hand_keypoints  float32 (2, 21, 3) L/R × MANO 21 joints in world
  observation.state.gripper         float32 (2,)       L/R × [0=closed, 1=open]
                                                       from MANO thumb-tip↔index-tip

  observation.left_confidence       float32 (1,)       HaMeR detector confidence raw
  observation.right_confidence      float32 (1,)       (mask / weight loss with this)

  # Action (absolute target = next-frame state, hold-last at episode boundary)
  action.wrist_pose                 float32 (2, 7)     same shape as state
  action.gripper                    float32 (2,)       same

  # Per-frame extrinsics (kept so customer can re-anchor / project)
  observation.T_world_cam           float32 (4, 4)

Per-frame validity strategy:
  Episode frame range = union of left_trim and right_trim. Within union,
  invalid (out-of-trim or NaN) frames get placeholder values + valid=0:
      wrist_pose  = [0,0,0, 0,0,0,1]   (identity quat)
      hand_keypoints = zeros(21, 3)
      gripper     = 0.0
  Customer masks via observation.state.wrist_valid before computing loss.

Why bimanual `(2, ...)` stacking:
  Matches AgiBot World v2.1 dim convention (`/state/end/position` is (N,2,3)).
  Single-array bimanual reduces field-name proliferation and lets policies
  treat L/R uniformly; single-arm consumers index `[..., 0]` or `[..., 1]`.

Usage:
    conda activate lerobot
    cd code/opc_data_pipeline

    # Default uint16 mm depth (cross-line consistent + lossless within sensor noise)
    python scripts/03_build_source.py --repo-id opc/iphone_source_v1 --task "pick_red_cup"

    # uint8 cm fallback (lossy, but ~120x smaller for storage-sensitive captures)
    python scripts/03_build_source.py --repo-id ... --task ... --depth-encoding uint8_cm

    # Filter to specific episodes
    python scripts/03_build_source.py --repo-id ... --task ... --episodes EF-XXX,EF-YYY
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.dataset.core import (   # noqa: E402
    FrameContext,
    build_lerobot_dataset,
    pack_depth_uint8_cm,
    pack_depth_uint16_mm,
    pack_rgb,
)
from utils.dataset.iphone_writer import iter_episodes_from_r3d   # noqa: E402


_LINE_ROOT = _PROJECT_ROOT / "output" / "iphone"
_STAGE = "03_source"


def _resolve_batch(args) -> str:
    """Derive batch from --processed-root convention `<_LINE_ROOT>/<batch>/02_processed/`."""
    if args.batch:
        return args.batch
    pr = args.processed_root.resolve()
    return pr.parent.name

_IDENTITY_WRIST_POSE = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32,
)
_ZERO_JOINTS = np.zeros((21, 3), dtype=np.float32)


# =============================================================================
# Feature schema
# =============================================================================

def make_features(img_hw: tuple[int, int], *, include_depth: bool,
                   depth_encoding: str) -> dict:
    """LeRobot v3 features schema (v2: world-frame, bimanual stacked).

    Wrist + keypoints convention:
      - Frame: ARKit world frame (gravity-aligned per ARSession start).
              Recover original cam-frame data via observation.T_world_cam.
      - Position units: meters.
      - Orientation: scipy xyzw quaternion. Recover with
              `R = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()`.

    Bimanual layout (matches AgiBot World v2.1 (N, 2, ...) convention):
      Index 0 = left hand, Index 1 = right hand. Single-arm consumers
      index `[..., 0]` or `[..., 1]`; bimanual policies treat both uniformly.

    Action convention (absolute, hold-last at episode boundary):
      action[t] = state[t+1]; for t = T-1, action equals state[T-1].

    Depth encoding:
      - "uint16_mm": (H, W) uint16 raw mm, lossless.
      - "uint8_cm":  (H, W, 3) uint8 cm-quantized, lossy. Capped at 2.55 m.
                    Recover meters via `image[..., 0].astype(float32) / 100`.
    """
    h, w = img_hw
    wrist_axes = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    features: dict = {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },

        # State (world-frame, bimanual stacked)
        "observation.state.wrist_pose": {
            "dtype": "float32",
            "shape": (2, 7),
            "names": ["hand", "pose"],
        },
        "observation.state.wrist_valid": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["hand"],
        },
        "observation.state.hand_keypoints": {
            "dtype": "float32",
            "shape": (2, 21, 3),
            "names": ["hand", "joint", "xyz"],
        },
        "observation.state.gripper": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["hand"],
        },

        # Confidence (kept per-hand for masking / loss weighting)
        "observation.left_confidence": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["confidence"],
        },
        "observation.right_confidence": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["confidence"],
        },

        # Action (absolute, world-frame, bimanual stacked)
        "action.wrist_pose": {
            "dtype": "float32",
            "shape": (2, 7),
            "names": ["hand", "pose"],
        },
        "action.gripper": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["hand"],
        },

        # ARKit per-frame extrinsics (so customer can recover cam-frame data
        # / re-anchor world / project depth back into pixels). 64 B/frame.
        "observation.T_world_cam": {
            "dtype": "float32",
            "shape": (4, 4),
            "names": ["row", "col"],
        },
    }
    if include_depth:
        if depth_encoding == "uint16_mm":
            features["observation.depth"] = {
                "dtype": "uint16",
                "shape": (h, w),
                "names": ["height", "width"],
            }
        elif depth_encoding == "uint8_cm":
            features["observation.depth"] = {
                "dtype": "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            }
        else:
            raise ValueError(
                f"Unknown depth_encoding {depth_encoding!r}; "
                f"use 'uint16_mm' or 'uint8_cm'.")
    return features


# =============================================================================
# Per-frame builder
# =============================================================================

def _stack_per_hand(left: np.ndarray | None, right: np.ndarray | None,
                     placeholder: np.ndarray) -> np.ndarray:
    """Bimanual (2, ...) stack with per-hand placeholder fallback. Returns
    a fresh float32 array (no aliasing across frames)."""
    l = left if left is not None else placeholder
    r = right if right is not None else placeholder
    return np.stack([l.astype(np.float32), r.astype(np.float32)], axis=0)


def make_frame_builder(img_hw: tuple[int, int], *, include_depth: bool,
                        depth_encoding: str):
    """Closure over output config; returns a frame_builder for the writer.

    v2: assembles bimanual world-frame state + action from FrameContext's
    per-hand fields. Invalid hands get placeholder values so LeRobot's
    parquet writer never sees NaN.
    """

    pack_depth = (
        pack_depth_uint16_mm if depth_encoding == "uint16_mm"
        else pack_depth_uint8_cm
    )

    def frame_builder(ctx: FrameContext) -> dict:
        out: dict = {"observation.images.rgb": pack_rgb(ctx, img_hw)}
        if include_depth:
            depth_packed = pack_depth(ctx, img_hw)
            if depth_packed is not None:
                out["observation.depth"] = depth_packed

        # State: world-frame, bimanual stacked
        out["observation.state.wrist_pose"] = _stack_per_hand(
            ctx.left_wrist_pose_world, ctx.right_wrist_pose_world,
            _IDENTITY_WRIST_POSE,
        )
        out["observation.state.hand_keypoints"] = _stack_per_hand(
            ctx.left_hand_keypoints_world, ctx.right_hand_keypoints_world,
            _ZERO_JOINTS,
        )
        out["observation.state.wrist_valid"] = np.array(
            [
                1.0 if ctx.left_wrist_pose_world is not None else 0.0,
                1.0 if ctx.right_wrist_pose_world is not None else 0.0,
            ],
            dtype=np.float32,
        )
        out["observation.state.gripper"] = np.array(
            [
                ctx.left_gripper if ctx.left_gripper is not None else 0.0,
                ctx.right_gripper if ctx.right_gripper is not None else 0.0,
            ],
            dtype=np.float32,
        )

        # Confidence: passthrough per hand (kept as separate scalars for
        # backward compat with LeRobot consumers that mask by-hand)
        out["observation.left_confidence"] = np.array(
            [ctx.left_confidence], dtype=np.float32,
        )
        out["observation.right_confidence"] = np.array(
            [ctx.right_confidence], dtype=np.float32,
        )

        # Action: next-frame state target (absolute, hold-last at boundary,
        # pre-computed in iphone_writer via compute_action_arrays)
        out["action.wrist_pose"] = _stack_per_hand(
            ctx.left_action_wrist_pose_world, ctx.right_action_wrist_pose_world,
            _IDENTITY_WRIST_POSE,
        )
        out["action.gripper"] = np.array(
            [
                ctx.left_action_gripper if ctx.left_action_gripper is not None else 0.0,
                ctx.right_action_gripper if ctx.right_action_gripper is not None else 0.0,
            ],
            dtype=np.float32,
        )

        out["observation.T_world_cam"] = ctx.T_world_cam.astype(np.float32)
        return out

    return frame_builder


# =============================================================================
# Episode discovery
# =============================================================================

def discover_episodes(
    processed_root: Path, episodes: list[str] | None,
) -> list[Path]:
    """Find <sid>/<sid>.processed.npz under processed_root."""
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")
    paths: list[Path] = []
    for sub in sorted(processed_root.iterdir()):
        if not sub.is_dir():
            continue
        if episodes and sub.name not in episodes:
            continue
        npz = sub / f"{sub.name}.processed.npz"
        if npz.exists():
            paths.append(npz)
        else:
            print(f"  WARN: {sub.name} has no .processed.npz, skipping")
    return paths


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Build OPC source LeRobot v3 dataset (iPhone-line, embodiment-agnostic)")
    p.add_argument("--repo-id", required=True,
                   help="e.g., opc/iphone_source_v1 (LeRobot HF identifier; "
                        "stored as metadata, not used in path)")
    p.add_argument("--task", required=True,
                   help="Task description string written to every frame")
    p.add_argument("--batch", type=str, default=None,
                   help="Batch name (output goes to "
                        f"{_LINE_ROOT}/<batch>/{_STAGE}/). "
                        "Default: derived from --processed-root parent.")
    p.add_argument("--processed-root", type=Path, required=True,
                   help=f"Path to <_LINE_ROOT>/<batch>/02_processed/ "
                        f"(produced by 02_process.py)")
    p.add_argument("--r3d-root", type=Path, required=True,
                   help="Root containing <sid>.r3d files")
    p.add_argument("--output-dir", type=Path, default=None,
                   help=f"Override per-batch output. Default: "
                        f"{_LINE_ROOT}/<batch>/{_STAGE}/")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated sid list; default all")
    p.add_argument("--fps", type=int, default=None,
                   help=("Dataset fps label. Default: auto-derive from first "
                         "r3d's metadata (preserves source rate, no resample). "
                         "DexCap (60 Hz) and AgiBot (30 Hz) are both valid in "
                         "the LeRobot ecosystem; aloha_static is 50, "
                         "metaworld_mt50 is 80. Pass an int to override "
                         "(label only — frames are NOT decimated)."))
    p.add_argument("--img-h", type=int, default=480)
    p.add_argument("--img-w", type=int, default=640)
    p.add_argument("--depth-encoding", type=str, default="uint16_mm",
                   choices=["uint16_mm", "uint8_cm"],
                   help="uint16_mm (default, lossless, ~120x larger) or "
                        "uint8_cm (lossy, capped at 2.55m, video-compressed)")
    p.add_argument("--no-depth", action="store_true",
                   help="Disable depth feature entirely")
    p.add_argument("--orientation", type=str, default="auto",
                   choices=["auto", "landscape", "portrait"],
                   help=("Canvas orientation for r3d RGB/depth decoding. "
                         "MUST match what was passed to 01_hand_track for "
                         "this batch (K and T_world_cam in 02 npz are "
                         "already in that frame). Default 'auto' = legacy."))
    args = p.parse_args()

    sid_filter = (
        [e.strip() for e in args.episodes.split(",") if e.strip()]
        if args.episodes else None
    )
    npz_paths = discover_episodes(args.processed_root, sid_filter)
    if not npz_paths:
        print("No processed episodes found; nothing to do.")
        return 1

    batch = _resolve_batch(args)
    if args.output_dir is None:
        args.output_dir = _LINE_ROOT / batch / _STAGE

    img_hw = (args.img_h, args.img_w)
    include_depth = not args.no_depth
    features = make_features(
        img_hw, include_depth=include_depth, depth_encoding=args.depth_encoding,
    )
    frame_builder = make_frame_builder(
        img_hw, include_depth=include_depth, depth_encoding=args.depth_encoding,
    )

    def r3d_path_for(sid: str) -> Path:
        # Try flat layout first (output/iphone/00_record/<sid>.r3d), then
        # fall back to nested (output/iphone/00_record/<sid>/<sid>.r3d).
        flat = args.r3d_root / f"{sid}.r3d"
        if flat.exists():
            return flat
        return args.r3d_root / sid / f"{sid}.r3d"

    # Resolve fps. Default: read source rate from each r3d, ensure consistent,
    # use that value. Why not hardcode 30: LeRobot is fps-agnostic at the
    # storage layer (aloha=50, metaworld=80, droid=15) — relabeling 60-fps
    # captures as 30 is a half-speed-playback bug, not a normalization.
    # DexCap (closest peer for bare-hand MANO captures) ships at 60 Hz.
    fps_origin = "explicit --fps" if args.fps is not None else "auto from r3d"
    if args.fps is None:
        fps_per_episode = {}
        for npz_path in npz_paths:
            sid = npz_path.stem.replace(".processed", "")
            r3d = r3d_path_for(sid)
            if not r3d.exists():
                continue
            import zipfile, json as _json
            with zipfile.ZipFile(r3d, "r") as zf:
                meta = _json.loads(zf.read("metadata"))
            fps_per_episode[sid] = int(round(float(meta.get("fps", 30))))
        if not fps_per_episode:
            raise FileNotFoundError(
                "Cannot auto-derive fps: no .r3d files found. Pass --fps."
            )
        unique = set(fps_per_episode.values())
        first_sid = next(iter(fps_per_episode))
        resolved_fps = fps_per_episode[first_sid]
        if len(unique) > 1:
            print(
                f"  [fps] WARNING: episodes recorded at different fps: "
                f"{sorted(unique)}; using {resolved_fps} from first episode "
                f"({first_sid}). Pass --fps explicitly to override."
            )
        args.fps = resolved_fps

    print("Building source dataset (iPhone-line)")
    print(f"  episodes:        {len(npz_paths)}")
    print(f"  output_dir:      {args.output_dir}")
    print(f"  repo_id:         {args.repo_id}")
    print(f"  fps:             {args.fps} ({fps_origin})")
    print(f"  img_size:        {img_hw}")
    print(f"  include_depth:   {include_depth}")
    print(f"  depth_encoding:  {args.depth_encoding}")
    print(f"  orientation:     {args.orientation}")

    episodes = iter_episodes_from_r3d(
        npz_paths, r3d_path_for, include_depth=include_depth,
        orientation=args.orientation,
    )
    stats = build_lerobot_dataset(
        episodes=episodes,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        img_size=img_hw,
        task_description=args.task,
        features=features,
        frame_builder=frame_builder,
        robot_type="human_hand_iphone_source",
    )

    print()
    print("=" * 60)
    print("Done.")
    print(f"  episodes accepted: {stats['n_episodes_accepted']}")
    print(f"  total frames:      {stats['n_frames_total']}")
    print(f"  output:            {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
