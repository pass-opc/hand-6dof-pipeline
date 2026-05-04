"""
CLI: `.qpos.npz` → MuJoCo viewer / mp4 / real arm.

Pipeline position: stage 6 of either recording line. Reads
`.qpos.npz` + `.qpos.meta.json` from `python -m retarget`,
dispatches to the registered `(robot, env)` backend.

Layout convention (line-agnostic):
    INPUT  : <_LINE_ROOT>/<batch>/05_qpos_<robot>/<sid>/<sid>.qpos.npz
                                                  + <sid>.qpos.meta.json
    OUTPUT : <_LINE_ROOT>/<batch>/06_replay_<robot>/<sid>/<sid>.replay.mp4
             (only when --output mp4)

Robot + hand are read from the meta file — the CLI doesn't take them
again. If the dataset's qpos shape doesn't match the registered
backend's expectations, the backend errors out clearly and points at
retarget.

Usage:
    conda activate lerobot                  # mujoco lives here
    cd code/opc_data_pipeline

    # Live viewer (single episode)
    python -m replay --qpos-root output/gemini335/<batch>/05_qpos_shadow \\
        --episodes <sid>

    # Headless mp4 (all episodes in batch)
    python -m replay --qpos-root output/gemini335/<batch>/05_qpos_shadow \\
        --output mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from replay import get_backend  # noqa: E402


def discover_qpos_episodes(
    qpos_root: Path, episodes: list[str] | None,
) -> list[Path]:
    """Find <sid>/<sid>.qpos.npz under qpos_root."""
    if not qpos_root.exists():
        raise FileNotFoundError(f"qpos_root not found: {qpos_root}")
    paths: list[Path] = []
    for sub in sorted(qpos_root.iterdir()):
        if not sub.is_dir():
            continue
        if episodes and sub.name not in episodes:
            continue
        npz = sub / f"{sub.name}.qpos.npz"
        if npz.exists():
            paths.append(npz)
        else:
            print(f"  WARN: {sub.name} has no .qpos.npz, skipping")
    return paths


def _resolve_replay_root(qpos_root: Path) -> Path:
    """Convention: 05_qpos_<robot>/ → 06_replay_<robot>/."""
    name = qpos_root.name
    if name.startswith("05_qpos_"):
        suffix = name[len("05_qpos_"):]
        return qpos_root.parent / f"06_replay_{suffix}"
    # Fallback — caller can pass --output-root explicitly if they
    # use a non-standard layout.
    return qpos_root.parent / f"06_replay_{name}"


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python -m replay",
        description="Replay .qpos.npz in MuJoCo (viewer / mp4) or on a "
                    "real arm. Pure replay — no IK / smoothing / "
                    "frame-shape changes. Backend resolved from "
                    ".qpos.meta.json's robot+env fields.",
    )
    p.add_argument("--qpos-root", type=Path, required=True,
                   help="Path to <line>/<batch>/05_qpos_<robot>/")
    p.add_argument("--episodes", type=str, default=None,
                   help="Comma-separated sid list; default all")
    p.add_argument("--output", choices=("viewer", "mp4", "real", "rerun", "rrd"),
                   default="viewer",
                   help="viewer = live MuJoCo window (default); "
                        "mp4 = headless offscreen render to file; "
                        "real = drive the physical arm (SO-101); "
                        "rerun = spawn rerun.io viewer with AR overlay "
                        "(hand on source RGB via pinhole projection); "
                        "rrd = save .rrd file (open later with `rerun <path>`).")
    p.add_argument("--output-root", type=Path, default=None,
                   help="(mp4) destination dir. Default: "
                        "<line>/<batch>/06_replay_<robot>/")
    p.add_argument("--camera", default="cam_frame",
                   help="(mp4 mode) MuJoCo camera name. dex backends "
                        "accept cam_frame (default; replicates the "
                        "Gemini 335 recording view, FOV from K_flat) or "
                        "hand_follow (palm tracker, easier to inspect).")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--fps", type=int, default=None,
                   help=("Playback / mp4 fps. Default: auto-derive from "
                         "first qpos.npz timestamps_us (preserves source "
                         "rate; iPhone Record3D = 60). Pass an int to "
                         "override (label only — qpos arrays are NOT "
                         "decimated)."))
    p.add_argument("--no-loop", action="store_true",
                   help="(viewer) play once instead of looping")
    # Real-arm flags. Ignored unless --output real.
    p.add_argument("--port", type=str, default=None,
                   help="(real) serial port, e.g. COM5 or /dev/ttyUSB0")
    p.add_argument("--arm-id", type=str, default="pass_follower_arm",
                   help="(real) calibration name under "
                        "~/.cache/huggingface/lerobot/calibration/")
    p.add_argument("--speed", type=float, default=0.3,
                   help="(real) playback speed multiplier (0.3 = 30%% — safer)")
    p.add_argument("--max-relative-target", type=float, default=10.0,
                   help="(real) max degrees per servo command step")
    p.add_argument("--dry-run", action="store_true",
                   help="(real) load + decode trajectory but skip the "
                        "serial connect — CI-friendly smoke test")
    # Rerun-specific flags. Ignored unless --output rerun / rrd.
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="(rerun/rrd) JPEG quality for source RGB log "
                        "(1-100). 85 ≈ visually lossless QA, 60 ≈ small. "
                        "Lower = smaller .rrd file.")
    p.add_argument("--image-plane-distance", type=float, default=2.0,
                   help="(rerun/rrd) Depth (m) at which the source image "
                        "renders in the rerun 3D Spatial view. Set > max "
                        "hand depth so hand mesh floats in front of the "
                        "background plate. Default 2.0 m (hand ~0.4 m). "
                        "The 2D Spatial view AR composite is unaffected.")
    p.add_argument("--source-mp4-stage",
                   choices=("auto", "02_processed", "03_optimized"),
                   default="auto",
                   help="(rerun/rrd) Which stage's preview mp4 to use as "
                        "the AR background plate. 'auto' (default) picks "
                        "the same stage as the qpos meta's source_npz — so "
                        "an optimize-fed retarget gets 03_optimized's "
                        "smoothed overlay, a raw-fed retarget gets "
                        "02_processed's raw overlay. 01_tracking is NOT a "
                        "valid choice (it carries pre-trim raw HaMeR which "
                        "doesn't match what retarget consumed).")
    p.add_argument("--source-mp4", type=Path, default=None,
                   help="(rerun/rrd) Full override path to the AR backplate "
                        "mp4. Takes precedence over --source-mp4-stage. Use "
                        "this to compare across stages (e.g. force "
                        "01_tracking) without touching the auto convention.")
    p.add_argument("--cam-zoom", type=float, default=1.0,
                   help="(mp4/rerun/rrd) Visual zoom multiplier applied at "
                        "render/log time only — does NOT modify qpos meta or "
                        "source mp4. < 1.0 pushes camera back along view "
                        "axis (robot covers fewer pixels); > 1.0 pulls in. "
                        "Default 1.0 = strict recording-anchor view "
                        "(retarget already medians the pinch anchor so "
                        "baseline is consistent across episodes).")
    args = p.parse_args()

    qpos_root: Path = args.qpos_root
    output_root = args.output_root or _resolve_replay_root(qpos_root)

    sid_filter = (
        [e.strip() for e in args.episodes.split(",") if e.strip()]
        if args.episodes else None
    )
    npz_paths = discover_qpos_episodes(qpos_root, sid_filter)
    if not npz_paths:
        print("No qpos episodes found; nothing to do.")
        return 1

    # Auto-derive fps from first qpos.npz timestamps_us. Same policy as 03:
    # LeRobot is fps-agnostic; re-labeling 60 Hz captures as 30 was a
    # half-speed bug. timestamps_us is monotonic per the trim window from 02.
    fps_origin = "explicit --fps" if args.fps is not None else "auto from qpos timestamps"
    if args.fps is None:
        import numpy as _np
        _ts = _np.load(npz_paths[0])["timestamps_us"]
        if len(_ts) >= 2:
            _dt = _np.median(_np.diff(_ts.astype(_np.int64)))
            args.fps = int(round(1e6 / max(_dt, 1)))
        else:
            args.fps = 30

    print(f"replay  ({args.output} mode)")
    print(f"  qpos_root:    {qpos_root}")
    print(f"  episodes:     {len(npz_paths)}")
    if args.output == "mp4":
        print(f"  output_root:  {output_root}")
        print(f"  resolution:   {args.width}x{args.height} @ {args.fps}fps ({fps_origin})")
        print(f"  camera:       {args.camera}")
    print()

    for npz_path in npz_paths:
        sid = npz_path.stem.replace(".qpos", "")
        meta_path = npz_path.parent / f"{sid}.qpos.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"meta sidecar missing: {meta_path}. Re-run "
                f"`python -m retarget` to regenerate."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        robot = meta["robot"]
        # `--output real` always dispatches to the (robot, "real") backend
        # regardless of meta.env (retarget bakes a default env hint, but
        # the user gets the final say at replay time).
        env = "real" if args.output == "real" else meta.get("env", "mujoco")
        run = get_backend(robot, env)

        if args.output == "viewer":
            print(f"  [{sid}] launching viewer ({robot}/{env})...")
            run(
                qpos_npz_path=npz_path,
                qpos_meta_path=meta_path,
                output="viewer",
                fps=args.fps,
                width=args.width, height=args.height,
                camera=args.camera,
                loop=not args.no_loop,
            )
        elif args.output == "mp4":
            out_dir = output_root / sid
            out_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = out_dir / f"{sid}.replay.mp4"
            print(f"  [{sid}] rendering ({robot}/{env}) → {out_mp4.name}")
            stats = run(
                qpos_npz_path=npz_path,
                qpos_meta_path=meta_path,
                output="mp4", out_mp4=out_mp4,
                fps=args.fps,
                width=args.width, height=args.height,
                camera=args.camera,
                cam_zoom=args.cam_zoom,
            )
            if stats:
                print(f"    rendered={stats['n_rendered']}/{stats['n_total']}  "
                      f"held={stats['n_held']}  "
                      f"skipped(leading)={stats['n_skipped_leading_invalid']}")
            print(f"    → {out_mp4}")
        elif args.output in ("rerun", "rrd"):
            # source npz from meta (retarget records the stage it consumed).
            # The AR backplate mp4 must come from the SAME stage so the
            # 2D overlay matches the keypoints that drove retarget.
            batch_root = qpos_root.resolve().parent  # <line>/<batch>/
            src_npz_str = meta.get("source_npz")
            if not src_npz_str:
                raise KeyError(
                    f"{sid}: meta missing 'source_npz' — re-run retarget."
                )
            source_npz = Path(src_npz_str)
            if not source_npz.is_absolute():
                source_npz = (
                    Path(__file__).resolve().parents[1] / source_npz
                )

            # Resolve backplate mp4. Precedence: explicit --source-mp4 →
            # explicit --source-mp4-stage → auto (= same stage as
            # source_npz). Never silently fall through to a stage the
            # user didn't ask for: a wrong stage's overlay defeats the
            # purpose of the QA rrd.
            if args.source_mp4 is not None:
                source_mp4 = args.source_mp4.resolve()
                if not source_mp4.exists():
                    raise FileNotFoundError(
                        f"--source-mp4 not found: {source_mp4}"
                    )
            else:
                if args.source_mp4_stage == "auto":
                    # `<batch>/<stage>/<sid>/...` — stage is the parent
                    # directory of the sid dir.
                    chosen_stage = source_npz.parent.parent.name
                else:
                    chosen_stage = args.source_mp4_stage
                source_mp4 = (
                    batch_root / chosen_stage / sid / f"{sid}_preview.mp4"
                )
                if not source_mp4.exists():
                    raise FileNotFoundError(
                        f"{sid}: no preview at {source_mp4}.\n"
                        f"  Either run the {chosen_stage} stage with "
                        f"preview enabled, pick a different stage with "
                        f"--source-mp4-stage, or pass --source-mp4 PATH "
                        f"to override."
                    )
            out_rrd = None
            if args.output == "rrd":
                out_dir = output_root / sid
                out_dir.mkdir(parents=True, exist_ok=True)
                out_rrd = out_dir / f"{sid}.rerun.rrd"
                print(f"  [{sid}] writing rerun .rrd → {out_rrd.name}")
            else:
                print(f"  [{sid}] spawning rerun viewer for {robot}...")
            stats = run(
                qpos_npz_path=npz_path,
                qpos_meta_path=meta_path,
                output=args.output,
                source_mp4=source_mp4,
                source_npz=source_npz,
                out_rrd=out_rrd,
                jpeg_quality=args.jpeg_quality,
                image_plane_dist_m=args.image_plane_distance,
                cam_zoom=args.cam_zoom,
            )
            if stats:
                print(f"    logged={stats['n_logged']}/{stats['n_total']}  "
                      f"sink={stats['sink']}")
        else:  # "real"
            print(f"  [{sid}] driving real {robot} on port={args.port or '<dry-run>'}")
            stats = run(
                qpos_npz_path=npz_path,
                qpos_meta_path=meta_path,
                output="real",
                port=args.port,
                arm_id=args.arm_id,
                speed=args.speed,
                max_relative_target=args.max_relative_target,
                dry_run=args.dry_run,
            )
            if stats:
                executed = "executed" if stats.get("executed") else "dry-run only"
                print(f"    {executed}: {stats['n_total']} frames")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
