"""
Package processed tracking + Record3D RGB/depth into a LeRobot v3 dataset.

Pipeline position: Step 3/4 (packaging layer — pure format writer)
Input:
  - Dual-hand processed pickle from 02_process.py
      (coord_frame='episode_local'; per-hand state/action/wrist_world,
       per-hand trim_slice into original r3d frame range)
  - Original .r3d files (streamed for per-frame RGB + optional LiDAR depth)
Output:
  LeRobot v3 dataset (Parquet + MP4) — one selected hand (--hand left|right).
  Depth is included by default (core differentiator of this pipeline); disable
  with --no-depth.

This script does NO motion processing. Everything (world transform, centering,
filtering, quality gating, state/action) lives in 02_process.py. Here we
just:
  1. Validate input contract (coord_frame='episode_local')
  2. Select hand (--hand)
  3. Skip hands that failed 02's quality check
  4. Stream r3d frames via trim_slice, resize, add to LeRobotDataset
  5. Merge center_offset_world and friends into meta/episodes/*.parquet so
     downstream consumers can reconstruct ARKit-absolute world coords

Action representation (set in 02):
  state[t]  = [x, y, z, rx, ry, rz, gripper]  (7D, absolute, episode-local)
  action[t] = state[t+1]                      (tail holds last state)

Usage:
    python 03_build_dataset.py \\
        --processed processed.pkl \\
        --r3d-dir ./data/raw \\
        --output-dir ./data/dataset_v3 \\
        --repo-id <user>/demo_v3 \\
        --task "pick up cup" \\
        --hand right
"""

import argparse
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.r3d_reader import iter_r3d_frames


# =============================================================================
# 1. Config
# =============================================================================

@dataclass
class BuildConfig:
    """Packaging parameters.

    Attributes:
        fps: Target frame rate (must match the capture rate and 02).
        img_size: (H, W) for RGB/depth after resize.
        include_depth: Embed LiDAR depth as a second video stream.
    """
    fps: int = 60
    img_size: tuple[int, int] = (480, 640)
    include_depth: bool = True


# =============================================================================
# 2. Input validation
# =============================================================================

def _validate_processed(processed: dict, episode_name: str) -> None:
    """Enforce the 02→03 contract: episode-local dual-hand processed tracking."""
    if processed.get("coord_frame") != "episode_local":
        raise ValueError(
            f"{episode_name}: expected coord_frame='episode_local', got "
            f"{processed.get('coord_frame')!r}. Run 02_process.py first."
        )
    if "left_hand" not in processed or "right_hand" not in processed:
        raise ValueError(
            f"{episode_name}: expected dual-hand format (left_hand, "
            f"right_hand); got keys {sorted(processed.keys())}"
        )


# =============================================================================
# 3. Streaming RGB / depth reader
# =============================================================================

def _iter_r3d_resized(
    r3d_path: Path,
    target_hw: tuple[int, int],
    read_depth: bool,
    frame_indices: set[int],
):
    """Yield (rgb, depth) resized to target_hw for the requested indices."""
    h, w = target_hw
    for _idx, rgb, _ts, depth in iter_r3d_frames(
        r3d_path, read_depth=read_depth, frame_indices=frame_indices,
    ):
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        if depth is not None:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
            depth = depth[:, :, np.newaxis]
        yield rgb, depth


# =============================================================================
# 4. Feature schema
# =============================================================================

def make_features(img_hw: tuple[int, int], include_depth: bool) -> dict:
    h, w = img_hw
    features = {
        "observation.images.rgb": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["eef_x", "eef_y", "eef_z",
                               "rx", "ry", "rz", "gripper"]},
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": {"axes": ["eef_x", "eef_y", "eef_z",
                               "rx", "ry", "rz", "gripper"]},
        },
        "observation.joints_3d": {
            "dtype": "float32",
            "shape": (63,),
            "names": {"axes": [f"j{i}_{c}" for i in range(21)
                               for c in ("x", "y", "z")]},
        },
        "observation.confidence": {
            "dtype": "float32",
            "shape": (1,),
            "names": {"axes": ["detection_confidence"]},
        },
    }
    if include_depth:
        # LeRobot stores images as 3-channel uint8. Encode depth (meters) as
        # cm resolution for the 0–2.55 m range: depth_cm = depth_m * 100.
        # Recover meters at load time: depth_m = image[..., 0].astype(f32) / 100
        features["observation.images.depth"] = {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# =============================================================================
# 5. Custom episodes-parquet extension (LeRobot v3)
# =============================================================================

def _append_custom_episode_fields(
    output_dir: Path,
    idx_to_extras: dict[int, dict],
) -> None:
    """Merge extra columns into LeRobot v3's meta/episodes/**/*.parquet.

    v3 replaced the v2 single-line-JSON episodes.jsonl with chunked parquet
    files under meta/episodes/. We add columns like center_offset_world,
    trim_first_frame, trim_last_frame, source, episode_name — keyed by
    episode_index — so 04_replay_on_arm.py can reconstruct world coords.
    """
    episodes_dir = output_dir / "meta" / "episodes"
    if not episodes_dir.exists():
        print(
            f"  WARN: {episodes_dir} not found; cannot write custom episode fields"
        )
        return

    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        print(
            f"  WARN: no parquet files under {episodes_dir}; skipping custom fields"
        )
        return

    # Determine the union of extra columns so every chunk has the same schema.
    extra_cols: set[str] = set()
    for extras in idx_to_extras.values():
        extra_cols.update(extras.keys())

    for pq in parquet_files:
        df = pd.read_parquet(pq)
        for col in extra_cols:
            # Default column value is None; filled per-row if index matches.
            df[col] = [
                idx_to_extras.get(int(idx), {}).get(col)
                for idx in df["episode_index"]
            ]
        df.to_parquet(pq, index=False)


# =============================================================================
# 6. Main builder
# =============================================================================

def build_dataset(
    processed_results: list[dict],
    r3d_dir: Path,
    output_dir: Path,
    repo_id: str,
    task_description: str,
    hand: str,
    config: BuildConfig,
) -> dict:
    """Write one-hand LeRobot v2 dataset. Returns a stats dict."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = make_features(config.img_size, include_depth=config.include_depth)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.fps,
        features=features,
        root=output_dir,
        robot_type="hand_6dof",
        use_videos=True,
        image_writer_threads=2,
    )

    n_in = len(processed_results)
    n_out = 0
    n_frames_total = 0
    skipped: list[tuple[str, str]] = []
    ep_extras: dict[int, dict] = {}

    for processed in processed_results:
        ep_name = processed.get("episode_name", "unknown")
        _validate_processed(processed, ep_name)

        hand_data = processed[f"{hand}_hand"]
        if not hand_data.get("quality_passed", False):
            reason = hand_data.get("skip_reason", "unknown")
            print(f"  SKIP {ep_name} [{hand}]: {reason}")
            skipped.append((ep_name, reason))
            continue

        states = hand_data["state"]
        actions = hand_data["action"]
        joints_cam = hand_data["joints_cam"]        # (T, 21, 3)
        confidence = hand_data["confidence"]
        trim_slice = hand_data["trim_slice"]
        center_offset = hand_data["center_offset_world"]

        n_frames = len(states)
        first, last = trim_slice
        expected = last - first
        if n_frames != expected:
            raise ValueError(
                f"{ep_name}: state length {n_frames} != trim_slice "
                f"span {expected}. 02 output is inconsistent."
            )

        # Stream the exact r3d window that this hand was trimmed to
        r3d_path = r3d_dir / f"{ep_name}.r3d"
        if not r3d_path.exists():
            print(f"  SKIP {ep_name}: .r3d not found at {r3d_path}")
            skipped.append((ep_name, "r3d not found"))
            continue

        frame_indices = set(range(first, last))
        joints_flat = joints_cam.reshape(-1, 63).astype(np.float32)
        confidence_arr = confidence.reshape(-1, 1).astype(np.float32)

        t = 0
        for rgb, depth in _iter_r3d_resized(
            r3d_path, config.img_size, config.include_depth, frame_indices,
        ):
            if t >= n_frames:
                break
            frame_dict = {
                "observation.images.rgb": rgb,
                "observation.state":      states[t],
                "action":                 actions[t],
                "observation.joints_3d":  joints_flat[t],
                "observation.confidence": confidence_arr[t],
                "task":                   task_description,
            }
            if depth is not None:
                # LiDAR can emit NaN/inf at edges; zero-fill before uint8 cast
                # to silence "invalid value in cast" and keep missing-depth
                # pixels as "0 cm" (matches the black-edge convention).
                depth_m = np.nan_to_num(
                    depth[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0,
                )
                depth_cm = (depth_m * 100).clip(0, 255).astype(np.uint8)
                frame_dict["observation.images.depth"] = np.stack([depth_cm] * 3, axis=-1)
            dataset.add_frame(frame_dict)
            t += 1

        if t != n_frames:
            print(f"  WARN {ep_name}: expected {n_frames} frames, got {t} from r3d")

        # Capture per-episode extras keyed by episode_index (ordinal, 0-based)
        ep_idx = n_out  # dataset assigns indices sequentially
        ep_extras[ep_idx] = {
            "episode_name":        ep_name,
            "center_offset_world": [float(x) for x in center_offset],
            "trim_first_frame":    int(first),
            "trim_last_frame":     int(last),
            "source":              processed.get("source", "unknown"),
        }

        dataset.save_episode()
        n_out += 1
        n_frames_total += n_frames
        offset_str = np.array2string(center_offset, precision=3, suppress_small=True)
        print(f"  SAVED {ep_name}: {n_frames} frames, "
              f"center_offset_world={offset_str}")

    dataset.finalize()

    # Write center_offset_world and friends into meta/episodes.jsonl
    _append_custom_episode_fields(output_dir, ep_extras)

    stats = {
        "n_episodes_in":      n_in,
        "n_episodes_out":     n_out,
        "n_episodes_skipped": n_in - n_out,
        "n_frames_total":     n_frames_total,
        "skipped":            skipped,
    }
    print("\nDataset generation complete:")
    print(f"  Episodes: {n_out}/{n_in} ({n_in - n_out} skipped)")
    print(f"  Total frames: {n_frames_total}")
    print(f"  Output: {output_dir}")
    return stats


# =============================================================================
# 7. Optional sim-check (MuJoCo workspace feasibility gate)
# =============================================================================

def run_sim_check(
    output_dir: Path,
    n_episodes: int,
    scale: float,
    timeout_s: int = 300,
) -> list[tuple[int, str, int]]:
    """Replay the first N episodes in headless MuJoCo to flag IK / workspace
    issues before the user spends real-arm time.

    Subprocess-calls scripts/05_replay_in_sim.py — keeps 03 decoupled from the
    IK / retarget / sim stack.  Each episode is graded by counting the
    `[mujoco] WARN` lines that `sim/mujoco_loader.py` emits when a joint
    command exceeds the MJCF jnt_range (i.e. geometrically infeasible).

    Returns: [(episode_idx, status, warn_count), ...]
      status ∈ {"PASS", "WARN (n)", "FAIL"}.
    """
    script = Path(__file__).resolve().parent / "05_replay_in_sim.py"
    results: list[tuple[int, str, int]] = []

    print(f"\n=== Sim-check: {n_episodes} episode(s), scale={scale} ===")
    for ep in range(n_episodes):
        print(f"\n  [sim-check] Episode {ep} ...")
        proc = subprocess.run(
            [
                sys.executable, str(script),
                "--dataset-root", str(output_dir),
                "--episode", str(ep),
                "--scale", str(scale),
                "--speed", "20.0",
                "--no-gui",
            ],
            capture_output=True, text=True, timeout=timeout_s,
        )
        warn_count = proc.stdout.count("[mujoco] WARN")
        if proc.returncode != 0:
            status = "FAIL"
            # Tail of stderr helps the user diagnose without re-running.
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
            print(f"    -> FAIL (rc={proc.returncode}). stderr tail:\n{tail}")
        elif warn_count > 0:
            status = f"WARN ({warn_count})"
            print(f"    -> {warn_count} out-of-range joint commands")
        else:
            status = "PASS"
            print("    -> PASS")
        results.append((ep, status, warn_count))

    print("\n=== Sim-check summary ===")
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    for ep, status, _ in results:
        print(f"  Episode {ep}: {status}")
    print(f"  {n_pass}/{len(results)} clean")
    if n_pass < len(results):
        print(
            "  NOTE: sim-check is a pre-real-arm filter, not a blocker. "
            "Investigate WARN/FAIL before running 04_replay_on_arm."
        )
    return results


# =============================================================================
# 8. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Package 02_process.py output into a LeRobot v2 dataset "
            "(one hand per dataset)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python 03_build_dataset.py \\
      --processed processed.pkl --r3d-dir ./data/raw \\
      --output-dir ./data/dataset_v3 --repo-id <user>/demo_v3 \\
      --task "pick up cup" --hand right
        """,
    )
    parser.add_argument("--processed", type=Path, required=True,
                        help="processed dual-hand pickle from 02_process.py")
    parser.add_argument("--r3d-dir", type=Path, required=True,
                        help="Directory containing the original .r3d files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--hand", type=str, required=True,
                        choices=["left", "right"],
                        help="Which hand to package into the dataset")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--img-height", type=int, default=480)
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable LiDAR depth stream (default: enabled)")
    parser.add_argument("--sim-check", action="store_true",
                        help="After build, replay first N episodes in headless "
                             "MuJoCo to flag workspace/IK issues. Requires "
                             "mujoco installed; default: off.")
    parser.add_argument("--sim-check-episodes", type=int, default=3,
                        help="How many episodes to sim-check (default: 3)")
    parser.add_argument("--sim-check-scale", type=float, default=0.5,
                        help="IK workspace scale for sim-check. HaMeR v3 "
                             "datasets need 0.5; default: 0.5")
    args = parser.parse_args()

    config = BuildConfig(
        fps=args.fps,
        img_size=(args.img_height, args.img_width),
        include_depth=not args.no_depth,
    )

    print(f"Loading {args.processed} ...")
    with open(args.processed, "rb") as f:
        processed_results = pickle.load(f)
    print(f"  {len(processed_results)} episodes")

    print("\nBuilding dataset...")
    stats = build_dataset(
        processed_results=processed_results,
        r3d_dir=args.r3d_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task_description=args.task,
        hand=args.hand,
        config=config,
    )

    if args.sim_check:
        # Only run if we actually produced episodes — sim-check on an empty
        # dataset would just subprocess-error and confuse the user.
        n_out = stats.get("n_episodes_out", 0)
        if n_out == 0:
            print("\n[sim-check] Skipped: no episodes were written.")
        else:
            n = min(args.sim_check_episodes, n_out)
            run_sim_check(
                output_dir=args.output_dir,
                n_episodes=n,
                scale=args.sim_check_scale,
            )


if __name__ == "__main__":
    main()
