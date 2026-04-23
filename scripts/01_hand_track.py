"""
Bare-hand detection from Record3D captures — raw cam-frame output.

Pipeline position: Step 1/4 (perception layer)
Input:  .r3d files (Record3D iPhone captures, ARKit VIO + LiDAR)
Output: Dual-hand cam-frame tracking pickle — wrist + joints in camera frame,
        LiDAR-corrected metric scale. No world transform, no filtering,
        no trimming. ARKit T_world_cam is passed through for 02 to apply.

This layer does "see the hand + know its metric position in camera frame".
World-frame transform, trimming, filtering, and quality analysis all happen
in 02_process.py.

Per-frame pipeline:
  1. Read RGB + LiDAR depth + T_world_cam[t] (IO / ARKit passthrough)
  2. HaMeR detect → wrist_cam + joints_cam + bbox (cam frame, virtual scale)
  3. Spatial tracker corrects MediaPipe handedness flips
  4. LiDAR depth correction: fix wrist_cam Z via real K (bbox-center sample);
     scale joints proportionally so hand shape matches corrected metric scale.
     Result is metric-correct cam-frame geometry.

Output schema (one dict per episode):
  {
      "timestamps":   (T,)         float64  # seconds
      "T_world_cam":  (T, 4, 4)    float64  # ARKit pose (portrait-adjusted)
      "K":            (3, 3)       float64  # iPhone intrinsics (portrait)
      "left_hand": {
          "wrist_cam":      (T, 3)    float64  # cam frame
          "wrist_rot_cam":  (T, 3)    float64  # axis-angle, cam frame
          "joints_cam":     (T,21,3)  float64  # cam frame
          "bbox":           (T, 4)    float64  # (x1, y1, x2, y2) pixels
          "gripper_width":  (T,)      float64  # meters (thumb↔index tip)
          "confidence":     (T,)      float64
      },
      "right_hand":   {same fields},
      "source":       str                      # tracker backend name
      "episode_name": str
      "coord_frame":  "camera"                 # contract marker for 02
  }

Usage:
    python 01_hand_track.py --r3d-dir ./data/raw --output tracking.pkl
    python 01_hand_track.py --r3d-dir ./data/raw --output tracking.pkl --no-depth
"""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.hand_tracker import HandDetection, HandTracker, create_tracker
from utils.depth_correction import back_project_depth, print_depth_correction_summary
from utils.r3d_reader import (
    iter_r3d_frames,
    read_r3d_metadata,
    read_iphone_intrinsics,
    read_poses,
)
from utils.spatial_tracker import SpatialHandTracker


# =============================================================================
# 1. Config
# =============================================================================

@dataclass
class HandTrackConfig:
    """Configuration for hand perception pipeline."""
    backend: str = "hamer"
    read_depth: bool = True  # LiDAR metric-scale correction


# =============================================================================
# 2. Per-hand data container
# =============================================================================

def _make_empty_hand_arrays(n_frames: int) -> dict:
    """NaN-filled arrays for one hand. Layout matches the output schema."""
    return {
        "wrist_cam":     np.full((n_frames, 3), np.nan, dtype=np.float64),
        "wrist_rot_cam": np.full((n_frames, 3), np.nan, dtype=np.float64),
        "joints_cam":    np.full((n_frames, 21, 3), np.nan, dtype=np.float64),
        "bbox":          np.full((n_frames, 4), np.nan, dtype=np.float64),
        "gripper_width": np.full(n_frames, np.nan, dtype=np.float64),
        "confidence":    np.zeros(n_frames, dtype=np.float64),
    }


# =============================================================================
# 3. Core: process one episode
# =============================================================================

def process_episode(
    r3d_path: Path,
    config: HandTrackConfig,
    tracker: HandTracker,
) -> dict:
    """Process one .r3d episode into dual-hand cam-frame tracking.

    Output arrays are full-length (same as r3d frame count), with NaN at
    frames where no detection occurred. 02_process.py consumes this.
    """
    metadata, jpg_names = read_r3d_metadata(r3d_path)
    n_frames = len(jpg_names)

    T_world_cam = read_poses(r3d_path)
    if len(T_world_cam) != n_frames:
        raise ValueError(
            f"{r3d_path.name}: poses count {len(T_world_cam)} != "
            f"jpg count {n_frames}"
        )

    # iPhone K — used for both depth back-projection AND HaMeR's
    # cam_crop_to_full, so pred_cam_t is metric-correct from the start
    # (not a ~3x overestimate that LiDAR must rescue by shrinking the hand).
    K_real = read_iphone_intrinsics(metadata)
    tracker.set_focal_length_px(float(K_real[0, 0]))

    left = _make_empty_hand_arrays(n_frames)
    right = _make_empty_hand_arrays(n_frames)
    timestamps = np.zeros(n_frames, dtype=np.float64)

    spatial_tracker = SpatialHandTracker()

    depth_stats_left: list[dict] = []
    depth_stats_right: list[dict] = []

    for i, rgb, ts, depth in tqdm(
        iter_r3d_frames(r3d_path, read_depth=config.read_depth),
        total=n_frames,
        desc=f"Tracking {r3d_path.name}",
    ):
        timestamps[i] = ts

        detections = tracker.detect(rgb)
        detections = spatial_tracker.update(detections)

        for det in detections:
            is_left = det.handedness == "left"
            hand_data = left if is_left else right
            stats_list = depth_stats_left if is_left else depth_stats_right

            pos_cam = det.wrist_pos.copy()
            rot_cam = det.wrist_rot.copy()
            joints_cam = det.joints_3d.copy()

            # LiDAR residual correction (cam frame):
            #   With real focal passed to HaMeR, pred_cam_t is already near-
            #   metric; LiDAR only does a small wrist translation correction
            #   (betas/pose error, ~5-10% scale). Joint offsets from wrist stay
            #   at HaMeR's predicted MANO shape — no 3x shrinkage.
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
                if bp_pos is not None:
                    # Translate the whole joint cloud by (bp_pos - wrist_hamer)
                    # so joints[0] lands on LiDAR depth and joints[1..20] keep
                    # their real MANO offsets from the wrist.
                    delta = bp_pos - pos_cam
                    joints_cam = joints_cam + delta
                    pos_cam = bp_pos

            hand_data["wrist_cam"][i] = pos_cam
            hand_data["wrist_rot_cam"][i] = rot_cam
            hand_data["joints_cam"][i] = joints_cam
            if det.bbox is not None:
                hand_data["bbox"][i] = det.bbox
            hand_data["gripper_width"][i] = float(
                np.linalg.norm(joints_cam[4] - joints_cam[8])
            )
            hand_data["confidence"][i] = det.confidence

    # Detection summary
    for hand_name, hand_data in [("left", left), ("right", right)]:
        detected = ~np.isnan(hand_data["wrist_cam"][:, 0])
        rate = detected.sum() / n_frames if n_frames > 0 else 0
        print(f"  {hand_name} hand: {detected.sum()}/{n_frames} frames ({rate:.1%})")

    # Depth-correction stats
    if config.read_depth:
        for hand_name, stats_list in [("left", depth_stats_left),
                                       ("right", depth_stats_right)]:
            if stats_list:
                print(f"\n  [{hand_name} hand depth correction]")
                print_depth_correction_summary(stats_list)

    return {
        "timestamps":   timestamps,
        "T_world_cam":  T_world_cam,
        "K":            K_real,
        "left_hand":    left,
        "right_hand":   right,
        "source":       tracker.get_backend_name(),
        "episode_name": r3d_path.stem,
        "coord_frame":  "camera",
    }


# =============================================================================
# 4. CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect bare hands in Record3D captures — cam-frame output with "
            "LiDAR-corrected metric scale. ARKit pose passed through for 02."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 01_hand_track.py --r3d-dir ./data/raw --output tracking.pkl
  python 01_hand_track.py --r3d-dir ./data/raw --output tracking.pkl --no-depth
        """,
    )
    parser.add_argument("--r3d-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", type=str, default="hamer",
                        choices=["hamer", "wilor"])
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable LiDAR metric-scale correction")

    args = parser.parse_args()

    config = HandTrackConfig(
        backend=args.backend,
        read_depth=not args.no_depth,
    )

    print(f"Backend: {config.backend}")
    tracker = create_tracker(config.backend)
    print(f"Tracker: {tracker.get_backend_name()}")

    r3d_files = sorted(args.r3d_dir.glob("*.r3d"))
    if not r3d_files:
        raise FileNotFoundError(f"No .r3d files found in {args.r3d_dir}")
    print(f"Found {len(r3d_files)} episodes")

    all_results = []
    for r3d_path in r3d_files:
        result = process_episode(r3d_path, config, tracker)
        all_results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved {len(all_results)} episodes to {args.output}")


if __name__ == "__main__":
    main()
