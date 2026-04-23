"""
World-frame processing of cam-frame tracking data.

Pipeline position: Step 2/4 (processing layer)
Input:  Dual-hand cam-frame tracking pickle from 01_hand_track.py
        (coord_frame='camera'; wrist/rot/joints in cam frame; raw)
Output: Dual-hand episode-local world-frame tracking pickle consumed by 03
        (coord_frame='episode_local'; per-hand state/action; trimmed per hand)

Per-hand pipeline (hands processed independently — each has own trim window):
  [1] trim leading/trailing NaN   → records trim_slice (into original r3d range)
  [2] world transform             → p_world = R_wc @ p_cam + t_wc
                                    R_hand_world = R_wc @ R_hand_cam
  [3] center to robust anchor     → subtract median of first N valid wrists
                                    (3 axes); records center_offset_world
  [4] mark bad frames             → position jumps > max_pos_jump_m → NaN
  [5] quality check               → detection rate / max gap / duration
  [6] fill interior NaN           → linear + Slerp via PoseInterpolator
  [7] One-Euro filter             → pos (VectorOneEuro) + rot (slerp-OneEuro)
  [8] rotation jump warning       → log if frame-to-frame angle > threshold
  [9] gripper normalize           → (gw - min) / (max - min), clip [0, 1]
 [10] build state/action          → 7D absolute, action[t] = state[t+1]

Output schema (one dict per episode):
  {
      "timestamps":   (T_full,)      float64   # full (untrimmed) episode
      "T_world_cam":  (T_full, 4, 4) float64   # ARKit pose, untrimmed
      "K":            (3, 3)         float64
      "left_hand": {
          # Post-processing arrays, length T_trim_left:
          "state":              (T_trim, 7)   float32  # [x,y,z,rx,ry,rz,g]
          "action":             (T_trim, 7)   float32  # action[t]=state[t+1]
          "wrist_world":        (T_trim, 3)   float64
          "wrist_rot_world":    (T_trim, 3)   float64  # axis-angle
          "gripper_normalized": (T_trim,)     float32  # in [0, 1]
          "joints_cam":         (T_trim,21,3) float64  # passthrough cam frame
          "confidence":         (T_trim,)     float64
          # Metadata:
          "trim_slice":          (first, last)           # into full r3d range
          "center_offset_world": (3,)         float64    # episode origin in world
          "quality_passed":      bool
          "skip_reason":         str
      },
      "right_hand":   {same fields},
      "source":       str
      "episode_name": str
      "coord_frame":  "episode_local"                     # contract marker for 03
  }

Usage:
    python 02_process.py --input tracking.pkl --output processed.pkl
    python 02_process.py --input tracking.pkl --output processed.pkl --no-filter
"""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.interpolation import (
    fill_tracking_result,
    mark_bad_frames,
    max_consecutive_nans,
)
from utils.one_euro_filter import PoseOneEuroFilter


# =============================================================================
# 1. Config
# =============================================================================

@dataclass
class ProcessConfig:
    """Processing parameters.

    Attributes:
        fps: Capture rate (used for duration check).
        min_detect_rate: Minimum fraction of valid frames; below → reject.
        max_consecutive_gap: Longest interior NaN run allowed; above → reject.
        max_pos_jump_m: Per-frame position jump above this marks a bad frame.
        min_duration_s: Minimum valid-frame duration (s) for the hand to pass.
        filter_min_cutoff: One-Euro min cutoff (Hz). Lower = more smoothing.
        filter_beta: One-Euro speed coefficient.
        rot_jump_warn_deg: Log warning for frame-to-frame rotation > this.
        enable_filter: Toggle One-Euro smoothing.
        anchor_window_n: Number of leading valid frames used to compute the
            episode origin via per-axis median. Robust to single-frame LiDAR
            outliers at the trim boundary.
    """
    fps: int = 60
    min_detect_rate: float = 0.5
    max_consecutive_gap: int = 30
    max_pos_jump_m: float = 0.05
    min_duration_s: float = 2.0
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.007
    rot_jump_warn_deg: float = 45.0
    enable_filter: bool = True
    anchor_window_n: int = 10


# =============================================================================
# 2. Input validation
# =============================================================================

def _validate_input(tracking: dict, episode_name: str) -> None:
    """Enforce the 01→02 contract: cam-frame dual-hand tracking."""
    if tracking.get("coord_frame") != "camera":
        raise ValueError(
            f"{episode_name}: expected coord_frame='camera', got "
            f"{tracking.get('coord_frame')!r}. Run 01_hand_track.py first."
        )
    if "left_hand" not in tracking or "right_hand" not in tracking:
        raise ValueError(
            f"{episode_name}: expected dual-hand format (left_hand, "
            f"right_hand); got keys {sorted(tracking.keys())}"
        )
    for required in ("timestamps", "T_world_cam", "K"):
        if required not in tracking:
            raise ValueError(f"{episode_name}: missing top-level key {required!r}")


def _extract_hand_cam(tracking: dict, hand: str) -> dict:
    """Pull single-hand cam-frame dict from dual-hand input (deep copy)."""
    h = tracking[f"{hand}_hand"]
    return {
        "timestamps":    tracking["timestamps"].copy(),
        "T_world_cam":   tracking["T_world_cam"].copy(),
        "wrist_cam":     h["wrist_cam"].copy(),
        "wrist_rot_cam": h["wrist_rot_cam"].copy(),
        "joints_cam":    h["joints_cam"].copy(),
        "gripper_width": h["gripper_width"].copy(),
        "confidence":    h["confidence"].copy(),
    }


# =============================================================================
# 3. [1] Trim leading / trailing NaN
# =============================================================================

def trim_hand(hand_data: dict) -> tuple[dict, tuple[int, int]] | None:
    """Slice all per-frame arrays to the [first_valid, last_valid+1] range.

    Returns (trimmed_dict, (first, last)) or None if no valid frames exist.
    All per-frame arrays (including T_world_cam, timestamps, joints_cam,
    gripper_width, confidence) are sliced consistently so downstream steps
    can assume every frame index is a legitimate hand-in-frame sample.
    """
    valid = ~np.isnan(hand_data["wrist_cam"][:, 0])
    if not np.any(valid):
        return None
    idx = np.where(valid)[0]
    first, last = int(idx[0]), int(idx[-1] + 1)  # slice end is exclusive
    seg = slice(first, last)
    trimmed = {
        "timestamps":    hand_data["timestamps"][seg],
        "T_world_cam":   hand_data["T_world_cam"][seg],
        "wrist_cam":     hand_data["wrist_cam"][seg],
        "wrist_rot_cam": hand_data["wrist_rot_cam"][seg],
        "joints_cam":    hand_data["joints_cam"][seg],
        "gripper_width": hand_data["gripper_width"][seg],
        "confidence":    hand_data["confidence"][seg],
    }
    return trimmed, (first, last)


# =============================================================================
# 4. [2] World transform
# =============================================================================

def world_transform(hand_data: dict) -> dict:
    """Transform wrist position + rotation from camera frame to ARKit world.

    Math (per valid frame t):
        p_world = R_wc @ p_cam + t_wc
        R_hand_world = R_wc @ R_hand_cam      # stacking rotations
        wrist_rot_world = Rodrigues^-1(R_hand_world)

    joints_cam stays in cam frame (used for 2D observation overlay, not world
    coordinates). This keeps the output compact and decouples the joints from
    the episode-local centering applied later.

    Adds keys:
        eef_pos: (T, 3) wrist in world frame (NaN at no-detect frames)
        eef_rot: (T, 3) axis-angle in world frame (NaN at no-detect frames)
    """
    T_wc = hand_data["T_world_cam"]          # (T, 4, 4)
    wrist_cam = hand_data["wrist_cam"]       # (T, 3)
    rot_cam = hand_data["wrist_rot_cam"]     # (T, 3) axis-angle
    n = len(wrist_cam)

    R_wc = T_wc[:, :3, :3]
    t_wc = T_wc[:, :3, 3]

    valid = ~np.isnan(wrist_cam[:, 0])
    wrist_world = np.full_like(wrist_cam, np.nan, dtype=np.float64)
    rot_world = np.full_like(rot_cam, np.nan, dtype=np.float64)

    for t in range(n):
        if not valid[t]:
            continue
        wrist_world[t] = R_wc[t] @ wrist_cam[t] + t_wc[t]
        R_hand_cam = Rotation.from_rotvec(rot_cam[t]).as_matrix()
        R_hand_world = R_wc[t] @ R_hand_cam
        rot_world[t] = Rotation.from_matrix(R_hand_world).as_rotvec()

    return {
        **hand_data,
        # "eef_pos"/"eef_rot" keys match what utils/interpolation.py expects
        "eef_pos": wrist_world,
        "eef_rot": rot_world,
    }


# =============================================================================
# 5. [3] Center to robust anchor
# =============================================================================

def center_to_anchor(
    hand_data: dict, window_n: int = 10,
) -> tuple[dict, np.ndarray]:
    """Shift eef_pos so the episode anchor sits at 0 (all 3 axes).

    Anchor = per-axis median over the first `window_n` valid frames. Median
    is immune to single-frame LiDAR outliers at the trim boundary (up to
    window_n // 2 - 1 bad frames tolerated per axis).

    joints_cam is NOT translated — it lives in cam frame, not world.

    Returns:
        (hand_data_out, origin_world) — origin is the subtracted offset
        (float64, shape (3,)). Downstream consumers can reconstruct absolute
        world coordinates by adding origin_world back.
    """
    pos = hand_data["eef_pos"]
    valid = ~np.isnan(pos[:, 0])
    if not np.any(valid):
        return {**hand_data}, np.zeros(3, dtype=np.float64)

    # First `window_n` valid frames → per-axis median anchor.
    valid_idx = np.flatnonzero(valid)[:window_n]
    origin = np.nanmedian(pos[valid_idx], axis=0).astype(np.float64)
    new_pos = pos.copy()
    new_pos[valid] = pos[valid] - origin
    return {**hand_data, "eef_pos": new_pos}, origin


# =============================================================================
# 6. [5] Quality check
# =============================================================================

def quality_check(hand_data: dict, config: ProcessConfig) -> tuple[bool, str]:
    """Return (passed, reason)."""
    pos = hand_data["eef_pos"]
    n_total = len(pos)
    n_valid = int((~np.isnan(pos[:, 0])).sum())
    detect_rate = n_valid / n_total if n_total else 0.0

    if detect_rate < config.min_detect_rate:
        return False, f"detection rate {detect_rate:.0%} < {config.min_detect_rate:.0%}"
    max_gap = max_consecutive_nans(pos)
    if max_gap > config.max_consecutive_gap:
        return False, f"max gap {max_gap} > {config.max_consecutive_gap}"
    valid_duration = n_valid / config.fps
    if valid_duration < config.min_duration_s:
        return False, f"valid duration {valid_duration:.1f}s < {config.min_duration_s}s"
    return True, "ok"


# =============================================================================
# 7. [7] One-Euro filter
# =============================================================================

def apply_one_euro(hand_data: dict, config: ProcessConfig) -> dict:
    """Per-frame One-Euro filter on 6DoF world-frame pose.

    Input must be NaN-free (run after fill_tracking_result).
    """
    pos = hand_data["eef_pos"]
    rot = hand_data["eef_rot"]
    ts = hand_data["timestamps"]

    filt = PoseOneEuroFilter(
        min_cutoff=config.filter_min_cutoff,
        beta=config.filter_beta,
    )
    pos_out = np.empty_like(pos)
    rot_out = np.empty_like(rot)
    for t in range(len(pos)):
        p_f, r_f = filt.filter(pos[t], rot[t], timestamp=float(ts[t]))
        pos_out[t] = p_f
        rot_out[t] = r_f
    return {**hand_data, "eef_pos": pos_out, "eef_rot": rot_out}


# =============================================================================
# 8. [8] Rotation-jump diagnostic
# =============================================================================

def check_rotation_jumps(rot: np.ndarray, threshold_deg: float, label: str) -> int:
    """Log frame-to-frame rotation changes above threshold (no state mutation).

    dR = R[t+1] * R[t]^-1 ; angle = |axis-angle(dR)|
    """
    if len(rot) < 2:
        return 0
    R_t = Rotation.from_rotvec(rot)
    dR = R_t[1:] * R_t[:-1].inv()
    angles_deg = np.rad2deg(np.linalg.norm(dR.as_rotvec(), axis=1))
    n_jumps = int((angles_deg > threshold_deg).sum())
    if n_jumps > 0:
        print(f"  [{label}] {n_jumps} rotation jumps > {threshold_deg:.0f}°/frame "
              f"(max {angles_deg.max():.1f}°)")
    return n_jumps


# =============================================================================
# 9. [9] Gripper normalize
# =============================================================================

def normalize_gripper(gripper_width: np.ndarray) -> np.ndarray:
    """Map episode-local [min, max] → [0, 1]. Flat signal → zeros."""
    gw_min = float(np.nanmin(gripper_width))
    gw_max = float(np.nanmax(gripper_width))
    if gw_max > gw_min:
        norm = (gripper_width - gw_min) / (gw_max - gw_min)
        return np.clip(norm, 0.0, 1.0).astype(np.float32)
    return np.zeros_like(gripper_width, dtype=np.float32)


# =============================================================================
# 10. [10] State / action
# =============================================================================

def build_states_and_actions(
    pos: np.ndarray,
    rot: np.ndarray,
    gripper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute-pose state/action. action[t] = state[t+1]; tail holds pose.

    state[t] = [x, y, z, rx, ry, rz, gripper]  (7D, float32)
    """
    states = np.concatenate(
        [pos, rot, gripper.reshape(-1, 1)], axis=1,
    ).astype(np.float32)
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return states, actions


# =============================================================================
# 11. Per-hand pipeline
# =============================================================================

def process_hand(hand_data: dict, config: ProcessConfig, label: str) -> dict:
    """Run [1]–[10] on one hand. Returns the hand output dict.

    If the hand is all-NaN or fails quality, returns a minimal dict with
    quality_passed=False and a skip_reason (no trajectories inside).
    """
    # [1] Trim
    trimmed = trim_hand(hand_data)
    if trimmed is None:
        print(f"  [{label}] no valid frames — skipping hand")
        return {
            "quality_passed": False,
            "skip_reason":    "no valid frames",
            "trim_slice":     None,
        }
    data, trim_slice = trimmed
    n_trim = trim_slice[1] - trim_slice[0]
    print(f"  [{label}] trim {trim_slice[0]}:{trim_slice[1]} ({n_trim} frames)")

    # [2] World transform (adds eef_pos / eef_rot keys)
    data = world_transform(data)

    # [3] Center to robust anchor (median of first N valid frames)
    data, center_offset = center_to_anchor(data, window_n=config.anchor_window_n)

    # [4] Mark bad frames (position jumps → NaN)
    data = mark_bad_frames(data, max_pos_jump_m=config.max_pos_jump_m)

    # [5] Quality check (early reject)
    passed, reason = quality_check(data, config)
    if not passed:
        print(f"  [{label}] SKIP: {reason}")
        return {
            "quality_passed":      False,
            "skip_reason":         reason,
            "trim_slice":          trim_slice,
            "center_offset_world": center_offset,
        }

    # [6] Fill interior NaN
    data = fill_tracking_result(data, trim_boundary_nans=False)

    # [7] One-Euro filter
    if config.enable_filter:
        data = apply_one_euro(data, config)

    # [8] Rotation-jump diagnostic
    check_rotation_jumps(data["eef_rot"], config.rot_jump_warn_deg, label)

    # [9] Gripper normalize
    gripper_norm = normalize_gripper(data["gripper_width"])

    # [10] State / action
    states, actions = build_states_and_actions(
        data["eef_pos"], data["eef_rot"], gripper_norm,
    )

    return {
        "state":               states,
        "action":              actions,
        "wrist_world":         data["eef_pos"].astype(np.float64),
        "wrist_rot_world":     data["eef_rot"].astype(np.float64),
        "gripper_normalized":  gripper_norm,
        "joints_cam":          data["joints_cam"],
        "confidence":          data["confidence"],
        "trim_slice":          trim_slice,
        "center_offset_world": center_offset,
        "quality_passed":      True,
        "skip_reason":         "",
    }


# =============================================================================
# 12. Per-episode pipeline
# =============================================================================

def process_episode(tracking: dict, config: ProcessConfig) -> dict:
    """Apply processing pipeline to both hands of one episode."""
    ep_name = tracking.get("episode_name", "unknown")
    _validate_input(tracking, ep_name)
    print(f"\n=== {ep_name} ===")

    out = {
        "timestamps":   tracking["timestamps"].copy(),
        "T_world_cam":  tracking["T_world_cam"].copy(),
        "K":            tracking["K"].copy(),
        "source":       tracking.get("source", "unknown"),
        "episode_name": ep_name,
        "coord_frame":  "episode_local",
    }
    for hand in ("left", "right"):
        cam_dict = _extract_hand_cam(tracking, hand)
        out[f"{hand}_hand"] = process_hand(cam_dict, config, hand)
    return out


# =============================================================================
# 13. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process cam-frame tracking → world-frame episode-local dataset "
            "input. Hands are processed independently; each gets its own trim "
            "window and quality verdict."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 02_process.py --input tracking.pkl --output processed.pkl
  python 02_process.py --input tracking.pkl --output processed.pkl --no-filter
        """,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="cam-frame tracking pickle from 01_hand_track.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--min-detect-rate", type=float, default=0.5)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--max-pos-jump-m", type=float, default=0.05)
    parser.add_argument("--min-duration-s", type=float, default=2.0)
    parser.add_argument("--filter-min-cutoff", type=float, default=1.0)
    parser.add_argument("--filter-beta", type=float, default=0.007)
    parser.add_argument("--rot-jump-warn-deg", type=float, default=45.0)
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable One-Euro filter")
    args = parser.parse_args()

    config = ProcessConfig(
        fps=args.fps,
        min_detect_rate=args.min_detect_rate,
        max_consecutive_gap=args.max_gap,
        max_pos_jump_m=args.max_pos_jump_m,
        min_duration_s=args.min_duration_s,
        filter_min_cutoff=args.filter_min_cutoff,
        filter_beta=args.filter_beta,
        rot_jump_warn_deg=args.rot_jump_warn_deg,
        enable_filter=not args.no_filter,
    )

    print(f"Loading {args.input} ...")
    with open(args.input, "rb") as f:
        tracking_results = pickle.load(f)
    print(f"  {len(tracking_results)} episodes")

    processed = [process_episode(t, config) for t in tracking_results]

    # Per-hand pass / skip summary
    print("\n=== Summary ===")
    for hand in ("left", "right"):
        n_pass = sum(r[f"{hand}_hand"].get("quality_passed", False)
                     for r in processed)
        print(f"  {hand}: {n_pass}/{len(processed)} episodes passed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(processed, f)
    print(f"\nSaved {len(processed)} episodes to {args.output}")


if __name__ == "__main__":
    main()
