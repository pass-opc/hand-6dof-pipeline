#!/usr/bin/env python3
"""
Replay recorded EE trajectories on a robot arm via retarget + IK.

Pipeline position: Step 4/4 (replay layer)
Consumes a LeRobot dataset produced by 03_build_dataset.py. No external
calibration file: the user parametrises where the episode's workspace sits
relative to the arm base with a handful of CLI flags.

Pipeline:
    1. Load episode EE trajectory (world frame) from the LeRobot dataset
    2. Build T_arm_world. Auto-placement derives distance/table_height from
       the replay_start EEF (FK on output/replay_start.json). --distance /
       --table-height / --rotate-deg override per-axis.
    3. Retarget: world frame → arm base frame (R_arm_world, +translation)
    4. IK warm-start from replay_start (far-from-limits pose) → joint angles
    5. Workspace validation (abort on excessive IK error)
    6. Move safe_home → replay_start (leave the power-down-safe pose)
    7. Move replay_start → episode first frame
    8. Replay trajectory
    9. SafeHome returns to the original safe_home, then disconnects.

Usage:
    # Dry run: hard defaults (30 cm in front, table at base height)
    python scripts/04_replay_on_arm.py --dry-run --episode 0 \\
        --dataset-root ./data/dataset_v3

    # Real execution: pose the arm to the intended start, then launch.
    # Placement is inferred from the current EEF (X → distance, Z → table).
    python scripts/04_replay_on_arm.py \\
        --robot so101 --port COM5 --episode 0 --speed 0.3

    # Override placement explicitly if needed:
    python scripts/04_replay_on_arm.py \\
        --robot so101 --port COM5 --episode 0 --speed 0.3 \\
        --distance 0.30 --table-height 0.0 --rotate-deg 45
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ikpy.chain

from robots.base import RobotArm
from utils.safe_home import SafeHome


# ============================================================
# Robot driver registry (same as 05)
# ============================================================
ROBOT_DRIVERS = {
    "so101": ("robots.so101", "SO101Arm"),
}


def create_robot(robot_type: str, **kwargs) -> RobotArm:
    if robot_type not in ROBOT_DRIVERS:
        raise ValueError(f"Unknown robot: {robot_type}. Available: {list(ROBOT_DRIVERS)}")
    module_name, class_name = ROBOT_DRIVERS[robot_type]
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


# ============================================================
# Step 1: T_arm_world from workspace placement + episode data
# ============================================================
def compute_T_arm_world(
    distance: float = 0.30,
    table_height: float = 0.0,
    rotate_deg: float = 0.0,
    flip: bool = False,
    flip_lateral: bool = False,
) -> np.ndarray:
    """Build T_arm_world from workspace-placement parameters.

    World frame (ARKit, as stored in the dataset):
        +Y up (gravity), +X / +Z horizontal (orientation inherited from the
        capture session; epid origin = first-valid wrist).

    Arm frame (SO-arm 101 base_link, per URDF):
        +X forward, +Y left, +Z up.

    Transform chain (applied to world vectors):
        1. Optional world-space flips:
               flip         → diag(-1, 1, 1)  (mirror world X, forward axis)
               flip_lateral → diag(1, 1, -1)  (mirror world Z, maps to arm Y
                              sign — use when iPhone capture orientation and
                              robot deployment are mirrored left/right)
        2. Axis remap world→arm: Rx(90°), which sends world Y → arm Z
        3. Rotation around arm Z by rotate_deg (horizontal fine-tuning)
        4. Translation: (distance, 0, table_height) in arm frame
    """
    M_flip = np.diag([
        -1.0 if flip else 1.0,
        1.0,
        -1.0 if flip_lateral else 1.0,
    ]).astype(np.float64)
    R_x = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0],
    ], dtype=np.float64)
    th = np.radians(rotate_deg)
    c, s = np.cos(th), np.sin(th)
    R_z = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R = R_z @ R_x @ M_flip
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [distance, 0.0, table_height]

    print(f"  T_arm_world built from placement:")
    print(f"    distance={distance:.3f} m, table_height={table_height:.3f} m, "
          f"rotate_deg={rotate_deg:.1f}, flip={flip}, "
          f"flip_lateral={flip_lateral}")
    print(f"    Translation: [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}] m")
    print(f"    det(R) = {np.linalg.det(T[:3, :3]):.4f}")
    return T


def auto_placement_from_home(
    chain: "ikpy.chain.Chain", home_joints: dict[str, float],
) -> tuple[float, float, float]:
    """Derive (distance, table_height, rotate_deg) from a joint configuration.

    Maps episode anchor (first-N-frame wrist median) to the FK end-of-chain
    position. Pass the wrist sub-chain (ends at wrist_flex) so that the world
    origin aligns with the human-wrist keypoint that the dataset is anchored
    to; passing the full chain would anchor to gripper_frame instead, which
    offsets by the wrist→gripper link length and breaks the correspondence.
        distance     = end.X  (arm +X, forward extent)
        table_height = end.Z  (arm +Z, height above base_link)
        rotate_deg   = 0      (assume capture faced arm +X)
    """
    joint_vec = [0.0] * len(chain.links)
    for i, link in enumerate(chain.links):
        if link.name in home_joints:
            joint_vec[i] = np.radians(home_joints[link.name])
    T = chain.forward_kinematics(joint_vec)
    p = T[:3, 3]
    return float(p[0]), float(p[2]), 0.0


def load_replay_start(path: Path) -> tuple[dict[str, float], float]:
    """Load IK-friendly start pose recorded by scripts/_record_replay_start.py.

    Returns (arm_joints_dict, gripper_deg). Raises if the file is missing —
    the caller should instruct the user to record one first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"replay_start file not found at {path}.\n"
            "Pose the arm to an IK-friendly (extended) configuration, then "
            "record it with a short one-off utility before running 04."
        )
    with open(path, "r") as f:
        pose = json.load(f)
    gripper = float(pose.pop("gripper"))
    joints = {k: float(v) for k, v in pose.items()}
    return joints, gripper


def load_episode_actions(dataset_root: Path, episode_idx: int) -> dict:
    """Read one episode's action sequence from parquet files.

    Returns dict with:
        positions:  (T, 3) xyz in world frame (meters)
        rotations:  (T, 3) axis-angle in world frame
        grippers:   (T,)   normalized [0, 1]
        fps, n_frames
    """
    import pyarrow.parquet as pq
    import pyarrow as pa

    meta_path = dataset_root / "meta" / "info.json"
    with open(meta_path) as f:
        info = json.load(f)
    fps = info["fps"]

    data_dir = dataset_root / "data"
    tables = []
    for chunk_dir in sorted(data_dir.iterdir()):
        for pf in sorted(chunk_dir.glob("*.parquet")):
            tables.append(pq.read_table(pf))

    table = pa.concat_tables(tables)
    df = table.to_pandas()

    episode_df = df[df["episode_index"] == episode_idx].sort_values("frame_index")
    if len(episode_df) == 0:
        available = sorted(df["episode_index"].unique())
        raise ValueError(f"Episode {episode_idx} not found. Available: {available}")

    actions = np.stack(episode_df["action"].values)

    return {
        "positions": actions[:, :3].astype(np.float64),
        "rotations": actions[:, 3:6].astype(np.float64),
        "grippers": actions[:, 6].astype(np.float64),
        "fps": fps,
        "n_frames": len(actions),
    }


# ============================================================
# Step 2: Retarget — world frame → robot base frame
# ============================================================
def retarget_trajectory(
    positions: np.ndarray,
    rotations: np.ndarray,
    grippers: np.ndarray,
    T_robot_world: np.ndarray,
    scale: float = 1.0,
    gripper_open_deg: float = 0.0,
    gripper_close_deg: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform EE pose from world frame to robot base frame.

    Args:
        positions: (T, 3) xyz in world frame
        rotations: (T, 3) axis-angle in world frame
        grippers:  (T,)   normalized [0, 1]
        T_robot_world: 4x4 rigid transform
        scale: workspace scale factor

    Returns:
        positions_robot: (T, 3) in robot base frame
        rot_matrices_robot: (T, 3, 3) rotation matrices in robot base frame
        gripper_deg: (T,) gripper angle in degrees
    """
    R_rw = T_robot_world[:3, :3]
    t_rw = T_robot_world[:3, 3]

    # Scale around anchor (= world origin, positions already anchored to 0 in
    # 02_process.center_to_anchor). Scaling around the mean would leave
    # trajectory centered at its mean, breaking the "anchor ↔ replay_start
    # wrist_flex" correspondence whenever scale != 1.
    if scale != 1.0:
        positions = positions * scale

    # Transform positions: p_robot = R_rw @ p_world + t_rw
    pos_robot = (R_rw @ positions.T).T + t_rw

    # Transform rotations: R_robot = R_rw @ R_world
    R_world = Rotation.from_rotvec(rotations).as_matrix()  # (T, 3, 3)
    rot_robot = np.einsum('ij,tjk->tik', R_rw, R_world)   # (T, 3, 3)

    gripper_deg = gripper_open_deg + grippers * (gripper_close_deg - gripper_open_deg)

    print(f"  Retarget summary:")
    print(f"    Scale: {scale}")
    print(f"    Position range (robot frame):")
    print(f"      X: [{pos_robot[:,0].min():.4f}, {pos_robot[:,0].max():.4f}] m")
    print(f"      Y: [{pos_robot[:,1].min():.4f}, {pos_robot[:,1].max():.4f}] m")
    print(f"      Z: [{pos_robot[:,2].min():.4f}, {pos_robot[:,2].max():.4f}] m")
    print(f"    Gripper: [{gripper_deg.min():.1f}, {gripper_deg.max():.1f}] deg")

    return pos_robot, rot_robot, gripper_deg


# ============================================================
# Step 3: IK — EE position → joint angles
# ============================================================
def build_ik_chain(urdf_path: Path):
    """Build ikpy chain from URDF."""
    chain = ikpy.chain.Chain.from_urdf_file(
        str(urdf_path),
        base_elements=["base_link"],
        name="robot",
    )
    print(f"  IK chain: {len(chain.links)} links")
    return chain


def build_wrist_subchain(chain):
    """Sub-chain ending at wrist_flex — IK target aligns with dataset wrist.

    Dataset `eef_pos` is anchored to the human wrist keypoint (HaMeR), so the
    robot equivalent is wrist_flex, not gripper_frame. Solving IK on this
    truncated chain puts wrist_flex at the target instead of the gripper tip,
    so the physical correspondence (wrist ↔ wrist) is preserved.
    """
    wrist_idx = None
    for i, link in enumerate(chain.links):
        if link.name == "wrist_flex":
            wrist_idx = i
            break
    if wrist_idx is None:
        raise RuntimeError("wrist_flex link not found in chain")
    sub = ikpy.chain.Chain(chain.links[: wrist_idx + 1])
    print(f"  Wrist sub-chain: {len(sub.links)} links (ends at wrist_flex)")
    return sub


def get_revolute_indices(chain) -> list[int]:
    """Find indices of revolute joints in the ikpy chain."""
    indices = []
    for i, link in enumerate(chain.links):
        jt = getattr(link, '_joint_type', None) or getattr(link, 'joint_type', 'fixed')
        if jt == 'revolute':
            indices.append(i)
    return indices


def _find_wrist_roll_index(chain) -> int | None:
    """Find the ikpy chain index for wrist_roll joint."""
    for i, link in enumerate(chain.links):
        if link.name == "wrist_roll":
            return i
    return None


def _extract_wrist_roll_from_orientation(
    # TODO: Temporary approach for Phase 0 validation.
    # Extracts wrist_roll from ArUco rotation data via FK residual.
    # Not fully rigorous — should be replaced with visual-feedback-based
    # orientation alignment once the vision pipeline is integrated.
    chain,
    joint_angles: np.ndarray,
    target_orientations: np.ndarray,
    wrist_roll_idx: int,
) -> np.ndarray:
    """Compute wrist_roll angle from target orientation for each frame.

    Strategy: FK with wrist_roll=0 gives R_base_to_wrist. The target
    orientation is R_target. The wrist_roll rotation Rz(theta) satisfies:
        R_base_to_wrist @ Rz(theta) ≈ R_target
    So: Rz(theta) = R_base_to_wrist.T @ R_target
    Extract theta as atan2 of the resulting rotation around local Z axis.
    """
    T = len(joint_angles)
    wrist_roll_angles = np.zeros(T)

    for t in range(T):
        q = joint_angles[t].copy()
        q[wrist_roll_idx] = 0.0  # zero out wrist_roll to get R_base_to_wrist

        fk_no_roll = chain.forward_kinematics(q)
        R_no_roll = fk_no_roll[:3, :3]

        # R_residual = R_no_roll.T @ R_target → local rotation needed
        R_residual = R_no_roll.T @ target_orientations[t]

        # Extract Z-axis rotation from R_residual
        # For Rz(theta): R[0,0]=cos, R[1,0]=sin
        theta = np.arctan2(R_residual[1, 0], R_residual[0, 0])
        wrist_roll_angles[t] = theta

    return wrist_roll_angles


def compute_ik_trajectory(
    chain,
    positions: np.ndarray,
    orientations: np.ndarray | None = None,
    reference_pose_deg: list[float] | None = None,
    wrist_chain=None,
    wrist_roll_anchor_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert EE pose sequence to joint angle sequence via IK.

    Position IK runs on the wrist sub-chain (ends at wrist_flex) so the target
    maps to the robot wrist, matching the dataset's wrist-keypoint anchor.
    wrist_roll is solved separately from target orientation using the full
    chain. If `wrist_chain` is not provided it is derived from `chain`.

    Args:
        chain: ikpy Chain (full)
        positions: (T, 3) target wrist positions in robot frame
        orientations: (T, 3, 3) target rotation matrices in robot frame.
            If None, wrist_roll is unconstrained (position-only IK).
        reference_pose_deg: initial IK guess as full-chain degrees.
            If None, uses zeros.

    Returns:
        joint_angles_deg: (T, N_links) all link angles in degrees
        ik_errors: (T,) position error per frame in meters
    """
    T = len(positions)
    n_links = len(chain.links)
    if wrist_chain is None:
        wrist_chain = build_wrist_subchain(chain)
    n_wrist = len(wrist_chain.links)

    joint_angles = np.zeros((T, n_links))
    ik_errors = np.zeros(T)

    # Full-chain reference: initial guess for IK + fallback for links beyond
    # wrist_flex (wrist_roll, gripper_frame — filled here, wrist_roll later
    # overwritten by orientation extraction).
    if reference_pose_deg is not None:
        q_full_ref = np.radians(reference_pose_deg).astype(float)
        for i, link in enumerate(chain.links):
            b = getattr(link, 'bounds', None)
            if b and b != (None, None):
                lo, hi = b
                if lo is not None:
                    q_full_ref[i] = max(q_full_ref[i], lo + 0.01)
                if hi is not None:
                    q_full_ref[i] = min(q_full_ref[i], hi - 0.01)
    else:
        q_full_ref = np.zeros(n_links)

    q_wrist_prev = q_full_ref[:n_wrist].copy()

    use_orientation = orientations is not None
    wrist_roll_idx = _find_wrist_roll_index(chain) if use_orientation else None

    if use_orientation and wrist_roll_idx is not None:
        print(f"  IK mode: position on wrist sub-chain (joints 1-4) + "
              f"wrist_roll from orientation data")
    else:
        print(f"  IK mode: position-only (wrist sub-chain)")

    # Step 1: position IK on wrist sub-chain — target is wrist_flex position
    for t in range(T):
        q_w = wrist_chain.inverse_kinematics(
            target_position=positions[t],
            initial_position=q_wrist_prev,
        )
        joint_angles[t, :n_wrist] = q_w
        joint_angles[t, n_wrist:] = q_full_ref[n_wrist:]
        q_wrist_prev = q_w

        fk = wrist_chain.forward_kinematics(q_w)
        ik_errors[t] = np.linalg.norm(positions[t] - fk[:3, 3])

        if (t + 1) % 100 == 0 or t == 0:
            print(f"    IK frame {t+1}/{T}, error={ik_errors[t]*1000:.2f}mm")

    # Step 2: override wrist_roll from orientation data
    if use_orientation and wrist_roll_idx is not None:
        print(f"  Extracting wrist_roll from orientation data...")
        wrist_roll_rad = _extract_wrist_roll_from_orientation(
            chain, joint_angles, orientations, wrist_roll_idx,
        )

        # Anchor frame 0 to replay_start wrist_roll so the arm's initial wrist
        # pose matches replay_start. Preserves frame-to-frame relative roll;
        # absolute dataset orientation is discarded except as deltas.
        if wrist_roll_anchor_deg is not None:
            delta = np.radians(wrist_roll_anchor_deg) - wrist_roll_rad[0]
            wrist_roll_rad = wrist_roll_rad + delta
            print(f"    wrist_roll anchored to replay_start "
                  f"(delta={np.degrees(delta):+.2f}°)")

        joint_angles[:, wrist_roll_idx] = wrist_roll_rad

        wrist_roll_deg = np.rad2deg(wrist_roll_rad)
        print(f"    wrist_roll range: [{wrist_roll_deg.min():.1f}, "
              f"{wrist_roll_deg.max():.1f}] deg")

    joint_angles_deg = np.rad2deg(joint_angles)

    # Post-process: smooth joint trajectory
    joint_angles_deg = smooth_joint_trajectory(joint_angles_deg)

    print(f"  IK summary:")
    print(f"    Mean error: {ik_errors.mean()*1000:.2f} mm")
    print(f"    Max error:  {ik_errors.max()*1000:.2f} mm")
    print(f"    Frames > 5mm: {(ik_errors > 0.005).sum()}/{T}")

    return joint_angles_deg, ik_errors


def smooth_joint_trajectory(
    joint_angles_deg: np.ndarray,
    max_jump_deg: float = 20.0,
    window_size: int = 5,
) -> np.ndarray:
    """Clamp large frame-to-frame jumps + moving average smoothing."""
    T, N = joint_angles_deg.shape
    result = joint_angles_deg.copy()
    n_clamped = 0

    for j in range(N):
        for t in range(1, T - 1):
            if abs(result[t, j] - result[t - 1, j]) > max_jump_deg:
                result[t, j] = (result[t - 1, j] + result[t + 1, j]) / 2.0
                n_clamped += 1

    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        for j in range(N):
            col = result[:, j]
            if col.max() - col.min() > 0.01:
                padded = np.pad(col, window_size // 2, mode="edge")
                result[:, j] = np.convolve(padded, kernel, mode="valid")

    if n_clamped > 0:
        print(f"    Smoothing: clamped {n_clamped} jumps (>{max_jump_deg}°)")

    return result


# ============================================================
# Step 4: Workspace validation (dry-run)
# ============================================================
def workspace_check(
    joint_angles_deg: np.ndarray,
    gripper_deg: np.ndarray,
    ik_errors: np.ndarray,
    active_indices: list[int],
    fps: int,
    speed: float,
) -> bool:
    """Print trajectory stats and check for issues. Returns True if OK."""
    T = len(joint_angles_deg)
    active_joints = joint_angles_deg[:, active_indices]
    duration = T / fps / speed

    print(f"\n{'='*50}")
    print(f"  Workspace Validation")
    print(f"{'='*50}")
    print(f"  Frames: {T}, Playback: {duration:.1f}s ({fps}fps × {speed}x)")
    print(f"  IK error — mean: {ik_errors.mean()*1000:.2f}mm, "
          f"max: {ik_errors.max()*1000:.2f}mm")

    print(f"\n  Joint ranges (degrees):")
    for j in range(active_joints.shape[1]):
        col = active_joints[:, j]
        if col.max() - col.min() > 0.01:
            print(f"    Joint {active_indices[j]}: "
                  f"[{col.min():>8.2f}, {col.max():>8.2f}]  "
                  f"range={col.max()-col.min():.1f}°")
    print(f"  Gripper: [{gripper_deg.min():.1f}, {gripper_deg.max():.1f}]°")

    diffs = np.abs(np.diff(active_joints, axis=0))
    max_jumps = diffs.max(axis=0)
    print(f"\n  Max frame-to-frame jump:")
    for j in range(len(max_jumps)):
        if max_jumps[j] > 0.01:
            print(f"    Joint {active_indices[j]}: {max_jumps[j]:.2f}°")

    # Checks
    ok = True
    if ik_errors.max() > 0.01:
        print(f"\n  FAIL: IK error > 10mm on {(ik_errors > 0.01).sum()} frames")
        ok = False
    if diffs.max() > 30:
        print(f"  WARNING: large jumps > 30°/frame detected")

    if ok:
        print(f"\n  All checks passed.")
    return ok


# ============================================================
# Step 5: Initial pose alignment
# ============================================================
def move_to_start(
    robot: RobotArm,
    target_joints: dict[str, float],
    target_gripper: float,
    max_joint_diff_deg: float = 40.0,
    duration_s: float = 3.0,
    step_hz: float = 30.0,
) -> bool:
    """Move arm to first-frame pose with threshold check.

    Returns True if the move was executed, False if threshold exceeded.
    """
    current_joints = robot.get_joint_positions()
    current_gripper = robot.get_gripper_position()

    diffs = {
        k: abs(current_joints[k] - target_joints[k])
        for k in target_joints
    }
    max_diff = max(diffs.values())
    max_joint = max(diffs, key=diffs.get)

    print(f"\n  Moving to first-frame pose ({duration_s}s)...")
    print(f"    Current: {', '.join(f'{k}: {v:.1f}°' for k, v in current_joints.items())}")
    print(f"    Target:  {', '.join(f'{k}: {v:.1f}°' for k, v in target_joints.items())}")
    print(f"    Max diff: {max_diff:.1f}° ({max_joint})")

    if max_diff > max_joint_diff_deg:
        print(f"    WARNING: max joint diff {max_diff:.1f}° exceeds threshold "
              f"{max_joint_diff_deg:.1f}°")
        print(f"    The robot's current pose is far from the trajectory start.")
        print(f"    Consider manually posing the arm closer before running.")
        # Still proceed but warn — don't block
        print(f"    Proceeding with extended move time...")
        # Scale duration proportionally
        duration_s = max(duration_s, max_diff / 30.0)  # ~30°/s max speed
        print(f"    Adjusted duration: {duration_s:.1f}s")

    n_steps = int(duration_s * step_hz)
    dt = 1.0 / step_hz

    for s in range(1, n_steps + 1):
        t0 = time.perf_counter()
        alpha = s / n_steps

        interp_joints = {
            k: current_joints[k] + alpha * (target_joints[k] - current_joints[k])
            for k in target_joints
        }
        interp_gripper = current_gripper + alpha * (target_gripper - current_gripper)

        robot.send_all_positions(interp_joints, interp_gripper)

        elapsed = time.perf_counter() - t0
        time.sleep(max(dt - elapsed, 0))

    print(f"    Arrived at start position.")
    time.sleep(0.5)
    return True


# ============================================================
# Step 6: Replay trajectory
# ============================================================
def replay_trajectory(
    robot: RobotArm,
    joint_angles_deg: np.ndarray,
    gripper_deg: np.ndarray,
    active_indices: list[int],
    joint_names: list[str],
    fps: int,
    speed: float,
):
    """Send joint commands frame by frame."""
    active_joints = joint_angles_deg[:, active_indices]
    T = len(active_joints)
    effective_fps = fps * speed
    dt = 1.0 / effective_fps

    print(f"\n  Replaying {T} frames at {effective_fps:.0f} FPS ({T/effective_fps:.1f}s)")
    print(f"  Press Ctrl+C to stop.\n")

    for t in range(T):
        t0 = time.perf_counter()

        cmd_joints = {
            joint_names[i]: float(active_joints[t, i])
            for i in range(len(joint_names))
        }
        robot.send_all_positions(cmd_joints, float(gripper_deg[t]))

        if (t + 1) % 60 == 0:
            print(f"    Frame {t+1}/{T}")

        elapsed = time.perf_counter() - t0
        time.sleep(max(dt - elapsed, 0))

    print(f"\n  Replay complete!")


# ============================================================
# Main
# ============================================================
DEFAULT_DATASET = Path("data/test_pipeline_4_10/lerobot_dataset")


def main():
    parser = argparse.ArgumentParser(
        description="Replay EE trajectories on a robot arm",
    )
    parser.add_argument("--robot", type=str, default="so101")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (required for real execution)")
    parser.add_argument("--id", type=str, default="pass_follower_arm")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Playback speed (0.3 = 30%% speed, safer)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Workspace scale (<1 shrinks trajectory)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, no robot connection")
    parser.add_argument("--reference-pose", type=str, default=None,
                        help="IK reference pose as comma-separated degrees "
                             "(full chain, e.g. '0,-7,-105,97,12,-85,0'). "
                             "If omitted, uses replay_start pose.")
    parser.add_argument("--replay-start", type=Path,
                        default=Path("output/replay_start.json"),
                        help="IK-friendly start pose. Script moves here from "
                             "the connect-time safe pose before replay, and "
                             "auto-places the workspace around this EEF.")
    # Workspace placement. Defaults to None → auto-placement: the arm's
    # current pose (read on connect) becomes the episode anchor, so you can
    # pose the arm manually before launching and "that's where the episode
    # starts from". Explicit values override the auto defaults per-axis.
    parser.add_argument("--distance", type=float, default=None,
                        help="Distance from arm base_link (+X) to the "
                             "episode's anchor, meters. Default: home EEF X.")
    parser.add_argument("--table-height", type=float, default=None,
                        help="Episode anchor height in arm frame (+Z), meters. "
                             "Default: home EEF Z.")
    parser.add_argument("--rotate-deg", type=float, default=None,
                        help="Rotate episode around arm +Z (world +Y) axis "
                             "to match the horizontal orientation of the "
                             "capture. Default: 0.")
    parser.add_argument("--flip", action="store_true",
                        help="Mirror along world X (useful to repurpose a "
                             "right-hand episode for a left-hand workspace)")
    parser.add_argument("--flip-lateral", action="store_true",
                        help="Mirror along world Z, which becomes arm Y after "
                             "the world→arm remap. Use when the capture "
                             "session's horizontal orientation and the robot's "
                             "left/right are mirrored.")
    args = parser.parse_args()

    if not args.dry_run and args.port is None:
        parser.error("--port required for real execution. Use --dry-run to test.")

    # ----------------------------------------------------------
    # Step 1: Load episode data (no robot connection needed)
    # ----------------------------------------------------------
    print(f"\n=== Step 1: Load episode ===")
    data = load_episode_actions(args.dataset_root, args.episode)
    print(f"  Episode {args.episode}: {data['n_frames']} frames at {data['fps']} FPS")

    # Parse explicit reference pose if provided (IK warm-start)
    ref_pose = None
    if args.reference_pose:
        ref_pose = [float(x) for x in args.reference_pose.split(",")]

    # ----------------------------------------------------------
    # Dry run: hard defaults for any unspecified placement arg
    # ----------------------------------------------------------
    if args.dry_run:
        distance = args.distance if args.distance is not None else 0.30
        table_height = args.table_height if args.table_height is not None else 0.0
        rotate_deg = args.rotate_deg if args.rotate_deg is not None else 0.0

        print(f"\n=== Step 2: Build T_arm_world (dry-run defaults) ===")
        T_robot_world = compute_T_arm_world(
            distance=distance, table_height=table_height,
            rotate_deg=rotate_deg, flip=args.flip,
            flip_lateral=args.flip_lateral,
        )

        print(f"\n=== Step 3: Retarget ===")
        pos_robot, rot_robot, gripper_deg = retarget_trajectory(
            data["positions"], data["rotations"], data["grippers"],
            T_robot_world, scale=args.scale,
        )

        print(f"\n=== Step 4: Inverse Kinematics ===")
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent
        urdf_path = _PROJECT_ROOT / "assets" / "so101_new_calib.urdf"
        chain = build_ik_chain(urdf_path)
        wrist_chain = build_wrist_subchain(chain)
        active_indices = get_revolute_indices(chain)

        if ref_pose is None:
            ref_pose = [0.0] * len(chain.links)

        joint_angles_deg, ik_errors = compute_ik_trajectory(
            chain, pos_robot, orientations=rot_robot,
            reference_pose_deg=ref_pose,
            wrist_chain=wrist_chain,
        )
        workspace_check(
            joint_angles_deg, gripper_deg, ik_errors,
            active_indices, data["fps"], args.speed,
        )
        return

    # ----------------------------------------------------------
    # Real execution: connect robot, read home, then auto-place
    # ----------------------------------------------------------
    robot = create_robot(args.robot, port=args.port, id=args.id)
    robot.connect()

    chain = build_ik_chain(robot.urdf_path)
    wrist_chain = build_wrist_subchain(chain)
    active_indices = get_revolute_indices(chain)
    joint_names_active = [chain.links[i].name for i in active_indices]

    with SafeHome(robot) as home:
        try:
            # SafeHome's __enter__ recorded the current (safe) pose. Use a
            # separate replay_start pose (IK-friendly, extended) loaded from
            # disk for the actual episode motion. On exit SafeHome returns
            # to the safe pose, not replay_start — so the arm ends powered-
            # down-safe regardless of where the episode left it.
            replay_start_joints, replay_start_gripper = load_replay_start(
                args.replay_start,
            )
            print(f"  replay_start loaded from {args.replay_start}:")
            for k, v in replay_start_joints.items():
                print(f"    {k:16s}: {v:+7.2f}°")
            print(f"    {'gripper':16s}: {replay_start_gripper:+7.2f}°")

            # Auto-placement from replay_start wrist_flex position (not
            # gripper_frame): dataset anchor = human wrist keypoint, so the
            # world origin must map to the robot wrist, not the gripper tip.
            auto_d, auto_h, auto_r = auto_placement_from_home(
                wrist_chain, replay_start_joints,
            )
            distance = args.distance if args.distance is not None else auto_d
            table_height = args.table_height if args.table_height is not None else auto_h
            rotate_deg = args.rotate_deg if args.rotate_deg is not None else auto_r
            print(f"\n=== Step 2: Build T_arm_world ===")
            print(f"  auto-placement from replay_start wrist_flex: "
                  f"distance={auto_d:.3f} m, table_height={auto_h:.3f} m")
            T_robot_world = compute_T_arm_world(
                distance=distance, table_height=table_height,
                rotate_deg=rotate_deg, flip=args.flip,
                flip_lateral=args.flip_lateral,
            )

            print(f"\n=== Step 3: Retarget ===")
            pos_robot, rot_robot, gripper_deg = retarget_trajectory(
                data["positions"], data["rotations"], data["grippers"],
                T_robot_world, scale=args.scale,
            )

            # IK warm-start: replay_start (not safe pose) — far from limits.
            print(f"\n=== Step 4: Inverse Kinematics ===")
            if ref_pose is None:
                ref_pose = [0.0] * len(chain.links)
                for i, link in enumerate(chain.links):
                    if link.name in replay_start_joints:
                        ref_pose[i] = replay_start_joints[link.name]
                print(f"  Reference pose from replay_start: "
                      f"{', '.join(f'{v:.1f}°' for v in ref_pose if abs(v) > 0.01)}")

            joint_angles_deg, ik_errors = compute_ik_trajectory(
                chain, pos_robot, orientations=rot_robot,
                reference_pose_deg=ref_pose,
                wrist_chain=wrist_chain,
                wrist_roll_anchor_deg=replay_start_joints.get("wrist_roll"),
            )

            # Step 5: Workspace check
            print(f"\n=== Step 5: Workspace validation ===")
            ok = workspace_check(
                joint_angles_deg, gripper_deg, ik_errors,
                active_indices, data["fps"], args.speed,
            )
            if not ok:
                print(f"  Aborting due to workspace issues.")
                return

            # Step 6: safe → replay_start (leaves the safe pose, goes to
            # the IK-friendly configuration before episode motion begins).
            print(f"\n=== Step 6: Move safe → replay_start ===")
            move_to_start(robot, replay_start_joints, replay_start_gripper)

            # Step 7: replay_start → episode first frame
            print(f"\n=== Step 7: Move to episode start ===")
            active_joints_frame0 = joint_angles_deg[0, active_indices]
            target_joints = {
                joint_names_active[i]: float(active_joints_frame0[i])
                for i in range(len(joint_names_active))
            }
            move_to_start(robot, target_joints, float(gripper_deg[0]))

            # Step 8: Replay
            print(f"\n=== Step 8: Replay ===")
            replay_trajectory(
                robot, joint_angles_deg, gripper_deg,
                active_indices, joint_names_active,
                data["fps"], args.speed,
            )

        except KeyboardInterrupt:
            print(f"\n  Stopped by user.")

        # SafeHome.__exit__ returns to the recorded safe pose, then disconnects.


if __name__ == "__main__":
    main()
