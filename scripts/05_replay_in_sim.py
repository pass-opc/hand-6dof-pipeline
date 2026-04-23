#!/usr/bin/env python3
"""
Preview recorded EE trajectories in MuJoCo (pre-real-arm validation).

Pipeline position: Step 4b/4 — sim counterpart to 04_replay_on_arm.py.
Consumes a LeRobot dataset produced by 03_build_dataset.py, runs the same
EE→retarget→IK chain as 04, and drives trs_so_arm100 in MuJoCo instead of
real hardware. Purpose:
    (a) sanity-check a new dataset before spending real-arm time,
    (b) debug L1/L2 data-gate failures visually.

Reuses 04's pipeline functions via importlib dynamic load — 04's filename
starts with a digit and can't be imported normally. Hoisting the EE→IK
chain into a shared util module is the right long-term refactor; scoped out
here to keep 04 untouched.

Usage:
    conda activate lerobot
    cd hand-6dof-pipeline

    # Headless smoke (fast, no window). Good for CI / gate checks.
    python scripts/05_replay_in_sim.py \\
        --dataset-root output/03_dataset_v3 --episode 2 --no-gui

    # Full GUI preview of HaMeR v3 (needs --scale 0.5 per project memory).
    python scripts/05_replay_in_sim.py \\
        --dataset-root output/03_dataset_v3 --episode 2 --scale 0.5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Make sim.* / robots.* importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sim.mujoco_loader import load_so_arm101
from sim.mujoco_replay import ARM_JOINT_NAMES, replay_joint_trajectory


_DEFAULT_SCENE = (
    _PROJECT_ROOT / "assets" / "mujoco" / "trs_so101" / "scene.xml"
)
_DEFAULT_REPLAY_START = _PROJECT_ROOT / "output" / "replay_start.json"


def _load_04_module():
    """Dynamically load 04_replay_on_arm.py (digit-prefix filename)."""
    path = _PROJECT_ROOT / "scripts" / "04_replay_on_arm.py"
    spec = importlib.util.spec_from_file_location("_replay_on_arm_04", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to locate {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_placement(
    args, mod04, wrist_chain,
) -> tuple[float, float, float, dict | None, float | None]:
    """Pick placement (distance / table_height / rotate_deg).

    Priority: explicit CLI flag > auto-placement from replay_start.json >
    hard defaults (matches 04's --dry-run fallback).
    """
    joints: dict | None = None
    gripper: float | None = None
    auto_d = auto_h = auto_r = None

    if args.replay_start.exists():
        try:
            joints, gripper = mod04.load_replay_start(args.replay_start)
            auto_d, auto_h, auto_r = mod04.auto_placement_from_home(
                wrist_chain, joints,
            )
            print(
                f"  replay_start found → auto-placement: "
                f"distance={auto_d:.3f}, table_height={auto_h:.3f}"
            )
        except Exception as e:
            # Loud failure — don't swallow; the user asked for auto-placement
            # and got something broken instead of a silent default.
            print(f"  WARN: replay_start load failed ({e}); hard defaults")

    distance = args.distance if args.distance is not None else (
        auto_d if auto_d is not None else 0.30
    )
    table_height = args.table_height if args.table_height is not None else (
        auto_h if auto_h is not None else 0.0
    )
    rotate_deg = args.rotate_deg if args.rotate_deg is not None else (
        auto_r if auto_r is not None else 0.0
    )
    return distance, table_height, rotate_deg, joints, gripper


def _extract_arm_columns(
    joint_angles_deg, active_indices, joint_names_active,
):
    """Pick the 5 arm-joint columns in ARM_JOINT_NAMES order.

    04's `joint_angles_deg` is the full ikpy chain (virtual links included);
    active_indices / joint_names_active refer to revolute joints only. We
    need them in canonical sim order so the column index matches
    ARM_JOINT_NAMES in the replay driver.
    """
    active = joint_angles_deg[:, active_indices]   # (T, N_revolute)
    try:
        col_order = [joint_names_active.index(n) for n in ARM_JOINT_NAMES]
    except ValueError as e:
        raise RuntimeError(
            f"Joint name missing from IK chain ({joint_names_active}): {e}. "
            f"Expected all of {ARM_JOINT_NAMES}."
        ) from e
    return active[:, col_order]


def main():
    parser = argparse.ArgumentParser(
        description="Replay EE trajectories in MuJoCo (sim preview).",
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="LeRobot dataset root (v3.0)")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scene-xml", type=Path, default=_DEFAULT_SCENE)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0=realtime, 5.0=5× faster)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Workspace scale (<1 shrinks). HaMeR v3 → 0.5.")
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless mode (skip MuJoCo viewer).")
    parser.add_argument("--physics", action="store_true",
                        help="Use mj_step (full dynamics). Default is "
                             "kinematics-only preview.")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the trajectory until the user closes the "
                             "GUI window. Default (GUI) holds on the final "
                             "pose instead.")
    # Placement (same semantics as 04). Auto-placement from replay_start.json
    # if available, else the hard defaults from 04 --dry-run.
    parser.add_argument("--replay-start", type=Path,
                        default=_DEFAULT_REPLAY_START)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--table-height", type=float, default=None)
    parser.add_argument("--rotate-deg", type=float, default=None)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--flip-lateral", action="store_true")
    args = parser.parse_args()

    mod04 = _load_04_module()

    # --- Step 1: dataset → EE trajectory -----------------------------------
    print(f"\n=== Step 1: Load episode ===")
    data = mod04.load_episode_actions(args.dataset_root, args.episode)
    print(
        f"  Episode {args.episode}: {data['n_frames']} frames "
        f"@ {data['fps']} FPS"
    )

    # --- Step 2: IK chain + placement --------------------------------------
    print(f"\n=== Step 2: Build chain + placement ===")
    urdf_path = _PROJECT_ROOT / "assets" / "so101_new_calib.urdf"
    chain = mod04.build_ik_chain(urdf_path)
    wrist_chain = mod04.build_wrist_subchain(chain)
    active_indices = mod04.get_revolute_indices(chain)
    joint_names_active = [chain.links[i].name for i in active_indices]

    distance, table_height, rotate_deg, rs_joints, rs_gripper = (
        _resolve_placement(args, mod04, wrist_chain)
    )
    T_arm_world = mod04.compute_T_arm_world(
        distance=distance,
        table_height=table_height,
        rotate_deg=rotate_deg,
        flip=args.flip,
        flip_lateral=args.flip_lateral,
    )

    # --- Step 3: retarget world → arm frame --------------------------------
    print(f"\n=== Step 3: Retarget ===")
    pos_arm, rot_arm, gripper_deg = mod04.retarget_trajectory(
        data["positions"], data["rotations"], data["grippers"],
        T_arm_world, scale=args.scale,
    )

    # --- Step 4: inverse kinematics ----------------------------------------
    print(f"\n=== Step 4: Inverse Kinematics ===")
    ref_pose_deg = [0.0] * len(chain.links)
    if rs_joints:
        for i, link in enumerate(chain.links):
            if link.name in rs_joints:
                ref_pose_deg[i] = rs_joints[link.name]
    joint_angles_deg, ik_errors = mod04.compute_ik_trajectory(
        chain, pos_arm, orientations=rot_arm,
        reference_pose_deg=ref_pose_deg,
        wrist_chain=wrist_chain,
        wrist_roll_anchor_deg=(
            rs_joints.get("wrist_roll") if rs_joints else None
        ),
    )
    mod04.workspace_check(
        joint_angles_deg, gripper_deg, ik_errors,
        active_indices, data["fps"], args.speed,
    )

    # --- Step 5: MuJoCo drive ----------------------------------------------
    print(f"\n=== Step 5: MuJoCo replay ===")
    arm_seq = _extract_arm_columns(
        joint_angles_deg, active_indices, joint_names_active,
    )
    scene = load_so_arm101(args.scene_xml)
    replay_joint_trajectory(
        scene,
        joint_angles_deg=arm_seq,
        gripper_deg=gripper_deg,
        fps=int(data["fps"]),
        speed=args.speed,
        no_gui=args.no_gui,
        physics=args.physics,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
