"""
Load the SO-ARM101 MJCF and bridge LeRobot joint names to MuJoCo qpos / ctrl.

Pipeline position: helper for scripts/05_replay_in_sim.py.

Input:  scene_xml_path (TheRobotStudio SO-ARM100 Simulation/SO101/scene.xml),
        joint dicts in LeRobot naming (shoulder_pan, shoulder_lift, ...).
Output: LoadedScene bundling mjModel, mjData, and the index maps needed to
        write a joint command dict into the simulator.

Why a dedicated loader: even with same-source URDF+MJCF (both generated from
the same Onshape CAD via onshape-to-robot — joint zero-points and names
already aligned), we still need to resolve LeRobot joint name → mjModel
qposadr / actuator id. Centralizing that lookup here means replay scripts and
tests stay name-based and never hard-code numeric indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


# LeRobot naming → MJCF joint name.
# Identity mapping because TRS Simulation/SO101 MJCF is same-source with the
# LeRobot URDF (both from onshape-to-robot). The dict is retained as a single
# source of truth: if we later swap MJCF vendors (e.g. Menagerie's renamed
# Rotation/Pitch/Elbow/... variant), only this table changes.
LEROBOT_TO_MJCF: dict[str, str] = {
    "shoulder_pan":  "shoulder_pan",
    "shoulder_lift": "shoulder_lift",
    "elbow_flex":    "elbow_flex",
    "wrist_flex":    "wrist_flex",
    "wrist_roll":    "wrist_roll",
    "gripper":       "gripper",
}


@dataclass
class LoadedScene:
    """Bundle of MuJoCo state + name→index maps for downstream writers."""
    model: mujoco.MjModel
    data: mujoco.MjData
    # All 6 joints are 1-DoF hinges → qpos_idx == jnt_qposadr[j].
    # Actuators are 1:1 with joints (position servos), so ctrl_idx == act_id.
    lerobot_to_qpos_idx: dict[str, int]
    lerobot_to_ctrl_idx: dict[str, int]
    joint_range_rad: dict[str, tuple[float, float]]


def load_so_arm101(scene_xml_path: Path | str) -> LoadedScene:
    """Load SO-ARM101 scene.xml and build the LeRobot→MJCF index maps.

    Fails loudly if any of the 6 LeRobot joint names is missing (we must not
    silently drop a joint — a half-connected mapping would produce subtly
    wrong qpos writes that are hard to spot in the viewer).
    """
    xml = Path(scene_xml_path)
    if not xml.exists():
        raise FileNotFoundError(f"MuJoCo scene not found: {xml}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)

    lerobot_to_qpos_idx: dict[str, int] = {}
    lerobot_to_ctrl_idx: dict[str, int] = {}
    joint_range_rad: dict[str, tuple[float, float]] = {}

    for lerobot_name, mjcf_name in LEROBOT_TO_MJCF.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
        if jnt_id < 0:
            raise ValueError(
                f"Joint '{mjcf_name}' (LeRobot '{lerobot_name}') not in "
                f"MJCF {xml.name}. Check LEROBOT_TO_MJCF table."
            )
        lerobot_to_qpos_idx[lerobot_name] = int(model.jnt_qposadr[jnt_id])

        act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, mjcf_name,
        )
        if act_id < 0:
            raise ValueError(
                f"Actuator '{mjcf_name}' missing — position-servo "
                f"actuators expected for all 6 joints."
            )
        lerobot_to_ctrl_idx[lerobot_name] = int(act_id)

        lo, hi = model.jnt_range[jnt_id]
        joint_range_rad[lerobot_name] = (float(lo), float(hi))

    return LoadedScene(
        model=model,
        data=data,
        lerobot_to_qpos_idx=lerobot_to_qpos_idx,
        lerobot_to_ctrl_idx=lerobot_to_ctrl_idx,
        joint_range_rad=joint_range_rad,
    )


def apply_joint_positions_deg(
    scene: LoadedScene,
    joint_positions_deg: dict[str, float],
    *,
    target: str = "ctrl",
) -> None:
    """Write joint angles (degrees, LeRobot naming) to qpos or ctrl.

    target='ctrl'  : replay path — position actuators track to this set-point.
    target='qpos'  : direct state write. Use together with mj_kinematics for a
                     kinematics-only preview that bypasses actuator dynamics
                     (Phase 1.0: we only care about geometric feasibility).

    Out-of-range commands are warned about but not clamped — silent clamping
    would mask upstream bugs (smoothing / IK overshoot) that we want to see.
    """
    if target not in ("ctrl", "qpos"):
        raise ValueError(f"target must be 'ctrl' or 'qpos', got '{target}'")

    for lerobot_name, deg in joint_positions_deg.items():
        if lerobot_name not in LEROBOT_TO_MJCF:
            raise KeyError(
                f"Unknown LeRobot joint '{lerobot_name}'. "
                f"Known: {list(LEROBOT_TO_MJCF)}"
            )
        rad = float(np.radians(deg))
        lo, hi = scene.joint_range_rad[lerobot_name]
        if not (lo - 1e-3 <= rad <= hi + 1e-3):
            print(
                f"    [mujoco] WARN: {lerobot_name}={deg:+.2f}deg "
                f"(={rad:+.3f} rad) out of "
                f"[{np.degrees(lo):+.1f}, {np.degrees(hi):+.1f}]deg"
            )
        if target == "ctrl":
            idx = scene.lerobot_to_ctrl_idx[lerobot_name]
            scene.data.ctrl[idx] = rad
        else:
            idx = scene.lerobot_to_qpos_idx[lerobot_name]
            scene.data.qpos[idx] = rad


def reset_to_keyframe(scene: LoadedScene, keyframe_name: str = "home") -> None:
    """Reset qpos / qvel / ctrl to a named keyframe. Raises if missing."""
    key_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name,
    )
    if key_id < 0:
        raise ValueError(f"Keyframe '{keyframe_name}' not in MJCF")
    mujoco.mj_resetDataKeyframe(scene.model, scene.data, key_id)
