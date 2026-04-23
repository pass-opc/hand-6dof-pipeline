"""
MuJoCo replay driver for a pre-computed joint trajectory.

Pipeline position: consumed by scripts/05_replay_in_sim.py. Given a joint
trajectory (degrees, LeRobot naming) already produced upstream from the
EE→retarget→IK chain (reused from 04_replay_on_arm), drive trs_so101
either in a passive MuJoCo viewer (hands-on preview) or headless (CI / batch
validation of data-gate failures).

Input:  LoadedScene (from mujoco_loader), joint_angles_deg (T, 5 arm joints),
        gripper_deg (T,), fps, speed.
Output: renders frames / advances sim state (side effects only).

Why separate from the loader: the loader is stateless; this module owns the
render-loop timing and the kinematics-vs-dynamics decision. Keeping them
split lets tests drive `apply_joint_positions_deg` without pulling in the
viewer / timing machinery.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import mujoco
import mujoco.viewer
import numpy as np

from sim.mujoco_loader import (
    LoadedScene,
    apply_joint_positions_deg,
    reset_to_keyframe,
)


# Canonical arm joint order — matches SO101Arm.joint_names (robots/so101.py)
# so we can consume 04's `active_joints` columns by name.
ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def replay_joint_trajectory(
    scene: LoadedScene,
    joint_angles_deg: np.ndarray,
    gripper_deg: np.ndarray,
    *,
    fps: int = 30,
    speed: float = 1.0,
    no_gui: bool = False,
    reset_keyframe: str | None = "home",
    physics: bool = False,
    loop: bool = False,
) -> None:
    """Drive the SO-ARM101 through a pre-computed joint trajectory.

    Args:
        scene: loaded scene from sim.mujoco_loader.load_so_arm101.
        joint_angles_deg: (T, 5) arm joints in degrees. Column order must
            match ARM_JOINT_NAMES — the caller is responsible for aligning
            the ikpy chain's joint output to this order.
        gripper_deg: (T,) gripper angle in degrees.
        fps: dataset fps (defines nominal step size).
        speed: playback multiplier (1.0 = realtime, 5.0 = 5× faster).
        no_gui: headless mode; skips viewer. Required in CI / unit tests —
            launch_passive opens a GLFW window that blocks.
        reset_keyframe: jump to this keyframe before the loop; None skips.
            Default "home" puts the arm in a neutral pose so the first
            commanded frame has a predictable starting state.
        physics: True → mj_step (dynamics + contact); False → mj_kinematics
            (geometry only). Phase 1.0 default is False — we validate
            geometric feasibility, not servo tracking under load.
        loop: GUI only. True → restart from frame 0 when the trajectory ends.
            False → hold on the final pose until the user closes the window.
            Headless runs exit as soon as the trajectory finishes.
    """
    T = len(joint_angles_deg)
    if joint_angles_deg.shape[1] != len(ARM_JOINT_NAMES):
        raise ValueError(
            f"joint_angles_deg has {joint_angles_deg.shape[1]} cols, "
            f"expected {len(ARM_JOINT_NAMES)} ({ARM_JOINT_NAMES})"
        )
    if len(gripper_deg) != T:
        raise ValueError(
            f"gripper_deg length {len(gripper_deg)} != trajectory length {T}"
        )

    if reset_keyframe is not None:
        reset_to_keyframe(scene, reset_keyframe)

    effective_fps = max(fps * speed, 1e-6)
    dt = 1.0 / effective_fps

    viewer_ctx = (
        nullcontext(None) if no_gui
        else mujoco.viewer.launch_passive(scene.model, scene.data)
    )

    print(
        f"  [sim] Replaying {T} frames at {effective_fps:.1f} Hz  "
        f"({'kinematics' if not physics else 'physics'}, "
        f"{'headless' if no_gui else 'gui'})"
    )

    with viewer_ctx as viewer:
        pass_num = 0
        while True:
            pass_num += 1
            if loop and pass_num > 1:
                print(f"  [sim] Loop pass #{pass_num}")

            user_closed = False
            for t in range(T):
                # Break out promptly if the user closes the GUI window.
                if viewer is not None and not viewer.is_running():
                    user_closed = True
                    break
                t0 = time.perf_counter()

                cmd: dict[str, float] = {
                    name: float(joint_angles_deg[t, i])
                    for i, name in enumerate(ARM_JOINT_NAMES)
                }
                cmd["gripper"] = float(gripper_deg[t])

                # ctrl drives position actuators (mj_step path). In kinematics
                # mode we also set qpos directly to bypass actuator dynamics —
                # kp/dampratio tuning would otherwise lag the preview behind
                # the commanded trajectory.
                apply_joint_positions_deg(scene, cmd, target="ctrl")
                if physics:
                    mujoco.mj_step(scene.model, scene.data)
                else:
                    apply_joint_positions_deg(scene, cmd, target="qpos")
                    mujoco.mj_kinematics(scene.model, scene.data)

                if viewer is not None:
                    viewer.sync()

                if (t + 1) % 60 == 0 or t == T - 1:
                    print(f"    [sim] frame {t + 1}/{T}")

                if not no_gui:
                    elapsed = time.perf_counter() - t0
                    time.sleep(max(dt - elapsed, 0.0))

            if user_closed or no_gui or not loop:
                break

        # GUI + single-play: hold on the final pose so the user can inspect
        # the scene instead of the window snapping shut.
        if viewer is not None and not loop:
            print("  [sim] Replay done. Window stays open — close it to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(1.0 / 30.0)

    print(f"  [sim] Replay complete.")
