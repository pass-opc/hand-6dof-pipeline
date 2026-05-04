"""
Real-arm replay backend for SO-arm 101.

Pipeline position: registered for `(so101, real)` in replay/__init__.py.
Consumed via `python -m replay --output real --port COMx`.

Reads `.qpos.npz` + `.qpos.meta.json` (qpos in radians, output of
`python -m retarget --robot so101`), drives the LeRobot SO100Follower
with SafeHome wrapping connect/disconnect.

Self-contained: SO101Arm driver + SafeHome live in this module rather
than under utils/ — they are real-arm-replay-only utilities, no
upstream pipeline stage uses them.

Hold-last on invalid frames so playback length matches source recording.
First valid frame anchors the move-to-start step; the arm is parked at
that pose (slowly) before episode replay begins.

Usage (single episode dry-run):
    python -m replay --qpos-root output/.../05_qpos_so101 \\
        --output real --port COM5 --speed 0.3 --episodes <sid>
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_URDF = _PROJECT_ROOT / "assets" / "so101_new_calib.urdf"


# Canonical 5-arm-joint order; matches retarget/so101.py's first 5
# JOINT_NAMES entries and replay/sim/mujoco_so101.ARM_JOINT_NAMES.
ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


# =============================================================================
# Hardware driver — wraps LeRobot SO100Follower
# =============================================================================

class SO101Arm:
    """SO-arm 101 driver using LeRobot SO100Follower backend.

    Constructed lazy-import so this module loads in environments without
    LeRobot (tests, CI). Connection happens in `connect()`.
    """

    def __init__(
        self,
        port: str,
        id: str = "pass_follower_arm",
        max_relative_target: float = 10.0,
    ):
        self._port = port
        self._id = id
        self._max_relative_target = max_relative_target
        self._robot = None  # SO100Follower instance, created on connect()

    @property
    def joint_names(self) -> list[str]:
        return list(ARM_JOINT_NAMES)

    def connect(self) -> None:
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
        config = SO100FollowerConfig(
            port=self._port,
            use_degrees=True,
            max_relative_target=self._max_relative_target,
            id=self._id,
        )
        self._robot = SO100Follower(config)
        self._robot.connect()
        print(f"  [SO101] Connected on {self._port} (id={self._id})")

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None
            print(f"  [SO101] Disconnected.")

    def get_joint_positions(self) -> dict[str, float]:
        obs = self._robot.get_observation()
        return {name: obs[f"{name}.pos"] for name in ARM_JOINT_NAMES}

    def get_gripper_position(self) -> float:
        obs = self._robot.get_observation()
        return obs["gripper.pos"]

    def send_all_positions(
        self, joint_pos: dict[str, float], gripper_deg: float,
    ) -> None:
        action = {f"{name}.pos": joint_pos[name] for name in joint_pos}
        action["gripper.pos"] = gripper_deg
        self._robot.send_action(action)


# =============================================================================
# Safety: record connect-time pose, return to it on disconnect
# =============================================================================

class SafeHome:
    """Context manager that records home position on enter and restores
    it on exit. Prevents the arm from falling when servos lose torque
    after disconnect."""

    def __init__(
        self, robot: SO101Arm,
        return_duration_s: float = 3.0,
        step_hz: float = 30.0,
    ):
        self.robot = robot
        self.return_duration_s = return_duration_s
        self.step_hz = step_hz
        self.home_joints: dict[str, float] | None = None
        self.home_gripper: float | None = None

    def __enter__(self):
        self.home_joints = self.robot.get_joint_positions()
        self.home_gripper = self.robot.get_gripper_position()
        pos_str = ", ".join(
            f"{n}: {v:.1f}°" for n, v in self.home_joints.items()
        )
        pos_str += f", gripper: {self.home_gripper:.1f}°"
        print(f"  [SafeHome] Home recorded: {pos_str}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._return_home()
        except Exception as e:
            print(f"  [SafeHome] WARNING: failed to return home: {e}")
        finally:
            try:
                self.robot.disconnect()
            except Exception as e:
                print(f"  [SafeHome] WARNING: disconnect failed: {e}")
        return False

    def _return_home(self) -> None:
        if self.home_joints is None:
            return
        current_joints = self.robot.get_joint_positions()
        current_gripper = self.robot.get_gripper_position()
        max_dist = max(
            abs(current_joints[k] - self.home_joints[k])
            for k in self.home_joints
        )
        max_dist = max(max_dist, abs(current_gripper - self.home_gripper))
        print(f"\n  [SafeHome] Returning to home ({self.return_duration_s}s, "
              f"max distance {max_dist:.1f}°)...")
        _interpolate_move(
            self.robot, current_joints, current_gripper,
            self.home_joints, self.home_gripper,
            duration_s=self.return_duration_s, step_hz=self.step_hz,
        )
        print(f"  [SafeHome] Home position reached.")


# =============================================================================
# Trajectory loader (.qpos.npz radians → arm + gripper degrees)
# =============================================================================

@dataclass
class _Episode:
    arm_deg: np.ndarray             # (T, 5)
    gripper_deg: np.ndarray         # (T,)
    qpos_valid: np.ndarray          # (T,)
    n_total: int
    fps: int


def _qpos_to_arm_gripper_deg(
    qpos_rad: np.ndarray, qpos_valid: np.ndarray, joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Slice qpos (radians, joint-name order) → (arm_deg, gripper_deg).

    Hold-last fill on invalid frames keeps the timeline aligned with the
    source recording without driving the arm with NaN commands. Same
    helper as the sim backend uses; duplicated here to keep the real
    backend self-contained (no cross-import to replay.sim).
    """
    name_to_col = {n: i for i, n in enumerate(joint_names)}
    expected = list(ARM_JOINT_NAMES) + ["gripper"]
    if not all(n in name_to_col for n in expected):
        raise ValueError(
            f"qpos joint_names {joint_names!r} missing expected "
            f"SO-101 joints {expected!r}"
        )
    arm_cols = [name_to_col[n] for n in ARM_JOINT_NAMES]
    gripper_col = name_to_col["gripper"]
    arm_rad = qpos_rad[:, arm_cols].astype(np.float64)
    gripper_rad = qpos_rad[:, gripper_col].astype(np.float64)

    valid_idx = np.flatnonzero(qpos_valid)
    if len(valid_idx) == 0:
        raise ValueError("qpos has no valid frames; nothing to replay")
    first_valid = int(valid_idx[0])
    arm_rad[:first_valid] = arm_rad[first_valid]
    gripper_rad[:first_valid] = gripper_rad[first_valid]
    last_arm = arm_rad[first_valid].copy()
    last_grip = float(gripper_rad[first_valid])
    for t in range(first_valid, len(qpos_valid)):
        if qpos_valid[t]:
            last_arm = arm_rad[t].copy()
            last_grip = float(gripper_rad[t])
        else:
            arm_rad[t] = last_arm
            gripper_rad[t] = last_grip
    return np.degrees(arm_rad), np.degrees(gripper_rad)


def _load_episode(qpos_npz_path: Path, qpos_meta_path: Path) -> _Episode:
    arr = np.load(qpos_npz_path, allow_pickle=False)
    meta = json.loads(qpos_meta_path.read_text(encoding="utf-8"))
    hand = meta["hand"]
    qpos_key = f"{hand}_qpos"
    valid_key = f"{hand}_qpos_valid"
    if qpos_key not in arr.files or valid_key not in arr.files:
        raise KeyError(
            f"{qpos_npz_path.name} missing {qpos_key!r}/{valid_key!r}. "
            f"Available: {arr.files}"
        )
    qpos = arr[qpos_key]
    valid = arr[valid_key].astype(bool)
    fps_meta = int(meta.get("fps", 30))
    if qpos.ndim != 2 or qpos.shape[1] != len(meta["joint_names"]):
        raise ValueError(
            f"qpos shape {qpos.shape} doesn't match joint_names "
            f"({len(meta['joint_names'])} cols). Re-run retarget."
        )
    arm_deg, gripper_deg = _qpos_to_arm_gripper_deg(
        qpos, valid, meta["joint_names"],
    )
    return _Episode(
        arm_deg=arm_deg, gripper_deg=gripper_deg, qpos_valid=valid,
        n_total=len(qpos), fps=fps_meta,
    )


# =============================================================================
# Movement primitives
# =============================================================================

def _interpolate_move(
    robot: SO101Arm,
    src_joints: dict[str, float], src_gripper: float,
    dst_joints: dict[str, float], dst_gripper: float,
    duration_s: float, step_hz: float,
) -> None:
    """Linear-in-degrees interpolation between two configurations.
    Bounded by `max_relative_target` per the SO100Follower driver."""
    n_steps = max(int(duration_s * step_hz), 1)
    dt = 1.0 / step_hz
    for s in range(1, n_steps + 1):
        t0 = time.perf_counter()
        alpha = s / n_steps
        interp_joints = {
            k: src_joints[k] + alpha * (dst_joints[k] - src_joints[k])
            for k in dst_joints
        }
        interp_gripper = src_gripper + alpha * (dst_gripper - src_gripper)
        robot.send_all_positions(interp_joints, interp_gripper)
        elapsed = time.perf_counter() - t0
        time.sleep(max(dt - elapsed, 0.0))


def _move_to_first_frame(
    robot: SO101Arm, ep: _Episode,
    *, duration_s: float = 3.0, step_hz: float = 30.0,
) -> None:
    """Slowly park the arm at frame 0's pose before episode replay."""
    target_joints = {
        ARM_JOINT_NAMES[i]: float(ep.arm_deg[0, i])
        for i in range(len(ARM_JOINT_NAMES))
    }
    target_gripper = float(ep.gripper_deg[0])
    src_joints = robot.get_joint_positions()
    src_gripper = robot.get_gripper_position()
    diffs = {k: abs(src_joints[k] - target_joints[k]) for k in target_joints}
    max_diff = max(diffs.values())
    print(f"  [real] Move to first frame  (max diff {max_diff:.1f}°, "
          f"{duration_s:.1f}s)")
    _interpolate_move(
        robot, src_joints, src_gripper, target_joints, target_gripper,
        duration_s=max(duration_s, max_diff / 30.0), step_hz=step_hz,
    )


def _replay_episode(
    robot: SO101Arm, ep: _Episode,
    *, speed: float, step_hz_log: int = 60,
) -> None:
    effective_fps = max(ep.fps * speed, 1e-6)
    dt = 1.0 / effective_fps
    print(f"  [real] Replaying {ep.n_total} frames "
          f"@ {effective_fps:.1f} Hz  (Ctrl+C to stop)")
    for t in range(ep.n_total):
        t0 = time.perf_counter()
        cmd = {
            ARM_JOINT_NAMES[i]: float(ep.arm_deg[t, i])
            for i in range(len(ARM_JOINT_NAMES))
        }
        robot.send_all_positions(cmd, float(ep.gripper_deg[t]))
        if (t + 1) % step_hz_log == 0:
            print(f"    [real] frame {t + 1}/{ep.n_total}")
        elapsed = time.perf_counter() - t0
        time.sleep(max(dt - elapsed, 0.0))
    print(f"  [real] Replay complete.")


# =============================================================================
# Backend entrypoint (called by replay/__main__.py)
# =============================================================================

def run(
    *,
    qpos_npz_path: Path,
    qpos_meta_path: Path,
    output: str,                # informational; real backend ignores
    port: str | None = None,
    arm_id: str = "pass_follower_arm",
    speed: float = 0.3,
    max_relative_target: float = 10.0,
    move_to_start_duration_s: float = 3.0,
    dry_run: bool = False,
    **_unused,
) -> dict | None:
    """Drive the SO-arm 101 with `.qpos.npz`. SafeHome wraps connect/disconnect.

    `dry_run=True` loads + decodes the trajectory but never opens the
    serial port; useful as a CI-friendly smoke test.
    """
    ep = _load_episode(qpos_npz_path, qpos_meta_path)
    print(f"  [real] {ep.n_total} frames @ {ep.fps} fps  "
          f"(valid={int(ep.qpos_valid.sum())})")

    if dry_run:
        print(f"  [real] dry-run: would arm-replay episode of "
              f"{ep.n_total} frames at speed {speed:.2f}")
        return {"n_total": ep.n_total, "executed": False}

    if port is None:
        raise ValueError(
            "real replay requires --port (e.g. COM5). Use --dry-run to "
            "smoke-test without hardware."
        )
    robot = SO101Arm(
        port=port, id=arm_id, max_relative_target=max_relative_target,
    )
    robot.connect()
    with SafeHome(robot):
        try:
            _move_to_first_frame(
                robot, ep, duration_s=move_to_start_duration_s,
            )
            _replay_episode(robot, ep, speed=speed)
        except KeyboardInterrupt:
            print(f"\n  [real] Stopped by user.")
    return {"n_total": ep.n_total, "executed": True}
