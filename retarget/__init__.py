"""
MANO-keypoint → robot-qpos retargeting (recording-tool agnostic).

Pipeline position: bridges the per-line raw output (`*.processed.npz` from
scripts/02_process or scripts335/02_process) and the replay layer
(`replay/`). Consumers should use the `python -m retarget` CLI; this
module exposes the registry and result type for in-process callers.

Boundary (intentionally narrow):
  - INPUTS  : `.processed.npz` only. Cam-frame, raw HaMeR/WiLor outputs.
  - DOES    : per-frame retarget (dex_retargeting / IK).
  - DOES NOT: smooth, interpolate, fill NaN, normalize gripper, build
              state/action. Frames the retarget can't solve come back
              with `qpos_valid[t] = False` and NaN qpos[t]. Quality
              gating happens upstream (process step's quality_passed).

Adding a new robot:
  1. Implement a backend in retarget/<robot>.py exposing a class with
     `name`, `required_keys(hand) -> set[str]`, and
     `retarget_episode(source, hand, **kwargs) -> RetargetResult`.
  2. Register it in `_BACKENDS` below.
  3. Document its required source keys + conda env in retarget/README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetargetResult:
    """Per-episode retarget output.

    Lengths match the trimmed slice the backend was given (T = trim_last
    - trim_first), not the source episode's full frame count. The CLI
    pads back to full length so downstream replay stays aligned with the
    original timestamps_us index.
    """
    qpos: np.ndarray            # (T, N) float32 — NaN at invalid frames
    qpos_valid: np.ndarray      # (T,) bool      — True where retarget ran
    joint_names: list[str]      # length N
    extras: dict                # backend-specific (target_human_indices etc.)


# (robot, env) -> "module:class_name" entrypoint. Lazy-imported so the
# heavy backends (dex_retargeting needs torch + numpy>=2) don't load
# unless the caller actually wants them.
_BACKENDS: dict[tuple[str, str], str] = {
    # dex_retargeting (opc-dex env). Same backend covers all dex hands.
    ("shadow",  "mujoco"): "retarget.dex_hands:DexBackend",
    ("leap",    "mujoco"): "retarget.dex_hands:DexBackend",
    ("allegro", "mujoco"): "retarget.dex_hands:DexBackend",
    ("inspire", "mujoco"): "retarget.dex_hands:DexBackend",
    ("svh",     "mujoco"): "retarget.dex_hands:DexBackend",
    ("ability", "mujoco"): "retarget.dex_hands:DexBackend",
    ("panda",   "mujoco"): "retarget.dex_hands:DexBackend",
    # SO-arm 101 (lerobot env) — single backend covers sim + real because
    # retarget output is the geometric joint trajectory; sim vs real is
    # a downstream replay choice. mink-based, modeled on lerobot-seeed
    # RobotKinematics (placo/frame-task soft constraints).
    ("so101",   "mujoco"): "retarget.so101:So101Backend",
    ("so101",   "real"):   "retarget.so101:So101Backend",
}


def get_backend(robot: str, env: str = "mujoco"):
    """Resolve `(robot, env)` to a backend class. Raises with a clear
    message listing supported combos if not registered."""
    key = (robot, env)
    if key not in _BACKENDS:
        avail = sorted(f"{r}/{e}" for r, e in _BACKENDS)
        raise ValueError(
            f"No retarget backend registered for robot={robot!r} env={env!r}. "
            f"Available: {avail}"
        )
    spec = _BACKENDS[key]
    mod_name, cls_name = spec.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def supported_robots(env: str | None = None) -> list[str]:
    """List robots registered. Pass `env` to filter; default lists all."""
    if env is None:
        return sorted({r for (r, _) in _BACKENDS})
    return sorted({r for (r, e) in _BACKENDS if e == env})


def supported_envs(robot: str | None = None) -> list[str]:
    """List envs registered. Pass `robot` to filter; default lists all."""
    if robot is None:
        return sorted({e for (_, e) in _BACKENDS})
    return sorted({e for (r, e) in _BACKENDS if r == robot})
